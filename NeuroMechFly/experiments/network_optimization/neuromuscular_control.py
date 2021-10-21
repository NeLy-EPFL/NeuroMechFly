""" Drosophila Simulation for visualization of optimization results."""

import farms_pylog as pylog
import numpy as np
import pybullet as p
from NeuroMechFly.control.spring_damper_muscles import (Parameters,
                                                        SDAntagonistMuscle)
from NeuroMechFly.sdf.sdf import ModelSDF
from NeuroMechFly.sdf.units import SimulationUnitScaling
from NeuroMechFly.simulation.bullet_simulation import BulletSimulation
from shapely.geometry import LinearRing, Point, Polygon

pylog.set_level('error')


class DrosophilaSimulation(BulletSimulation):
    """Drosophila Simulation Class.
        Parameters
        ----------
        container: <Container>
            Instance of the Container class.
        sim_options: <dict>
            Dictionary containing the simulation options.
        units: <obj>
            Instance of SimulationUnitScaling object to scale up the units during calculations.
    """

    def __init__(
        self,
        container,
        sim_options,
        units=SimulationUnitScaling(
            meters=1000,
            kilograms=1000)):
        # Add extra tables to container to store muscle variables and results
        container.add_namespace('muscle')
        container.muscle.add_table('parameters', table_type='CONSTANT')
        container.muscle.add_table('outputs')
        container.muscle.add_table('active_torques')
        container.muscle.add_table('passive_torques')
        # Initialize bullet simulation
        super().__init__(container, units, **sim_options)
        # Parameters
        self.sides = ('L', 'R')
        self.positions = ('F', 'M', 'H')
        self.feet_links = tuple([
            '{}{}Tarsus{}'.format(side, pos, seg)
            for side in self.sides
            for pos in self.positions
            for seg in range(1, 6)
        ])
        _joints = ('Coxa', 'Femur', 'Tibia')
        self.actuated_joints = [
            'joint_{}{}{}'.format(side, pos, joint)
            for side in self.sides
            for pos in self.positions
            for joint in _joints
        ]
        for j, joint in enumerate(self.actuated_joints):
            pos = joint.split('_')[1][1]
            if (pos in ('M', 'H')) and ('Coxa' in joint):
                self.actuated_joints[j] = joint.replace('Coxa', 'Coxa_roll')

        self.num_oscillators = self.controller.graph.number_of_nodes()
        self.active_muscles = {}
        self.neural = self.container.neural
        self.physics = self.container.physics
        self.muscle = self.container.muscle
        # Initialize joint muscles
        for joint in self.actuated_joints:
            fmn = self.neural.states.get_parameter(
                'phase_' + joint + '_flexion')
            emn = self.neural.states.get_parameter(
                'phase_' + joint + '_extension')
            fmn_amp = self.neural.states.get_parameter(
                'amp_' + joint + '_flexion')
            _emn_amp = self.neural.states.get_parameter(
                'amp_' + joint + '_extension')
            jpos = self.physics.joint_positions.get_parameter(joint)
            jvel = self.physics.joint_velocities.get_parameter(joint)
            joint_info = p.getJointInfo(self.animal, self.joint_id[joint])
            _lower_limit = joint_info[8]
            _upper_limit = joint_info[9]
            self.active_muscles[joint] = SDAntagonistMuscle(
                self.container,
                name=joint,
                joint_pos=jpos,
                joint_vel=jvel,
                flexor_mn=fmn,
                extensor_mn=emn,
                flexor_amp=fmn_amp,
                extensor_amp=fmn_amp,
            )

        # Initialize container
        self.container.initialize()

        # Set the physical properties of the environment
        # TODO: Move this to bullet
        dynamics = {
            "lateralFriction": 1.0, "restitution": 0.0, "spinningFriction": 0.0,
            "rollingFriction": 0.0, "linearDamping": 0.0, "angularDamping": 0.0,
            "maxJointVelocity": 1e8
        }
        for _link, idx in self.link_id.items():
            for name, value in dynamics.items():
                p.changeDynamics(self.animal, idx, **{name: value})

        # Debug parameter
        self.draw_ss_line_ids = [
            p.addUserDebugLine(
                (0., 0., 0.), (0., 0., 0.), lineColorRGB=[1, 0, 0]
            )
            for j in range(6)
        ]
        self.draw_ss_horz_line_ids = [
            p.addUserDebugLine(
                (0., 0., 0.), (0., 0., 0.), lineColorRGB=[1, 0, 0]
            )
            for j in range(6)
        ]
        self.draw_com_line_vert_id = p.addUserDebugLine(
            (0., 0., 0.), (0., 0., 0.), lineColorRGB=[1, 0, 0]
        )
        self.draw_com_line_horz_id = p.addUserDebugLine(
            (0., 0., 0.), (0., 0., 0.), lineColorRGB=[1, 0, 0]
        )

        # Data variables
        self.opti_lava = 0
        self.opti_touch = 0
        self.opti_velocity = 0
        self.opti_stability = 0
        self.opti_joint_limit = 0
        self.stance_count = 0
        self.last_draw = []
        self.check_is_all_legs = np.asarray(
            [False for leg in self.feet_links if "Tarsus5" in leg]
        )
        # Read joint limits from the model
        _sdf_model = ModelSDF.read(self.model)[0]
        self.joint_limits = {
            joint.name : joint.axis.limits[:2]
            for joint in _sdf_model.joints
            if joint.axis.limits
        }

    def muscle_controller(self):
        """ Muscle controller. """
        utorque = self.units.torques
        torques = {
            self.joint_id[key] : value.compute_torque(only_passive=False)*utorque
            for key, value in self.active_muscles.items()
        }
        p.setJointMotorControlArray(
            self.animal,
            jointIndices=torques.keys(),
            controlMode=p.TORQUE_CONTROL,
            forces=torques.values()
        )

    def controller_to_actuator(self, t):
        """ Implementation of abstract method. """
        # Update muscles
        self.muscle_controller()

        if t == 2.999 * 1e4:
            average_speed = abs((self.ball_radius * self.units.meters * self.ball_rotations[0]) / 3.0)
            print(f'Fly average speed is: {average_speed} mm/s')
            print(f'Duty factor is: {self.duty_factor}')
        # Change the color of the colliding body segments
        if self.draw_collisions:
            draw = []
            links_contact = np.where(
                np.linalg.norm(
                    self.ground_reaction_forces, axis=1
                ) > 0)[0]
            for i in links_contact:
                link1 = self.ground_contacts[i][:-1]
                if link1 not in draw:
                    draw.append(link1)
                    self.change_color(link1 + '5', self.color_collision)
            for link in self.last_draw:
                if link not in draw:
                    self.change_color(link + '5', self.color_legs)
            self.last_draw = draw

    def change_color(self, identity, color):
        """ Change color of a given body segment. """
        p.changeVisualShape(
            self.animal,
            self.link_id[identity],
            rgbaColor=color)

    def feedback_to_controller(self):
        """ Implementation of abstractmethod. """

    @staticmethod
    def compute_line_coefficients(point_a, point_b):
        """ Compute the coefficient of a line

        Parameters
        ----------
        point_a :<array>
            2D position of a

        point_b :<array>
            2D position of b

        Returns
        -------
        coefficients :<array>

        """
        if abs(point_b[0] - point_a[0]) < 1e-10:
            return np.asarray([1, 0, -point_a[0]])
        slope = (point_b[1] - point_a[1]) / (point_b[0] - point_a[0])
        intercept = point_b[1] - slope * point_b[0]
        return np.asarray([slope, -1, intercept])

    @staticmethod
    def compute_perpendicular_distance(line, point):
        """ Compute the perpendicular distance between line and point

        Parameters
        ----------
        line: <array>
            Coefficients of line segments (ax+by+c)
        point: <array>
            2D point in space

        Returns
        -------
        distance: <float>
            Perpendicular distance from point to line
        """
        return abs(
            line[0] * point[0] + line[1] * point[1] + line[2]
        ) / np.sqrt(line[0]**2 + line[1]**2)

    @staticmethod
    def compute_perpendicular_point(point_a, point_b, point_c):
        """ Compute the perpendicular point between line and point

        Parameters
        ----------
        point_a: <array>
            2D point in space
        point_b: <array>
            2D point in space
        point_c: <array>
            2D point in space

        Returns
        -------
        point_d: <array>
            Perpendicular point from point to line
        """
        x1, y1 = point_a
        x2, y2 = point_b
        x3, y3 = point_c
        dx, dy = x2-x1, y2-y1
        det = dx*dx + dy*dy
        a = (dy*(y3-y1)+dx*(x3-x1))/det
        point_d = np.array([x1+a*dx, y1+a*dy])
        return point_d

    def compute_static_stability(self, draw_polygon=False):
        """ Computes static stability  of the model.

        Parameters
        ----------
        self :

        draw_polygon: <bool>
            Draws the stance polygon and center of mass if True

        Returns
        -------
        static_stability : <float>
            Value of static stability for the fly

        """
        # Initialize static stability
        static_stability = -1.0 # Why 10?
        # Ground contacts
        current_ground_contact_links = self.get_current_contacts()
        contact_points = [
            self.get_link_position(f"{side}Tarsus5")*self.units.meters + [0, 0, .2]
            for side in ("RF", "RM", "RH", "LH", "LM", "LF")
            if any(
                self.link_id[f"{side}Tarsus{num}"] in current_ground_contact_links
                for num in range(1, 6)
            )
        ]
        # compute center of mass of the model
        center_of_mass = self.center_of_mass
        # Update number of legs in stance
        self.stance_count += len(contact_points)
        # Make polygon from contact points
        polygon = Polygon(LinearRing(contact_points)) if (
            len(contact_points) > 2) else Polygon()
        # Get polygon exterior coords
        coords = polygon.exterior.coords
        # Compute distances to COM
        # NOTE : This only works for flat cases. Not for inclined walking
        distances = [
            DrosophilaSimulation.compute_perpendicular_distance(
                DrosophilaSimulation.compute_line_coefficients(
                    coords[idx], coords[idx+1]
                ),
                center_of_mass
            )
            for idx in range(len(coords)-1)
        ]
        # Check if COM is within the polygon
        com_inside = polygon.contains(Point(center_of_mass))
        # Compute static_stability
        static_stability =  np.min(distances) if com_inside else static_stability
        # DEBUG : Drawing
        if draw_polygon:
            # Draw the polygon
            num_coords = len(coords)
            for idx, line_id in enumerate(self.draw_ss_line_ids):
                from_coord, to_coord = (0, 0, 0), (0, 0, 0)
                contact_pos = (0, 0, 0)
                if idx < num_coords-1:
                    from_coord, to_coord = coords[idx], coords[idx+1]
                    contact_pos = contact_points[idx] - [0, 0, .2]
                p.addUserDebugLine(
                    from_coord, to_coord, lineColorRGB=(1,0,0),
                    replaceItemUniqueId=line_id
                )
                # Draw a vertical line from contact point to polygon
                p.addUserDebugLine(
                    contact_pos, from_coord, lineColorRGB=(0,0,1),
                    replaceItemUniqueId=self.draw_ss_horz_line_ids[idx]
                )
            # Draw a vertical line from center of mass
            color = (0, 1, 0) if com_inside else (1, 0, 0)
            p.addUserDebugLine(
                center_of_mass + [0, 0, -1e0],
                center_of_mass + [0, 0, 1e0],
                lineColorRGB=color,
                replaceItemUniqueId=self.draw_com_line_vert_id
            )
            # Draw a horizontal line from com to intersecting line
            if distances:
                p.addUserDebugLine(
                    center_of_mass + [0, 0, -1e0],
                    list(DrosophilaSimulation.compute_perpendicular_point(
                        np.array(coords[int(np.argmin(distances))])[:2],
                        np.array(coords[int(np.argmin(distances))+1])[:2],
                        center_of_mass[:2],
                    )) + [np.array(coords[int(np.argmin(distances))])[-1]],
                    lineColorRGB=(0, 1, 0),
                    replaceItemUniqueId=self.draw_com_line_horz_id
                )
            else:
                p.addUserDebugLine(
                    (0, 0, 0), (0, 0, 0),
                    lineColorRGB=(0, 1, 0),
                    replaceItemUniqueId=self.draw_com_line_horz_id
                )
        return static_stability

    def update_static_stability(self):
        """ Calculates the stability coefficient. """
        self.opti_stability += self.compute_static_stability()

    @property
    def mechanical_work(self):
        """ Mechanical work done by the animal. """
        muscle_torques = np.abs(self.container.muscle.active_torques.log)
        active_joint_ids = [
            self.joint_id[name] for name in self.actuated_joints
        ]
        joint_velocities = np.abs(
            self.container.physics.joint_velocities.log
        )[:, active_joint_ids]
        return self.compute_mechanical_work(joint_velocities, muscle_torques)

    @property
    def thermal_loss(self):
        """ Thermal loss for the animal. """
        muscle_torques = np.array(
            self.container.muscle.active_torques.log
        )
        return self.compute_thermal_loss(muscle_torques)

    @property
    def duty_factor(self):
        contact = self.container.physics.contact_flag.log

        return np.array(
            [
                np.count_nonzero(contact[:, leg_id]) / contact.shape[0]
                for leg_id in range(4,30,5)
            ]
        )


    def check_movement(self):
        """ State of lava approaching the model. """
        # Slow 2 rad (10 mm/sec), fast 6.8 rad (34 mm/sec)
        # The range is 1.2 < ball rotation per second < 7.2 rad/sec
        total_angular_dist = 1.2 * self.run_time
        ball_angular_position = np.array(self.ball_rotations)[0]
        moving_limit_lower = ((self.time / self.run_time)
                        * total_angular_dist) - 0.20
        moving_limit_upper = ((self.time / self.run_time)
                        * 6 * total_angular_dist)
        # print(moving_limit_lower, ball_angular_position, moving_limit_upper)

        self.opti_lava += 1.0 if np.any(
            np.abs(ball_angular_position) < moving_limit_lower
        ) or ball_angular_position < 0 else 0.0

        self.opti_lava += 1.0 if np.any(
            np.abs(ball_angular_position) > moving_limit_upper
        ) or ball_angular_position < 0 else 0.0

    def check_joint_limits(self):
        """ Check if the active exceed joint limits """
        for joint in self.actuated_joints:
            joint_position = self.physics.joint_positions.get_parameter_value(joint)
            limits = self.joint_limits[joint]
            if joint_position < limits[0]:
                self.opti_joint_limit += limits[0]-joint_position
            elif joint_position > limits[1]:
                self.opti_joint_limit += joint_position-limits[1]

    def check_velocity_limit(self):
        """ Check velocity limits. """
        self.opti_velocity += 1.0 if np.any(
            np.array(self.joint_velocities) > 250.0 # Can be changed!!
        ) else 0.0

    def optimization_check(self):
        """ Check the optimization status and update the penalties. """
        self.check_movement()
        self.check_velocity_limit()
        self.update_static_stability()
        self.check_joint_limits()
        return True

    def update_parameters(self, params):
        """ Implementation of abstract method. """
        parameters = self.container.neural.parameters
        n_nodes = int(self.controller.graph.number_of_nodes() / 4)
        # Number of joints, muscle gains, phase variables
        edges_joints = int(self.controller.graph.number_of_nodes() / 3)
        opti_frequency = params[0]
        params = np.delete(params, 0)
        opti_active_muscle_gains = params[:5 * n_nodes]
        opti_joint_phases = params[5 * n_nodes:5 * n_nodes + edges_joints]
        opti_base_phases = params[5 * n_nodes + edges_joints:]

        # Update frequencies
        for name in parameters.names:
            if 'freq' in name:
                parameters.get_parameter(name).value = opti_frequency

        # Update active muscle parameters
        symmetry_joints = filter(
            lambda x: x.split('_')[1][0] != 'R', self.actuated_joints
        )

        for j, joint in enumerate(symmetry_joints):
            right_parameters = Parameters(
                *opti_active_muscle_gains[5 * j:5 * (j + 1)]
            )
            left_parameters = Parameters(
                *opti_active_muscle_gains[5 * j:5 * (j + 1)]
            )
            self.active_muscles[joint.replace('L', 'R', 1)].update_parameters(
                right_parameters
            )
            self.active_muscles[joint].update_parameters(left_parameters)
        # Update phases for intraleg phase relationships
        # Edges to set phases for
        phase_edges = [['Coxa', 'Femur'], ['Femur', 'Tibia']]
        for side in ('L', 'R'):
            for j0, pos in enumerate(('F', 'M', 'H')):
                if pos != 'F':
                    coxa_label = 'Coxa_roll'
                else:
                    coxa_label = 'Coxa'
                for j1, ed in enumerate(phase_edges):
                    if ed[0] == 'Coxa':
                        from_node = coxa_label
                    else:
                        from_node = ed[0]
                    to_node = ed[1]
                    for j2, action in enumerate(('flexion', 'extension')):
                        node_1 = f"joint_{side}{pos}{from_node}_{action}"
                        node_2 = f"joint_{side}{pos}{to_node}_{action}"
                        parameters.get_parameter(
                            f"phi_{node_1}_to_{node_2}"
                        ).value = opti_joint_phases[4 * j0 + 2 * j1 + j2]
                        parameters.get_parameter(
                            f"phi_{node_2}_to_{node_1}"
                        ).value = -1*opti_joint_phases[4 * j0 + 2 * j1 + j2]

        # Update the phases for interleg phase relationships
        coxae_edges = [
            ['LFCoxa', 'RFCoxa'],
            ['LFCoxa', 'RMCoxa_roll'],
            ['RMCoxa_roll', 'LHCoxa_roll'],
            ['RFCoxa', 'LMCoxa_roll'],
            ['LMCoxa_roll', 'RHCoxa_roll']
        ]

        for j1, ed in enumerate(coxae_edges):
            for j2, action in enumerate(('flexion', 'extension')):
                node_1 = f"joint_{ed[0]}_{action}"
                node_2 = f"joint_{ed[1]}_{action}"
                parameters.get_parameter(
                    f"phi_{node_1}_to_{node_2}"
                ).value = opti_base_phases[j1]
                parameters.get_parameter(
                    f"phi_{node_2}_to_{node_1}"
                ).value = -1 * opti_base_phases[j1]

    @staticmethod
    def select_solution(criteria, fun):
        """ Selects a solution given a criteria.

        Parameters
        ----------
        criteria: <str>
            criteria for selecting a solution

        fun: <list>
            Solutions from optimization

        Returns
        -------
        out : Index of the solution fulfilling the criteria

        """
        norm_fun = (fun - np.min(fun, axis=0)) / \
            (np.max(fun, axis=0) - np.min(fun, axis=0))

        if criteria == 'fastest':
            return np.argmin(norm_fun[:, 0])
        if criteria == 'slowest':
            return np.argmax(norm_fun[:, 0])
        if criteria == 'tradeoff':
            return np.argmin(np.sqrt(norm_fun[:, 0]**2 + norm_fun[:, 1]**2))
        if criteria == 'medium':
            mida = mid(norm_fun[:, 0])
            midb = mid(norm_fun[:, 1])
            return np.argmin(
                np.sqrt((norm_fun[:, 0] - mida)**2 + (norm_fun[:, 1] - midb)**2))
        return int(criteria)


def mid(array):
    """ Middle between min and max of array x

    Parameters
    ----------
    array: <array>
        Input array

    Returns
    -------
    out : (max(x)+min(x))*0.5

    """
    return (max(array) + min(array)) * 0.5

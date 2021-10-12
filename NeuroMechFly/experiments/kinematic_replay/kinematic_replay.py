""" Drosophila simulation class for kinematic replay for the ball experiments. """

import numpy as np
import pandas as pd
import pybullet as p

from shapely.geometry import LinearRing, Point, Polygon
from NeuroMechFly.sdf.units import SimulationUnitScaling
from NeuroMechFly.simulation.bullet_simulation import BulletSimulation
from NeuroMechFly.experiments.network_optimization.neuromuscular_control import DrosophilaSimulation as ds


class DrosophilaSimulation(BulletSimulation):
    """ Drosophila Simulation Class for kinematic replay.

    Parameters
    ----------
    container: <Container>
        Instance of the Container class.
    sim_options: <dict>
        Dictionary containing the simulation options.
    kp: <float>
        Proportional gain of the position controller.
    kv: <float>
        Derivative gain of the position controller.
    angles_path: <str>
        Path of the joint position .pkl file.
    velocity_path: <str>
        Path of the joint velocity .pkl file.
    fixed_positions: <dict>
        Dictionary containing the positions for the fixed joints that should be different from the zero pose.
    units: <obj>
        Instance of SimulationUnitScaling object to scale up the units during calculations.
    """

    def __init__(
        self,
        container,
        sim_options,
        kp,
        kv,
        angles_path,
        velocity_path,
        fixed_positions=None,
        units=SimulationUnitScaling(
            meters=1000,
            kilograms=1000)):

        # Add table for mechanical work and thermal loss
        self.analysis_data = container.add_namespace('analysis')
        self.analysis_data.add_table('mechanical_work')
        self.analysis_data.add_table('thermal_loss')
        self.analysis_data.add_table('static_stability')
        self.analysis_data.mechanical_work.add_parameter('mechanical_work')
        self.analysis_data.thermal_loss.add_parameter('thermal_loss')
        self.analysis_data.static_stability.add_parameter('static_stability')

        self.last_draw = []
        self.grf = []
        self.kp = kp
        self.kv = kv
        self.angles_path = angles_path
        self.fixed_positions = fixed_positions
        
        super().__init__(container, units, **sim_options)
        
        self.pose = [0] * self.num_joints
        self.vel = [0] * self.num_joints
        self.angles = self.load_data(angles_path)
        self.velocities = self.load_data(velocity_path)
                    

        # Debug parameter
        self.draw_ss_line_ids = [
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

    @staticmethod
    def load_data(data_path):
        """ Function that loads the pickle format joint angle or velocity gile.

        Parameters
        ----------
        data_path : <str>
            Path of the .pkl file.

        Returns
        -------
        dict
            Returns the joint angles in a dictionary.
        """
        names_equivalence = {
            'ThC_pitch': 'Coxa',
            'ThC_yaw': 'Coxa_yaw',
            'ThC_roll': 'Coxa_roll',
            'CTr_pitch': 'Femur',
            'CTr_roll': 'Femur_roll',
            'FTi_pitch': 'Tibia',
            'TiTa_pitch': 'Tarsus1'
        }
        converted_dict = {}
        try:
            data = pd.read_pickle(data_path)
            for leg, joints in data.items():
                for joint_name, val in joints.items():
                    new_name = 'joint_' + leg[:2] + \
                        names_equivalence[joint_name]
                    converted_dict[new_name] = val
            return converted_dict
        except BaseException:
            FileNotFoundError(f"File {data_path} not found!")

    
    def load_ball_info(self):

        data_path = self.angles_path.replace('joint_angles','treadmill_info')
        
        try:
            data = pd.read_pickle(data_path)
            ball_rad = data['radius']
            ball_pos = data['position']
            
            return ball_rad, ball_pos

        except BaseException:
            FileNotFoundError(f"File {data_path} not found!")
    

    def controller_to_actuator(self, t):
        """
        Code that glues the controller the actuator in the system.
        If there are muscles then contoller actuates the muscles.
        If not then the controller directly actuates the joints.

        Parameters
        ----------
        t : <int>
            Time running in the physics engine.
        """
        
        # Update logs for physical quantities
        self.update_data_logs()
        
        # Setting the joint angular positions of the fixed joints
        if not self.fixed_positions:
            self.fixed_positions = {
                'joint_LAntenna': 35,
                'joint_RAntenna': -35,
            }
        for joint_name, joint_pos in self.fixed_positions.items():
            self.pose[self.joint_id[joint_name]] = np.deg2rad(joint_pos)

        # Setting the joint angular positions of leg DOFs based on pose
        # estimation
        for joint_name, joint_pos in self.angles.items():
            self.pose[self.joint_id[joint_name]] = joint_pos[t+14000]

        # Setting the joint angular velocities of leg DOFs based on pose
        # estimation
        for joint_name, joint_vel in self.velocities.items():
            self.vel[self.joint_id[joint_name]] = joint_vel[t+14000]

        # Control the joints through position controller
        # Velocity can be discarded if not available and gains can be changed
        for joint in range(self.num_joints):
            p.setJointMotorControl2(
                self.animal, joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self.pose[joint],
                targetVelocity=self.vel[joint],
                positionGain=self.kp,
                velocityGain=self.kv,
                maxVelocity=1e8
            )
            p.changeDynamics(self.animal, joint, maxJointVelocity=1e8)

        # Change the color of the colliding body segments
        if self.draw_collisions:
            draw = []
            if self.behavior == 'walking':
                # Only take into account the ground sensors
                #ground_reaction_force = self.contact_normal_force[:len(
                #    self.ground_contacts), :]
                #links_contact = np.where(
                #    np.linalg.norm(
                #        ground_reaction_force,
                #        axis=1) > 0)[0]
                links_contact = self.get_current_contacts()
                link_names = list(self.link_id.keys())
                link_ids = list(self.link_id.values())
                for i in links_contact:
                    link1 = link_names[link_ids.index(i)][:-1]
                    if link1 not in draw:
                        draw.append(link1)
                        self.change_color(link1 + '5', self.color_collision)
                for link in self.last_draw:
                    if link not in draw:
                        self.change_color(link + '5', self.color_legs)

            elif self.behavior == 'grooming':
                # Don't consider the ground sensors
                collision_forces = self.contact_normal_force[len(
                    self.ground_contacts):, :]
                links_contact = np.where(
                    np.linalg.norm(collision_forces, axis=1) > 0
                )[0]
                for i in links_contact:
                    link1 = self.self_collisions[i][0]
                    link2 = self.self_collisions[i][1]
                    if link1 not in draw:
                        draw.append(link1)
                        self.change_color(link1, self.color_collision)
                    if link2 not in draw:
                        draw.append(link2)
                        self.change_color(link2, self.color_collision)
                for link in self.last_draw:
                    if link not in draw:
                        if 'Antenna' in link:
                            self.change_color(link, self.color_body)
                        else:
                            self.change_color(link, self.color_legs)
            self.last_draw = draw

    def change_color(self, identity, color):
        """ Change color of a given body segment. """
        p.changeVisualShape(
            self.animal,
            self.link_id[identity],
            rgbaColor=color)

    def feedback_to_controller(self):
        """
        Code that glues the sensors/feedback to controller in the system.
        """

    def update_parameters(self, params):
        """ Update parameters. """

    def optimization_check(self):
        """ Optimization check. """

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
        static_stability = -10.0  # Why 10?
        # Ground contacts
        current_ground_contact_links = self.get_current_contacts()
        contact_points = [
            self.get_link_position(f"{side}Tarsus5") * self.units.meters + [0, 0, .2]
            for side in ("RF", "RM", "RH", "LH", "LM", "LF")
            if any(
                self.link_id[f"{side}Tarsus{num}"] in current_ground_contact_links
                for num in range(1, 6)
            )
        ]
        contact_points = [[]] if not contact_points else contact_points
        assert len(contact_points) <= 6
        # compute center of mass of the model
        center_of_mass = self.center_of_mass

        # Make polygon from contact points
        # TODO: Refactor
        try:
            polygon = Polygon(LinearRing(contact_points))
        except ValueError:
            return static_stability
        # Get polygon exterior coords
        coords = polygon.exterior.coords
        # Compute distances to COM
        # NOTE : This only works for flat cases. Not for inclined walking
        distances = [
            ds.compute_perpendicular_distance(
                ds.compute_line_coefficients(
                    coords[idx], coords[idx + 1]
                ),
                center_of_mass
            )
            for idx in range(len(coords) - 1)
        ]
        # Check if COM is within the polygon
        com_inside = polygon.contains(Point(center_of_mass))
        # Compute static_stability
        min_distance = np.min(distances)
        static_stability = min_distance if com_inside else -1 * min_distance
        # DEBUG : Drawing
        if draw_polygon:
            # Draw the polygon
            num_coords = len(coords)
            for idx, line_id in enumerate(self.draw_ss_line_ids):
                from_coord, to_coord = (0, 0, 0), (0, 0, 0)
                if idx < num_coords - 1:
                    from_coord, to_coord = coords[idx], coords[idx + 1]
                p.addUserDebugLine(
                    from_coord, to_coord, lineColorRGB=(1, 0, 0),
                    replaceItemUniqueId=line_id
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
            p.addUserDebugLine(
                center_of_mass + [0, 0, -1e0],
                list(ds.compute_perpendicular_point(
                    np.array(coords[int(np.argmin(distances))])[:2],
                    np.array(coords[int(np.argmin(distances)) + 1])[:2],
                    center_of_mass[:2],
                )) + [np.array(coords[int(np.argmin(distances))])[-1]],
                lineColorRGB=(0, 1, 0),
                replaceItemUniqueId=self.draw_com_line_horz_id
            )
        return static_stability

    @property
    def mechanical_work(self):
        """ Mechanical work done by the animal. """
        return self.compute_mechanical_work(
            np.array(self.joint_torques),
            np.array(self.joint_velocities)
        )

    @property
    def thermal_loss(self):
        """ Thermal loss for the animal. """
        return self.compute_thermal_loss(np.array(self.joint_torques))

    def update_data_logs(self):
        """ Update the logs that are implemented in this class. """
        self.analysis_data.mechanical_work.values = np.asarray(self.mechanical_work, dtype='double').reshape((1,))
        self.analysis_data.thermal_loss.values = np.asarray(self.thermal_loss, dtype='double').reshape((1,))
        self.analysis_data.static_stability.values = np.asarray(self.compute_static_stability(), dtype='double').reshape((1,))
        

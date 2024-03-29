""" Class to run animal model. """

import abc
import os
import pkgutil
import time
from pathlib import Path
from PIL import Image
import numpy as np
import pybullet as p
import pybullet_data
import yaml
from farms_network.neural_system import NeuralSystem
from NeuroMechFly.sdf.bullet_load_sdf import load_sdf
from NeuroMechFly.simulation.bullet_sensors import (
    COMSensor, ContactSensors, JointSensors
)
from tqdm import tqdm

neuromechfly_path = Path(pkgutil.get_loader(
    'NeuroMechFly').get_filename()).parents[1]


class BulletSimulation(metaclass=abc.ABCMeta):
    """Methods to run bullet simulation."""

    def __init__(self, container, units, **kwargs):
        super(BulletSimulation, self).__init__()
        self.units = units
        # Simulation options
        self.gui = p.DIRECT if kwargs['headless'] else p.GUI
        self.gravity = np.array(kwargs.get('gravity', [0, 0, -9.81]))
        self.time_step = kwargs.get('time_step', 1e-3) * self.units.seconds
        self.real_time = kwargs.get('real_time', 0)
        self.run_time = kwargs.get('run_time', 10) * self.units.seconds
        self.solver_iterations = kwargs.get('solver_iterations', 1000)
        self.model = kwargs.get('model', None)
        self.model_offset = np.array(
            kwargs.get('model_offset', [0., 0., 0.])
        ) * self.units.meters
        self.num_substep = kwargs.get('num_substep', 1)
        self.ground_contacts = kwargs.get('ground_contacts', ())
        self.base_link = kwargs.get('base_link', None)
        self.controller_config = kwargs.get('controller', None)
        self.pose_file = kwargs.get('pose', None)
        self.muscle_config_file = kwargs.get('muscles', None)
        self.container = container
        self.camera_distance = kwargs.get('camera_distance', 0.1)
        self.track_animal = kwargs.get('track', True)
        self.slow_down = kwargs.get('slow_down', False)
        self.sleep_time = kwargs.get('sleep_time', 0.001)
        self.vis_options_background_color_red = kwargs.get(
            'background_color_red', 1)
        self.vis_options_background_color_green = kwargs.get(
            'background_color_GREEN', 1)
        self.vis_options_background_color_blue = kwargs.get(
            'background_color_BLUE', 1)
        self.record_movie = kwargs.get('record', False)
        self.movie_name = kwargs.get('moviename', 'default_movie.mp4')
        self.movie_speed = kwargs.get('moviespeed', 1)
        self.rotate_camera = kwargs.get('rot_cam', False)
        self.behavior = kwargs.get('behavior', None)
        self.ground = kwargs.get('ground', 'ball')
        # Ball properties
        self.ball_density = kwargs.get(
            'ball_density', 96) * (self.units.kilograms / self.units.volume)  # kg/m^3
        self.ball_radius = kwargs.get(
            'ball_radius', 5.0e-3)  # 1x (real size 10mm)
        self.ball_mass = kwargs.get('ball_mass', 0) * self.units.kilograms
        self.ground_friction_coef = kwargs.get('ground_friction_coef', 10)
        self.enable_concave_mesh = kwargs.get(
            'enable_concave_mesh',
            True if self.behavior == 'grooming' else False
        )
        self.self_collisions = kwargs.get('self_collisions', [])
        self.draw_collisions = kwargs.get('draw_collisions', False)
        self.ball_info = kwargs.get('ball_info', False)
        self.contactERP = kwargs.get('contactERP', 0.1)
        self.globalCFM = kwargs.get('globalCFM', 3.0)
        self.save_frames = kwargs.get('save_frames', False)
        self.path_imgs = kwargs.get(
            'results_path',
            'last_simulation').replace(
            'kinematic_replay_',
            'images_')

        # Init
        self.time = 0.0
        self.floor = None
        self.plane = None
        self.link_plane = None
        self.animal = None
        self.control = None
        self.num_joints = 0
        self.joint_id = {}
        self.joint_type = {}
        self.link_id = {}
        self.joint_sensors = None
        self.contact_sensors = None
        self.com_sensor = None

        # ADD 'Physics' namespace to container
        self.sim_data = self.container.add_namespace('physics')
        # ADD Tables to physics container
        self.sim_data.add_table('base_position')
        self.sim_data.add_table('center_of_mass')
        self.sim_data.add_table('joint_positions')
        self.sim_data.add_table('joint_velocities')
        self.sim_data.add_table('joint_torques')
        self.sim_data.add_table('contact_flag')
        self.sim_data.add_table('contact_position')
        self.sim_data.add_table('contact_normal_force')
        self.sim_data.add_table('contact_lateral_force')
        self.ZEROS_3x1 = np.zeros((3,))

        # Muscles
        self.use_muscles = bool(self.muscle_config_file)

        # Setup
        self.setup_simulation()

        # Enable rendering
        self.rendering(1)

        # Initialize pose
        if self.pose_file:
            self.initialize_position(self.pose_file)

        # Camera
        if self.gui == p.GUI and not self.track_animal:
            base = np.array(self.base_position) * self.units.meters
            p.resetDebugVisualizerCamera(
                self.camera_distance,
                cameraYaw=5,
                cameraPitch=-10,
                cameraTargetPosition=base)

        # Initialize simulation
        self.initialize_simulation()

    def __del__(self):
        print('Simulation has ended')
        p.disconnect()

    @staticmethod
    def rendering(render=1):
        """Enable/disable rendering"""
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, render)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    def setup_simulation(self):
        """ Setup the simulation. """
        ########## PYBULLET SETUP ##########
        if self.record_movie and self.gui == p.GUI:
            p.connect(
                self.gui,
                options=(
                    '--background_color_red={}'
                    ' --background_color_green={}'
                    ' --background_color_blue={}'
                    ' --mp4={}'
                    ' --mp4fps={}'
                ).format(
                    self.vis_options_background_color_red,
                    self.vis_options_background_color_green,
                    self.vis_options_background_color_blue,
                    self.movie_name,
                    int(self.movie_speed / self.time_step)
                )
            )
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        elif self.gui == p.GUI:
            p.connect(
                self.gui,
                options=(
                    ' --background_color_red={}'
                    ' --background_color_green={}'
                    ' --background_color_blue={}'
                ).format(
                    self.vis_options_background_color_red,
                    self.vis_options_background_color_green,
                    self.vis_options_background_color_blue
                )
            )
        else:
            p.connect(self.gui)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Everything should fall down
        p.setGravity(*[g * self.units.gravity for g in self.gravity])

        p.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step,
            numSolverIterations=self.solver_iterations,
            numSubSteps=self.num_substep,
            solverResidualThreshold=1e-10,
            erp=0.0,
            contactERP=self.contactERP,
            frictionERP=0.0,
            globalCFM=self.globalCFM,
            reportSolverAnalytics=1
        )

        # Turn off rendering while loading the models
        self.rendering(0)

        if self.ground == 'floor':
            # Add floor
            self.plane = p.loadURDF(
                'plane.urdf', [0, 0, -0.],
                globalScaling=0.01 * self.units.meters
            )
            # When plane is used the link id is -1
            self.link_plane = -1
            p.changeDynamics(self.plane, -
                             1, lateralFriction=self.ground_friction_coef)
            self.sim_data.add_table('base_linear_velocity')
            self.sim_data.add_table('base_angular_velocity')
            self.sim_data.add_table('base_orientation')
        elif self.ground == 'ball':
            # Add floor and ball
            self.floor = p.loadURDF(
                'plane.urdf', [0, 0, -0.],
                globalScaling=0.01 * self.units.meters
            )

            if self.ball_info:
                self.ball_radius, ball_pos = self.load_ball_info()
            else:
                ball_pos = None

            self.plane = self.add_ball(
                self.ball_radius,
                self.ball_density,
                self.ball_mass,
                self.ground_friction_coef,
                ball_pos
            )

            # When ball is used the plane id is 2 as the ball has 3 links
            self.link_plane = 2
            self.sim_data.add_table('ball_rotations')
            self.sim_data.add_table('ball_velocity')

        # Add the animal model
        if '.sdf' in self.model and self.behavior is not None:
            self.animal, self.link_id, self.joint_id = load_sdf(
                self.model, force_concave=self.enable_concave_mesh
            )
        elif '.sdf' in self.model and self.behavior is None:
            self.animal = p.loadSDF(self.model)[0]
            # Generate joint_name to id dict
            self.link_id[p.getBodyInfo(self.animal)[0].decode('UTF-8')] = -1
            for n in range(p.getNumJoints(self.animal)):
                info = p.getJointInfo(self.animal, n)
                _id = info[0]
                joint_name = info[1].decode('UTF-8')
                link_name = info[12].decode('UTF-8')
                _type = info[2]
                self.joint_id[joint_name] = _id
                self.joint_type[joint_name] = _type
                self.link_id[link_name] = _id
        elif '.urdf' in self.model:
            self.animal = p.loadURDF(self.model)
        p.resetBasePositionAndOrientation(
            self.animal, self.model_offset,
            p.getQuaternionFromEuler([0., 0., 0.]))
        self.num_joints = p.getNumJoints(self.animal)

        # Body colors
        color_wings = [91 / 100, 96 / 100, 97 / 100, 0.7]
        color_eyes = [67 / 100, 21 / 100, 12 / 100, 1]
        self.color_body = [140 / 255, 100 / 255, 30 / 255, 1]
        self.color_legs = [170 / 255, 130 / 255, 50 / 255, 1]
        self.color_collision = [0, 1, 0, 1]
        nospecular = [0.5, 0.5, 0.5]
        # Color the animal
        p.changeVisualShape(
            self.animal, - 1, rgbaColor=self.color_body, specularColor=nospecular)

        for link_name, _id in self.joint_id.items():
            if 'Wing' in link_name and 'Fake' not in link_name:
                p.changeVisualShape(self.animal, _id, rgbaColor=color_wings)
            elif 'Eye' in link_name and 'Fake' not in link_name:
                p.changeVisualShape(self.animal, _id, rgbaColor=color_eyes)
            # and 'Fake' not in link_name:
            elif (
                    'Tarsus' in link_name
                    or 'Tibia' in link_name
                    or 'Femur' in link_name
                    or 'Coxa' in link_name
            ):
                p.changeVisualShape(
                    self.animal,
                    _id,
                    rgbaColor=self.color_legs,
                    specularColor=nospecular)
            elif 'Fake' not in link_name:
                p.changeVisualShape(
                    self.animal,
                    _id,
                    rgbaColor=self.color_body,
                    specularColor=nospecular)

            # print('Link name {} id {}'.format(link_name, _id))

        # Configure contacts

        # Disable/Enable all self-collisions
        for link0 in self.link_id.keys():
            for link1 in self.link_id.keys():
                p.setCollisionFilterPair(
                    bodyUniqueIdA=self.animal,
                    bodyUniqueIdB=self.animal,
                    linkIndexA=self.link_id[link0],
                    linkIndexB=self.link_id[link1],
                    enableCollision=0,
                )

        # Disable/Enable tarsi-ground collisions
        for link in self.link_id.keys():
            if 'Tarsus' in link:
                p.setCollisionFilterPair(
                    bodyUniqueIdA=self.animal,
                    bodyUniqueIdB=self.plane,
                    linkIndexA=self.link_id[link],
                    linkIndexB=self.link_plane,
                    enableCollision=1
                )

        # Disable/Enable selected self-collisions
        for (link0, link1) in self.self_collisions:
            p.setCollisionFilterPair(
                bodyUniqueIdA=self.animal,
                bodyUniqueIdB=self.animal,
                linkIndexA=self.link_id[link0],
                linkIndexB=self.link_id[link1],
                enableCollision=1,
            )

        # ADD container columns
        # ADD ground reaction forces and friction forces
        _ground_sensor_ids = []
        for contact in self.ground_contacts:
            _ground_sensor_ids.append(
                (
                    self.animal, self.plane, self.link_id[contact],
                    self.link_plane
                )
            )
            self.sim_data.contact_flag.add_parameter(
                f"{contact}_flag"
            )
            for axis in ('x', 'y', 'z'):
                self.sim_data.contact_position.add_parameter(
                    contact + '_' + axis)
                self.sim_data.contact_normal_force.add_parameter(
                    contact + '_' + axis)
                self.sim_data.contact_lateral_force.add_parameter(
                    contact + '_' + axis)

        # ADD self collision forces
        _collision_sensor_ids = []
        for link0, link1 in self.self_collisions:
            _collision_sensor_ids.append(
                (
                    self.animal, self.animal, self.link_id[link0],
                    self.link_id[link1]
                )
            )
            contacts = '-'.join((link0, link1))
            for axis in ['x', 'y', 'z']:
                self.sim_data.contact_position.add_parameter(
                    contacts + '_' + axis)
                self.sim_data.contact_normal_force.add_parameter(
                    contacts + '_' + axis)
                self.sim_data.contact_lateral_force.add_parameter(
                    contacts + '_' + axis)

        # Generate sensors
        self.joint_sensors = JointSensors(
            self.animal, self.sim_data, meters=self.units.meters,
            velocity=self.units.velocity, torques=self.units.torques
        )
        self.contact_sensors = ContactSensors(
            self.sim_data,
            tuple([*_ground_sensor_ids, *_collision_sensor_ids]),
            meters=self.units.meters, newtons=self.units.newtons
        )
        self.com_sensor = COMSensor(
            self.animal, self.sim_data, meters=self.units.meters,
            kilograms=self.units.kilograms
        )

        # ADD base position parameters
        for axis in ['x', 'y', 'z']:
            self.sim_data.base_position.add_parameter(f'{axis}')
            # self.sim_data.thorax_force.add_parameter(f'{axis}')
            if self.ground == 'ball':
                self.sim_data.ball_rotations.add_parameter(f'{axis}')
                self.sim_data.ball_velocity.add_parameter(f'{axis}')
            if self.ground == 'floor':
                self.sim_data.base_linear_velocity.add_parameter(f'{axis}')
                self.sim_data.base_angular_velocity.add_parameter(f'{axis}')
                self.sim_data.base_orientation.add_parameter(f'{axis}')

        # ADD joint parameters
        for name, _ in self.joint_id.items():
            self.sim_data.joint_positions.add_parameter(name)
            self.sim_data.joint_velocities.add_parameter(name)
            self.sim_data.joint_torques.add_parameter(name)

        # ADD Center of mass parameters
        for axis in ('x', 'y', 'z'):
            self.sim_data.center_of_mass.add_parameter(f"{axis}")

        # ADD muscles
        if self.use_muscles:
            self.initialize_muscles()

        # ADD controller
        if self.controller_config:
            self.controller = NeuralSystem(
                config_path=self.controller_config,
                container=self.container,
            )

        # Disable default bullet controllers
        p.setJointMotorControlArray(
            self.animal,
            np.arange(self.num_joints),
            p.VELOCITY_CONTROL,
            targetVelocities=np.zeros((self.num_joints, 1)),
            forces=np.zeros((self.num_joints, 1))
        )
        p.setJointMotorControlArray(
            self.animal,
            np.arange(self.num_joints),
            p.POSITION_CONTROL,
            forces=np.zeros((self.num_joints, 1))
        )
        p.setJointMotorControlArray(
            self.animal,
            np.arange(self.num_joints),
            p.TORQUE_CONTROL,
            forces=np.zeros((self.num_joints, 1))
        )

        # Disable link linear and angular damping
        for njoint in range(self.num_joints):
            p.changeDynamics(self.animal, njoint, linearDamping=0.0)
            p.changeDynamics(self.animal, njoint, angularDamping=0.0)
            p.changeDynamics(self.animal, njoint, jointDamping=0.0)

        self.total_mass = 0.0

        for j in np.arange(-1, p.getNumJoints(self.animal)):
            self.total_mass += p.getDynamicsInfo(
                self.animal, j)[0] / self.units.kilograms

        self.bodyweight = -1 * self.total_mass * self.gravity[2]
        print('Total mass = {}'.format(self.total_mass))

        if self.gui == p.GUI:
            self.rendering(1)

    def set_collisions(self, links, group=1, mask=1):
        """Activate/Deactivate leg collisions"""
        for link in links:
            p.setCollisionFilterGroupMask(
                bodyUniqueId=self.animal,
                linkIndexA=self.link_id[link],
                collisionFilterGroup=group,
                collisionFilterMask=mask
            )

    def set_collisions_whole_body(self, group=1, mask=1):
        """Activate/Deactivate leg collisions"""
        for link in range(-1, p.getNumJoints(self.animal)):
            p.setCollisionFilterGroupMask(
                bodyUniqueId=self.animal,
                linkIndexA=link,
                collisionFilterGroup=group,
                collisionFilterMask=mask
            )

    def initialize_simulation(self):
        """ Initialize simulation. """
        # Initialize the container
        self.container.initialize()

        # Setup the integrator
        if self.controller_config:
            self.controller.setup_integrator()
        if self.use_muscles:
            self.muscles.setup_integrator()

        # Activate the force/torque sensor
        #self.thorax_id = self.link_id['Thorax']
        # for joint_id in range(0, self.num_joints):
        #     p.enableJointForceTorqueSensor(
        #         self.animal,
        #         joint_id,
        #         True
        #     )

    def initialize_muscles(self):
        """ Initialize the muscles of the animal. """
        self.muscles = MusculoSkeletalSystem(self.muscle_config_file)

    def initialize_position(self, pose_file=None):
        """Initialize the pose of the animal.
        Parameters:
        pose_file : <selftr>
             File path to the pose data
        """
        if pose_file:
            try:
                with open(pose_file) as stream:
                    data = yaml.load(stream, Loader=yaml.SafeLoader)
                    data = {k.lower(): v for k, v in data.items()}
            except FileNotFoundError:
                print('Pose file {} not found'.format(pose_file))
                return
            for joint, _id in self.joint_id.items():
                _pose = np.deg2rad(data['joints'].get(joint, 0))
                p.resetJointState(
                    self.animal, _id,
                    targetValue=_pose
                )
        else:
            return None

    def get_current_contacts(self):
        """ Check for ground contact """
        contacts = np.argwhere(self.sim_data.contact_flag.values).flatten()
        contact_links_ids = tuple([
            self.contact_sensors.contact_ids[index][2]
            for index in contacts
        ])
        return contact_links_ids

    def get_link_position(self, link_name):
        """' Return the position of the link. """
        return np.array((p.getLinkState(
            self.animal,
            self.link_id[link_name]))[0]) / self.units.meters

    def add_ball(self, radius, density, mass, ball_friction_coef, position):
        """ Create a ball with specified radius """
        ball_radius = radius * self.units.meters
        volume = 4 / 3 * np.pi * ball_radius**3
        calculated_mass = density * volume
        # Assert if calculated and measured ball mass are not the same

        if mass != 0:
            # TODO: Decide the threshold here, it is 8 mg now
            assert abs(
                mass - calculated_mass) < 8.0e-6 * self.units.kilograms, "Calculated ({} kg) and measured ({} kg) ball masses do not match!".format(
                calculated_mass / self.units.kilograms, mass / self.units.kilograms)
        else:
            mass = calculated_mass

        col_sphere_parent = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=ball_radius / 100,
        )
        col_sphere_id = p.createCollisionShape(
            p.GEOM_SPHERE, radius=ball_radius)

        mass_parent = 0
        visual_shape_id = -1

        #: Different ball positions used for different experiments
        #: Else corresponds to the ball position during optimization
        if position is None:
            if self.behavior == 'walking':
                base_position = np.array(
                    [0.28e-3, -0.2e-3, -4.965e-3]
                ) * self.units.meters + self.model_offset
            elif self.behavior == 'grooming':
                base_position = np.array(
                    [0.1e-3, -0.0e-3, -4.9e-3]
                ) * self.units.meters + self.model_offset
            else:
                base_position = np.array(
                    [-0.09e-3, -0.0e-3, -5.11e-3]
                ) * self.units.meters + self.model_offset
        else:
            base_position = np.array(position) * \
                self.units.meters + self.model_offset
            print("Adding ball position from file:", base_position)
            print("Adding ball radius from file:", ball_radius)

        # Create the sphere
        base_orientation = [0, 0, 0, 1]
        link_masses = np.array([0, 0, mass])
        link_collision_shape_indices = [-1, -1, col_sphere_id]
        link_visual_shape_indices = [-1, -1, -1]
        link_positions = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        link_orientations = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
        link_inertial_frame_positions = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        link_inertial_frame_orientations = [
            [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
        indices = [0, 1, 2]
        joint_types = [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]
        axis = [[0, -1, 0], [1, 0, 0], [0, 0, -1]]

        sphere_id = p.createMultiBody(
            mass_parent,
            col_sphere_parent,
            visual_shape_id,
            base_position,
            base_orientation,
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_shape_indices,
            linkVisualShapeIndices=link_visual_shape_indices,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_frame_positions,
            linkInertialFrameOrientations=link_inertial_frame_orientations,
            linkParentIndices=indices,
            linkJointTypes=joint_types,
            linkJointAxis=axis)

        # Physical properties of the ball can be changed here individually
        inertia_sim = p.getDynamicsInfo(sphere_id, 2)[2]
        inertia_th = [2 / 5 * link_masses[-1] * (ball_radius**2)] * 3

        p.changeDynamics(sphere_id, 2, restitution=0.0)
        p.changeDynamics(sphere_id, 2, lateralFriction=ball_friction_coef)
        p.changeDynamics(sphere_id, 2, spinningFriction=0.0)
        p.changeDynamics(sphere_id, 2, linearDamping=0.0)
        p.changeDynamics(sphere_id, 2, angularDamping=0.0)
        p.changeDynamics(sphere_id, 2, rollingFriction=0.0)

        # Assert if theoretical and computed inertia values are not the same
        assert any([np.isclose(s, t) for s, t in zip(inertia_sim, inertia_th)]), \
            'Theoretical inertia ({}) does not match with the simulation result ({})!'.format(
            inertia_th, inertia_sim
        )

        # Disable default bullet controllers
        p.setJointMotorControlArray(
            sphere_id,
            np.arange(3),
            p.VELOCITY_CONTROL,
            targetVelocities=np.zeros((3, 1)),
            forces=np.zeros((3, 1))
        )
        p.setJointMotorControlArray(
            sphere_id,
            np.arange(3),
            p.POSITION_CONTROL,
            forces=np.zeros((3, 1))
        )
        p.setJointMotorControlArray(
            sphere_id,
            np.arange(3),
            p.TORQUE_CONTROL,
            forces=np.zeros((3, 1))
        )

        texture_path = os.path.join(
            neuromechfly_path,
            'data/design/textures/ball/chequered_0048.jpg',
        )
        texture_ball = p.loadTexture(texture_path)
        p.changeVisualShape(
            sphere_id,
            2,
            rgbaColor=[
                225 / 255,
                225 / 255,
                210 / 255,
                1],
            specularColor=[
                0,
                0,
                0],
            textureUniqueId=texture_ball
        )

        return sphere_id

    @property
    def ball_rotations(self):
        """ Return the ball angular position. """
        return tuple(
            state[0] for state in p.getJointStates(
                self.plane,
                np.arange(0, p.getNumJoints(self.plane))
            )
        )

    @property
    def ball_velocity(self):
        """ Return the ball angular velocity. """
        return tuple(
            p.getLinkState(
                self.plane,
                2,
                computeLinkVelocity=1)[7])

    @property
    def base_linear_velocity(self):
        """ Return the base world linear velocity. """
        imeter = 1. / self.units.meters
        if self.base_link and self.link_id[self.base_link] != -1:
            link_id = self.link_id[self.base_link]
            return np.array(p.getLinkState(
                self.animal,
                self.link_id[self.base_link],
                computeLinkVelocity=1)[6]) * imeter
        else:
            return np.array(p.getBaseVelocity(self.animal)[0]) * imeter

    @property
    def base_angular_velocity(self):
        """ Return the base world angular velocity. """
        if self.base_link and self.link_id[self.base_link] != -1:
            link_id = self.link_id[self.base_link]
            return np.array(p.getLinkState(
                self.animal,
                self.link_id[self.base_link],
                computeLinkVelocity=1)[7])
        else:
            return np.array(p.getBaseVelocity(self.animal)[1])

    @property
    def base_position(self):
        """ Get the position of the animal  """
        imeter = 1. / self.units.meters
        if self.base_link and self.link_id[self.base_link] != -1:
            link_id = self.link_id[self.base_link]
            return np.array((p.getLinkState(self.animal, link_id))[
                            0]) * imeter
        else:
            return np.array(
                (p.getBasePositionAndOrientation(
                    self.animal))[0]) * imeter

    @property
    def base_orientation(self):
        """ Get the position of the animal  """
        if self.base_link and self.link_id[self.base_link] != -1:
            link_id = self.link_id[self.base_link]
            orientation = np.array((p.getLinkState(self.animal, link_id))[
                1])
        else:
            orientation = np.array(
                (p.getBasePositionAndOrientation(
                    self.animal))[1])
        return p.getEulerFromQuaternion(orientation)

    @property
    def joint_positions(self):
        """ Get the joint positions in the animal  """
        return np.asarray(
            self.sim_data.joint_positions.values
        )

    @property
    def joint_torques(self):
        """ Get the joint torques in the animal  """
        return np.asarray(
            self.sim_data.joint_torques.values
        )

    @property
    def joint_velocities(self):
        """ Get the joint velocities in the animal  """
        return np.asarray(
            self.sim_data.joint_velocities.values
        )

    @property
    def contact_position(self):
        """ Get the contact points of collision. """
        return np.asarray(
            self.sim_data.contact_position.values).reshape(
            (-1, 3))

    @property
    def contact_normal_force(self):
        """ Get the ground reaction forces between the surface. """
        return np.asarray(
            self.sim_data.contact_normal_force.values).reshape(
            (-1, 3))

    @property
    def contact_lateral_force(self):
        """ Get the ground reaction forces between the surface. """
        return np.asarray(
            self.sim_data.contact_lateral_force.values).reshape(
            (-1, 3))

    @property
    def distance_z(self):
        """ Distance the animal has travelled in z-direction. """
        return self.base_position[2] / self.units.meters

    @property
    def center_of_mass(self):
        """ Compute the center of mass  """
        return np.array(self.sim_data.center_of_mass.values)

    def compute_mechanical_work(self, joint_velocities, joint_torques):
        """ Computes the mechanical work spent by the animal. """
        return np.abs(
            joint_torques@joint_velocities.T
        ) * self.time_step / self.run_time

    def compute_thermal_loss(self, joint_torques):
        """ Computes the thermal loss exerted by the animal. """
        return np.sum(
            np.sum(joint_torques**2)
        ) * self.time_step / self.run_time

    def update_logs(self):
        """ Update all the physics logs. """
        self.sim_data.base_position.values = np.asarray(self.base_position)
        # Update sensors
        self.joint_sensors.update()
        self.contact_sensors.update()
        self.com_sensor.update()
        if self.ground == 'ball':
            self.sim_data.ball_rotations.values = np.asarray(
                self.ball_rotations).flatten()
            self.sim_data.ball_velocity.values = np.asarray(
                self.ball_velocity).flatten()
        if self.ground == 'floor':
            self.sim_data.base_linear_velocity.values = np.asarray(
                self.base_linear_velocity).flatten()
            self.sim_data.base_angular_velocity.values = np.asarray(
                self.base_angular_velocity).flatten()
            self.sim_data.base_orientation.values = np.asarray(
                self.base_orientation)

    @abc.abstractmethod
    def controller_to_actuator(self):
        """
        Code that glues the controller the actuator in the system.
        If there are muscles then contoller actuates the muscles.
        If not then the controller directly actuates the joints

        Parameters
        ----------
        None

        Returns
        -------
        out :

        """
        pass

    @abc.abstractmethod
    def feedback_to_controller(self):
        """
        Code that glues the sensors/feedback to controller in the system.

        Parameters
        ----------
        None

        Returns
        -------
        out:
        """
        pass

    @abc.abstractmethod
    def update_parameters(self, params):
        """ Update parameters. """
        pass

    @abc.abstractmethod
    def optimization_check(self):
        """ Optimization check. """
        pass

    def step(self, t, optimization=False):
        """ Step the simulation.

        Returns
        -------
        out :
        """
        #base = np.array(self.base_position) * self.units.meters

        # Camera
        if self.gui == p.GUI and self.track_animal:
            base = np.array(self.base_position) * self.units.meters
            base[2] = self.model_offset[2]
            yaw = 30
            pitch = -10
            p.resetDebugVisualizerCamera(
                self.camera_distance, yaw, pitch, base
            )

        # Walking camera sequence, set rotate_camera to True to activate
        if self.gui == p.GUI and self.rotate_camera and self.behavior == 'walking':
            base = np.array(self.base_position) * self.units.meters

            if t < 3 / self.time_step:
                yaw = 0
                pitch = -10
            elif t >= 3 / self.time_step and t < 4 / self.time_step:
                yaw = (t - (3 / self.time_step)) / (1 / self.time_step) * 90
                pitch = -10
            elif t >= 4 / self.time_step and t < 4.25 / self.time_step:
                yaw = 90
                pitch = -10
            elif t >= 4.25 / self.time_step and t < 4.75 / self.time_step:
                yaw = 90
                pitch = (t - (4.25 / self.time_step)) / \
                    (0.5 / self.time_step) * 70 - 10
            elif t >= 4.75 / self.time_step and t < 5 / self.time_step:
                yaw = 90
                pitch = 60
            elif t >= 5 / self.time_step and t < 5.5 / self.time_step:
                yaw = 90
                pitch = 60 - (t - (5 / self.time_step)) / \
                    (0.5 / self.time_step) * 70
            elif t >= 5.5 / self.time_step and t < 7 / self.time_step:
                yaw = (t - (5.5 / self.time_step)) / \
                    (1.5 / self.time_step) * 300 + 90
                pitch = -10
            else:
                yaw = 30
                pitch = -10
            p.resetDebugVisualizerCamera(
                self.camera_distance,
                yaw,
                pitch,
                base)

        # Grooming camera sequence, set rotate_camera to True to activate
        if self.gui == p.GUI and self.rotate_camera and self.behavior == 'grooming':
            base = np.array(self.base_position) * self.units.meters
            if t < 0.25 / self.time_step:
                yaw = 0
                pitch = -10
            elif t >= 0.25 / self.time_step and t < 2.0 / self.time_step:
                yaw = (t - (0.25 / self.time_step)) / \
                    (1.75 / self.time_step) * 150
                pitch = -10
            elif t >= 2.0 / self.time_step and t < 3.5 / self.time_step:
                yaw = 150 - (t - (2.0 / self.time_step)) / \
                    (1.5 / self.time_step) * 120
                pitch = -10
            else:
                yaw = 30
                pitch = -10
            p.resetDebugVisualizerCamera(
                self.camera_distance,
                yaw,
                pitch,
                base)

        if self.gui == p.GUI and self.rotate_camera and self.behavior is None:
            base = np.array(self.base_position) * self.units.meters
            yaw = (t - (self.run_time / self.time_step)) / \
                (self.run_time / self.time_step) * 360
            pitch = -10
            p.resetDebugVisualizerCamera(
                self.camera_distance,
                yaw,
                pitch,
                base)

        if self.save_frames:
            if self.gui == p.DIRECT:
                base = np.array(self.base_position) * self.units.meters
                matrix = p.computeViewMatrixFromYawPitchRoll(
                    base, self.camera_distance, 5, -10, 0, 2
                )
                projectionMatrix = [1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0,
                                    0.0, 0.0, 0.0, -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0]
                img = p.getCameraImage(1024,
                                       768,
                                       viewMatrix=matrix,
                                       projectionMatrix=projectionMatrix)
            if self.gui == p.GUI:
                img = p.getCameraImage(
                    1024, 768, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            rgb_array = img[2]
            im = Image.fromarray(rgb_array)

            im_name = f"{self.path_imgs}/Frame_{t:06d}.png"
            if not os.path.exists(self.path_imgs):
                os.mkdir(self.path_imgs)

            im.save(im_name)

            # disable rendering temporary makes adding objects faster
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

        # Update logs
        self.update_logs()
        # Update container log
        self.container.update_log()
        # Update the feedback to controller
        self.feedback_to_controller()
        # Step controller
        if self.controller_config:
            self.controller.step(self.time_step)
        # Update the controller_to_actuator
        self.controller_to_actuator(t)
        # Step muscles
        if self.use_muscles:
            self.muscles.step()
        # Step time
        self.time += self.time_step
        # Step physics
        solver = p.stepSimulation()

        # Slow down the simulation
        if self.slow_down:
            time.sleep(self.sleep_time)
        # Check if optimization is to be killed
        if optimization:
            optimization_status = self.optimization_check()
            return optimization_status
        return True

    def run(self, optimization=False):
        """ Run the full simulation. """
        total = int(self.run_time / self.time_step)
        for t in tqdm(range(0, total), disable=optimization):
            status = self.step(t, optimization=optimization)
            if not status:
                return False

""" Class to run animal model. """

import abc
import time
from pathlib import Path
import pkgutil
import os

import numpy as np
import pybullet as p
import pybullet_data
import yaml
from farms_network.neural_system import NeuralSystem
from NeuroMechFly.sdf.bullet_load_sdf import load_sdf
from tqdm import tqdm


neuromechfly_path = Path(pkgutil.get_loader('NeuroMechFly').get_filename()).parents[1]


class BulletSimulation(metaclass=abc.ABCMeta):
    """Methods to run bullet simulation."""

    def __init__(self, container, units, **kwargs):
        super(BulletSimulation, self).__init__()
        self.units = units
        #: Simulation options
        self.gui = p.DIRECT if kwargs['headless'] else p.GUI
        self.gravity = np.array(kwargs.get('gravity', [0, 0, -9.81]))
        self.time_step = kwargs.get('time_step', 0.001) * self.units.seconds
        self.real_time = kwargs.get('real_time', 0)
        self.run_time = kwargs.get('run_time', 10) * self.units.seconds
        self.solver_iterations = kwargs.get('solver_iterations', 50)
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
        self.enable_concave_mesh = kwargs.get(
            'enable_concave_mesh',
            True if self.behavior == 'grooming' else False
        )
        self.self_collisions = kwargs.get('self_collisions', [])
        self.draw_collisions = kwargs.get('draw_collisions', False)

        #: Init
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
        self.ground_sensors = {}
        self.collision_sensors = {}

        #: ADD 'Physics' namespace to container
        self.sim_data = self.container.add_namespace('physics')
        #: ADD Tables to physics container
        self.sim_data.add_table('base_position')
        self.sim_data.add_table('joint_positions')
        self.sim_data.add_table('joint_velocities')
        self.sim_data.add_table('joint_torques')
        self.sim_data.add_table('collision_forces')
        self.sim_data.add_table('ground_contacts')
        self.sim_data.add_table('ground_friction_dir1')
        self.sim_data.add_table('ground_friction_dir2')
        self.ZEROS_3x1 = np.zeros((3,))

        #: Muscles
        self.use_muscles = bool(self.muscle_config_file)

        #: Setup
        self.setup_simulation()

        #: Enable rendering
        self.rendering(1)

        # Initialize pose
        if self.pose_file:
            self.initialize_position(self.pose_file)

        #: Camera
        if self.gui == p.GUI and not self.track_animal:
            base = np.array(self.base_position) * self.units.meters
            p.resetDebugVisualizerCamera(
                self.camera_distance,
                5,
                -10,
                base)

        #: Initialize simulation
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
                    int(self.movie_speed/self.time_step)
                )
            )
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
        elif self.gui == p.GUI:
            p.connect(
                self.gui,
                options=(
                    '--background_color_red={}'
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

        #: Everything should fall down
        p.setGravity(*[g * self.units.gravity for g in self.gravity])

        p.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step,
            numSolverIterations=100,
            numSubSteps=self.num_substep,
            solverResidualThreshold=1e-10,
            #erp = 1e-1,
            contactERP=0.1,
            frictionERP=0.0,
        )

        #: Turn off rendering while loading the models
        self.rendering(0)

        if self.ground == 'floor':
            #: Add floor
            self.plane = p.loadURDF(
                'plane.urdf', [0, 0, -0.],
                globalScaling=0.01 * self.units.meters
            )
            #: When plane is used the link id is -1
            self.link_plane = -1
        elif self.ground == 'ball':
            #: Add floor and ball
            self.floor = p.loadURDF(
                'plane.urdf', [0, 0, -0.],
                globalScaling=0.01 * self.units.meters
            )
            self.ball_radius = 5e-3 * self.units.meters  # 1x (real size 10mm)
            self.plane = self.add_ball(self.ball_radius)
            #: When ball is used the plane id is 2 as the ball has 3 links
            self.link_plane = 2
            self.sim_data.add_table('ball_rotations')

        #: Add the animal model
        if '.sdf' in self.model:
            self.animal, self.link_id, self.joint_id = load_sdf(
                self.model, force_concave=self.enable_concave_mesh
            )
        elif '.urdf' in self.model:
            self.animal = p.loadURDF(self.model)
        p.resetBasePositionAndOrientation(
            self.animal, self.model_offset,
            p.getQuaternionFromEuler([0., 0., 0.]))
        self.num_joints = p.getNumJoints(self.animal)

        #: Body colors
        color_wings = [91 / 100, 96 / 100, 97 / 100, 0.7]
        color_eyes = [67 / 100, 21 / 100, 12 / 100, 1]
        self.color_body = [140 / 255, 100 / 255, 30 / 255, 1]
        self.color_legs = [170 / 255, 130 / 255, 50 / 255, 1]
        self.color_collision = [0, 1, 0, 1]
        nospecular = [0.5, 0.5, 0.5]
        #: Color the animal
        p.changeVisualShape(self.animal, -
                            1, rgbaColor=self.color_body, specularColor=nospecular)
        
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

            #print('Link name {} id {}'.format(link_name, _id))

        #: Configure contacts

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

        #: ADD container columns

        #: ADD ground reaction forces and friction forces
        for contact in self.ground_contacts:
            self.ground_sensors[contact] = self.link_id[contact]
            for axis in ['x', 'y', 'z']:
                self.sim_data.ground_contacts.add_parameter(
                    contact + '_' + axis)
                self.sim_data.ground_friction_dir1.add_parameter(
                    contact + '_' + axis)
                self.sim_data.ground_friction_dir2.add_parameter(
                    contact + '_' + axis)

        #: ADD self collision forces
        for link0, link1 in self.self_collisions:
            contact_links = '-'.join((link0, link1))
            self.collision_sensors[contact_links] = [
                self.link_id[link0], self.link_id[link1]]
            for axis in ['x', 'y', 'z']:
                self.sim_data.collision_forces.add_parameter(contact_links + '_' + axis)

        #: ADD base position parameters
        for axis in ['x', 'y', 'z']:
            self.sim_data.base_position.add_parameter(f'{axis}')
            #self.sim_data.thorax_force.add_parameter(f'{axis}')
            if self.ground == 'ball':
                self.sim_data.ball_rotations.add_parameter(f'{axis}')

        #: ADD joint parameters
        for name, _ in self.joint_id.items():
            self.sim_data.joint_positions.add_parameter(name)
            self.sim_data.joint_velocities.add_parameter(name)
            self.sim_data.joint_torques.add_parameter(name)

        #: ADD muscles
        if self.use_muscles:
            self.initialize_muscles()

        #: ADD controller
        if self.controller_config:
            self.controller = NeuralSystem(
                config_path=self.controller_config,
                container=self.container,
            )

        #: DIisable default bullet controllers

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
        #: Initialize the container
        self.container.initialize()

        #: Setup the integrator
        if self.controller_config:
            self.controller.setup_integrator()
        if self.use_muscles:
            self.muscles.setup_integrator()

        #: Activate the force/torque sensor
        #self.thorax_id = self.link_id['Thorax']
        #p.enableJointForceTorqueSensor(self.animal, self.thorax_id, True)

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

    def _get_contact_normal_force(self, link_id):
        """ Compute ground reaction force. """
        contacts = p.getContactPoints(
            self.animal, self.plane,
            link_id, self.link_plane
        )
        self.contact_pos = np.sum(
            [pt[5] for pt in contacts],
            axis=0,
        ) / len(contacts) if contacts else self.ZEROS_3x1
        self.normal_dir = -1 * np.sum(
            [pt[7]for pt in contacts],
            axis=0,
        ) / len(contacts) if contacts else self.ZEROS_3x1
        self.normal = np.sum(
            [pt[9]for pt in contacts],
            axis=0,
        ) if contacts else self.ZEROS_3x1
        contact_normal_force = self.normal * self.normal_dir
        return contact_normal_force / self.units.newtons

    def _get_lateral_friction_force_dir1(self, link_id):
        """ Compute lateral friction force along direction 1. """
        contacts = p.getContactPoints(
            self.animal, self.plane,
            link_id, self.link_plane,
        )
        lateral_friction_force = np.sum(
            [pt[10] * np.asarray(pt[11]) for pt in contacts],
            axis=0,
        ) / len(contacts) if contacts else self.ZEROS_3x1
        return lateral_friction_force / self.units.newtons

    def _get_lateral_friction_force_dir2(self, link_id):
        """ Compute lateral friction force along direction 2. """
        contacts = p.getContactPoints(
            self.animal, self.plane,
            link_id, self.link_plane,
        )
        lateral_friction_force = np.sum(
            [pt[12] * np.asarray(pt[13]) for pt in contacts],
            axis=0,
        ) / len(contacts) if contacts else self.ZEROS_3x1
        return lateral_friction_force / self.units.newtons

    def _get_contact_force_self_collisions(self, link_ids):
        """ Compute self collision forces. """
        contacts = p.getContactPoints(
            self.animal, self.animal,
            link_ids[0], link_ids[1])
        self.contact_pos = (
            np.sum([pt[5] for pt in contacts], axis=0) / len(contacts)
            if contacts
            else self.ZEROS_3x1
        )
        self.normal_dir = (
            1 * np.sum([pt[7]for pt in contacts], axis=0) / len(contacts)
            if contacts
            else self.ZEROS_3x1
        )
        self.normal = (
            np.sum([pt[9]for pt in contacts], axis=0)
            if contacts
            else self.ZEROS_3x1
        )
        collision_force = self.normal * self.normal_dir
        return collision_force / self.units.newtons

    def is_contact(self, link_name):
        """ Check if link is in contact with floor or ball. """
        return bool(p.getContactPoints(
            self.animal, self.plane,
            self.link_id[link_name],
            self.link_plane
        ))

    def get_link_position(self, link_name):
        """' Return the position of the link. """
        return np.array((p.getLinkState(
            self.animal,
            self.link_id[link_name]))[0]) / self.units.meters

    def add_ball(self, radius):
        """ Create a ball with specified radius """
        col_sphere_parent = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius / 100,
        )
        col_sphere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)

        mass_parent = 0
        visual_shape_id = -1
        #: Different ball positions used for different experiments
        #: Else corresponds to the ball position during optimization
        if self.behavior == 'walking':
            link_masses = np.array(
                [1e-11,1e-11,1e-11]
                )*self.units.kilograms
            base_position = np.array([0.28e-3, -0.2e-3,-4.965e-3])*self.units.meters+self.model_offset
        elif self.behavior == 'grooming':
            link_masses = np.array(
                [1e-11,1e-11,1e-11]
                )*self.units.kilograms
            base_position = np.array(
                [0.0e-3, 0.0e-3, -5e-3]
            ) * self.units.meters + self.model_offset
        else:
            link_masses = np.array(
                [5e-11,5e-11,5e-11]
                )*self.units.kilograms
            base_position = np.array(
                [-0.09e-3, -0.0e-3,-5.13e-3]
            ) * self.units.meters + self.model_offset
        #: Create the sphere
        base_orientation = [0, 0, 0, 1]
        link_collision_shape_indices = [-1, -1, col_sphere_id]
        link_visual_shape_indices = [-1, -1, -1]
        link_positions = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        link_orientations = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
        link_inertial_frame_positions = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        link_inertial_frame_orientations = [
            [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
        indices = [0, 1, 2]
        joint_types = [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]
        axis = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]

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
        #: Physical properties of the ball can be changed here
        p.changeDynamics(sphere_id,
                         -1,
                         spinningFriction=100,
                         lateralFriction=1.0,
                         linearDamping=0.0,
                         restitution=1.0)
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
        """ Return the ball position. """
        return tuple(
            state[0] for state in p.getJointStates(
                self.plane,
                np.arange(0, p.getNumJoints(self.plane))
            )
        )

    @property
    def joint_states(self):
        """ Get all joint states. """
        return p.getJointStates(
            self.animal,
            np.arange(0, p.getNumJoints(self.animal))
        )

    @property
    def ground_reaction_forces(self):
        """Get the ground reaction forces. """
        return list(
            map(self._get_contact_normal_force, self.ground_sensors.values())
        )

    @property
    def ground_lateral_friction_dir1(self):
        """Get the ball friction forces along direction 1.  """
        return list(map(self._get_lateral_friction_force_dir1,
                        self.ground_sensors.values()))

    @property
    def ground_lateral_friction_dir2(self):
        """Get the ball friction forces along direction 2.  """
        return list(map(self._get_lateral_friction_force_dir2,
                        self.ground_sensors.values()))

    @property
    def collision_forces(self):
        """ Get collision forces between limb segments. """
        return list(map(self._get_contact_force_self_collisions,
                        self.collision_sensors.values()))

    @property
    def base_position(self):
        """ Get the position of the animal  """
        if self.base_link and self.link_id[self.base_link] != -1:
            link_id = self.link_id[self.base_link]
            return np.array((p.getLinkState(self.animal, link_id))[
                            0]) / self.units.meters
        else:
            return np.array(
                (p.getBasePositionAndOrientation(
                    self.animal))[0]) / self.units.meters

    @property
    def joint_positions(self):
        """ Get the joint positions in the animal  """
        return tuple(
            state[0] for state in p.getJointStates(
                self.animal,
                np.arange(0, p.getNumJoints(self.animal))
            )
        )

    @property
    def joint_torques(self):
        """ Get the joint torques in the animal  """
        _joints = np.arange(0, p.getNumJoints(self.animal))
        return tuple(
            state[-1] / self.units.torques for state in p.getJointStates(
                self.animal, _joints)
        )

    @property
    def joint_velocities(self):
        """ Get the joint velocities in the animal  """
        return tuple(
            state[1] for state in p.getJointStates(
                self.animal,
                np.arange(0, p.getNumJoints(self.animal))
            )
        )

    @property
    def distance_x(self):
        """ Distance the animal has travelled in x-direction. """
        return self.base_position[0] / self.units.meters

    @property
    def distance_y(self):
        """ Distance the animal has travelled in y-direction. """
        return -self.base_position[1] / self.units.meters

    @property
    def distance_z(self):
        """ Distance the animal has travelled in z-direction. """
        return self.base_position[2] / self.units.meters

    @property
    def mechanical_work(self):
        """ Mechanical work done by the animal. """
        return np.sum(np.sum(
            np.abs(np.asarray(self.sim_data.joint_torques.log)
                   * np.asarray(self.sim_data.joint_velocities.log))
        )) * self.time_step / self.run_time

    @property
    def thermal_loss(self):
        """ Thermal loss for the animal. """
        return np.sum(np.sum(
            np.asarray(self.sim_data.joint_torques.log)**2
        )) * self.time_step / self.run_time

    def update_logs(self):
        """ Update all the physics logs. """
        self.sim_data.base_position.values = np.asarray(
            self.base_position)
        self.sim_data.joint_positions.values = np.asarray(
            self.joint_positions)
        self.sim_data.joint_velocities.values = np.asarray(
            self.joint_velocities)
        self.sim_data.joint_torques.values = np.asarray(
            self.joint_torques)
        self.sim_data.collision_forces.values = np.asarray(
            self.collision_forces).flatten()
        self.sim_data.ground_contacts.values = np.asarray(
            self.ground_reaction_forces).flatten()
        self.sim_data.ground_friction_dir1.values = np.asarray(
            self.ground_lateral_friction_dir1).flatten()
        self.sim_data.ground_friction_dir2.values = np.asarray(
            self.ground_lateral_friction_dir2).flatten()
        if self.ground == 'ball':
            self.sim_data.ball_rotations.values = np.asarray(
                self.ball_rotations).flatten()

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

        #: Camera
        if self.gui == p.GUI and self.track_animal:
            base = np.array(self.base_position) * self.units.meters
            yaw = 30
            pitch = -10
            p.resetDebugVisualizerCamera(
                self.camera_distance,
                yaw,
                pitch,
                base)

        #: Walking camera sequence, set rotate_camera to True to activate
        if self.gui == p.GUI and self.rotate_camera and self.behavior == 'walking':
            base = np.array(self.base_position)
            base[-1] = 1.10

            if t < 3000:
                yaw = 0
                pitch = -10
            elif t >= 3000 and t < 4000:
                yaw = (t - 3000) / 1000 * 90
                pitch = -10
            elif t >= 4000 and t < 4250:
                yaw = 90
                pitch = -10
            elif t >= 4250 and t < 4750:
                yaw = 90
                pitch = (t - 4250) / 500 * 70 - 10
            elif t >= 4750 and t < 5000:
                yaw = 90
                pitch = 60
            elif t >= 5000 and t < 5500:
                yaw = 90
                pitch = 60 - (t - 5000) / 500 * 70
            elif t >= 5500 and t < 7000:
                yaw = (t - 5500) / 1500 * 300 + 90
                pitch = -10
            else:
                yaw = 30
                pitch = -10
            p.resetDebugVisualizerCamera(
                self.camera_distance,
                yaw,
                pitch,
                base)

        #: Grooming camera sequence, set rotate_camera to True to activate
        if self.gui == p.GUI and self.rotate_camera and self.behavior == 'grooming':
            base = np.array(self.base_position)
            if t < 250:
                yaw = 0
                pitch = -10
            elif t >= 250 and t < 2000:
                yaw = (t - 250) / 1750 * 150
                pitch = -10
            elif t >= 2000 and t < 3500:
                yaw = 150 - (t - 2000) / 1500 * 120
                pitch = -10
            else:
                yaw = 30
                pitch = -10
            p.resetDebugVisualizerCamera(
                self.camera_distance,
                yaw,
                pitch,
                base)

        if self.gui == p.GUI and self.rotate_camera and self.behavior == None:
            base = np.array(self.base_position) * self.units.meters
            yaw = (t-4500)/4500*360
            pitch = -10
            p.resetDebugVisualizerCamera(
                self.camera_distance,
                yaw,
                pitch,
                base)

        #: Update the feedback to controller
        self.feedback_to_controller()
        #: Step controller
        if self.controller_config:
            self.controller.step(self.time_step)
        #: Update the controller_to_actuator
        self.controller_to_actuator(t)
        #: Step muscles
        if self.use_muscles:
            self.muscles.step()
        #: Step time
        self.time += self.time_step
        #: Step physics
        p.stepSimulation()
        #: Rendering
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
        #: Update logs
        self.update_logs()
        #: Update container log
        self.container.update_log()
        #: Slow down the simulation
        if self.slow_down:
            time.sleep(self.sleep_time)
        #: Check if optimization is to be killed
        if optimization:
            optimization_status = self.optimization_check()
            return optimization_status
        return True

    def run(self, optimization=False):
        """ Run the full simulation. """
        total = int(self.run_time / self.time_step)
        for t in tqdm(
            range(0, total),
            disable = optimization
        ):
            status = self.step(t, optimization=optimization)
            if not status:
                return False

""" Class to run animal model. """

import abc
import time

from tqdm import tqdm
import pybullet as p
import pybullet_data
import numpy as np
import yaml
try:
    from NeuroMechFly.network.neural_system import NeuralSystem
except ImportError:
    print('network module not found!')


class BulletSimulation(metaclass=abc.ABCMeta):
    """Methods to run bullet simulation.
    """

    def __init__(self, container, units, **kwargs):
        super(BulletSimulation, self).__init__()
        self.units = units
        #: Simulation options
        self.gui = p.DIRECT if kwargs['headless'] else p.GUI
        self.gravity = np.array(
            kwargs.get('gravity', [0, 0, -9.81])
        )*self.units.gravity
        self.time_step = kwargs.get('time_step', 0.001)*self.units.seconds
        self.real_time = kwargs.get('real_time', 0)
        self.run_time = kwargs.get('run_time', 10)*self.units.seconds
        self.solver_iterations = kwargs.get('solver_iterations', 50)
        self.model = kwargs.get('model', None)
        self.model_offset = np.array(
            kwargs.get('model_offset', [0., 0., 0.])
        )*self.units.meters
        self.ground_contacts = kwargs.get('ground_contacts', ())
        self.base_link = kwargs.get('base_link', None)
        self.controller = kwargs.get('controller', None)
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
        self.movie_fps = kwargs.get('moviefps', 60)
        self.behavior = kwargs.get('behavior', 'walking')
        self.rotate_camera = kwargs.get('rot_cam', False)
        self.self_collisions = kwargs.get('self_collisions', [])
        self.draw_collisions = kwargs.get('draw_collisions', False)
        self.is_ball = kwargs.get("is_ball", True)

        #: Init
        self.time = 0.0
        self.plane = None
        self.animal = None
        self.control = None
        self.num_joints = 0
        self.joint_id = {}
        self.joint_type = {}
        self.link_id = {}
        self.ground_sensors = {}
        self.pose_data = {}

        ##################
        self.link_names = []
        ####################

        #: ADD PHYSICS SIMULATION namespace to container
        self.sim_data = self.container.add_namespace('physics')
        #: ADD Tables to physics container
        self.sim_data.add_table('base_position')
        self.sim_data.add_table('joint_positions')
        self.sim_data.add_table('joint_velocities')
        self.sim_data.add_table('joint_torques')
        self.sim_data.add_table('ground_contacts')

        self.ZEROS_3x1 = np.zeros((3,))

        #: Muscles
        if self.muscle_config_file:
            self.muscles = True
        else:
            self.muscles = False

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
                0,
                -25,
                base,
            )

        #: Initialize simulation
        self.initialize_simulation()

    def __del__(self):
        print('Disconnecting pybullet')
        p.disconnect()

    def rendering(self, render=1):
        """Enable/disable rendering"""
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, render)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    def setup_simulation(self):
        """ Setup the simulation. """
        ########## PYBULLET SETUP ##########
        if self.record_movie and self.gui == p.GUI:
            p.connect(
                self.gui,
                options='--background_color_red={} --background_color_green={} --background_color_blue={} --mp4={} --mp4fps={}'.format(
                    self.vis_options_background_color_red,
                    self.vis_options_background_color_green,
                    self.vis_options_background_color_red,
                    self.movie_name,
                    int(1.0/self.time_step)
                )
            )
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
        elif self.gui == p.GUI:
            p.connect(
                self.gui,
                options='--background_color_red={} --background_color_green={} --background_color_blue={}'.format(
                    self.vis_options_background_color_red,
                    self.vis_options_background_color_green,
                    self.vis_options_background_color_red
                )
            )
        else:
            p.connect(self.gui)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        #: everything should fall down
        p.setGravity(self.gravity[0], self.gravity[1], self.gravity[2])
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step,
            numSolverIterations=100,
            enableFileCaching=0,
            numSubSteps=1,
            solverResidualThreshold=1e-10,
            # erp=1e-1,
            contactERP=0.0,
            frictionERP=0.0,
        )
        #: Turn off rendering while loading the models
        self.rendering(0)

        ########## ADD FLOOR ##########
        self.plane = p.loadURDF(
            'plane.urdf', [0, 0, -0.],
            globalScaling=self.units.meters*0.01
        )

        ########## ADD BALL ######################
        self.ball_radius = 5e-03*self.units.meters  # 100x (real size d=10mm)
        self.ball_id = self.add_ball(self.ball_radius)

        ########## ADD ANIMAL #########
        if '.sdf' in self.model:
            print(self.model)
            self.animal = p.loadSDF(self.model)[0]
        elif '.urdf' in self.model:
            self.animal = p.loadURDF(self.model)
        p.resetBasePositionAndOrientation(
            self.animal, self.model_offset,
            p.getQuaternionFromEuler([0., 0., 0.]))
        self.num_joints = p.getNumJoints(self.animal)

        #: Generate joint_name to id dict
        self.link_id[p.getBodyInfo(self.animal)[0].decode('UTF-8')] = -1

        colorWings = [91/100, 96/100, 97/100, 0.7]
        colorEyes = [67/100, 21/100, 12/100, 1]
        self.color_body = [140/255, 100/255, 30/255, 1]
        self.color_legs = [170/255, 130/255, 50/255, 1]
        self.color_collision = [0, 1, 0, 1]
        nospecular = [0.5, 0.5, 0.5]

        p.changeVisualShape(
            self.animal, -1,
            rgbaColor=self.color_body,
            specularColor=nospecular,
        )

        for n in range(self.num_joints):
            info = p.getJointInfo(self.animal, n)
            _id = info[0]
            joint_name = info[1].decode('UTF-8')
            link_name = info[12].decode('UTF-8')
            _type = info[2]
            self.joint_id[joint_name] = _id
            self.joint_type[joint_name] = _type
            self.link_id[link_name] = _id

            if 'Wing' in joint_name and 'Fake' not in joint_name:
                p.changeVisualShape(
                    self.animal, _id,
                    rgbaColor=colorWings,
                )
            elif 'Eye' in joint_name and 'Fake' not in joint_name:
                p.changeVisualShape(
                    self.animal, _id,
                    rgbaColor=colorEyes,
                )
            # and 'Fake' not in joint_name:
            elif (
                    'Tarsus' in joint_name
                    or 'Tibia' in joint_name
                    or 'Femur' in joint_name
                    or 'Coxa' in joint_name
            ):
                p.changeVisualShape(
                    self.animal, _id,
                    rgbaColor=self.color_legs,
                    specularColor=nospecular,
                )
            elif 'Fake' not in joint_name:
                p.changeVisualShape(
                    self.animal, _id,
                    rgbaColor=self.color_body,
                    specularColor=nospecular,
                )

            #print('Link name {} id {}'.format(link_name, _id))
            # self.link_names.append(link_name)
        """

        ########## ADD ANIMAL #########
        if '.sdf' in self.MODEL:
            #self.animal = p.loadSDF(self.MODEL)[0]
            self.animal, links, joints = load_sdf(self.MODEL)#, force_concave=True)
        elif '.urdf' in self.MODEL:
            self.animal = p.loadURDF(self.MODEL)
        p.resetBasePositionAndOrientation(
            self.animal, self.model_offset,
            p.getQuaternionFromEuler([0., 0., 0.]))
        self.num_joints = p.getNumJoints(self.animal)

        #: Generate joint_name to id dict
        #: FUCK : Need to clean this section
        self.link_id[p.getBodyInfo(self.animal)[0].decode('UTF-8')] = -1


        colorWings = [91/100,96/100,97/100,0.7]
        colorEyes = [67/100,21/100,12/100,1]
        colorBody = [140/255,100/255,30/255,1]
        colorLegs = [170/255,130/255,50/255,1]
        nospecular = [0.5,0.5,0.5]

        p.changeVisualShape(self.animal, -1, rgbaColor=colorBody,specularColor=nospecular)

        self.joint_id = joints
        self.link_id = links
        for link_name, _id in self.link_id.items():
            if 'Wing' in link_name and 'Fake' not in link_name:
                p.changeVisualShape(self.animal, _id, rgbaColor=colorWings)
            elif 'Eye' in link_name and 'Fake' not in link_name:
                p.changeVisualShape(self.animal, _id, rgbaColor=colorEyes)
            elif ('Tarsus' in link_name or 'Tibia' in link_name or 'Femur' in link_name or 'Coxa' in link_name):# and 'Fake' not in link_name:
                p.changeVisualShape(self.animal, _id, rgbaColor=colorLegs,specularColor=nospecular)
            elif 'Fake' not in link_name:
                p.changeVisualShape(self.animal, _id, rgbaColor=colorBody,specularColor=nospecular)

            #print('Link name {} id {}'.format(link_name, _id))
            pylog.debug('Link name {} id {}'.format(link_name, _id))
            #self.link_names.append(link_name)

        ############### CONFIGURE CONTACTS ###############
        self.set_collisions(links, group=0, mask=0)

        # Disable all self-collisions
        for link0 in self.link_id.keys():
            for link1 in self.link_id.keys():
                p.setCollisionFilterPair(
                    bodyUniqueIdA=self.animal,
                    bodyUniqueIdB=self.animal,
                    linkIndexA=self.link_id[link0],
                    linkIndexB=self.link_id[link1],
                    enableCollision=0,
                    )
        # Enable tarsi-ball collisions
        for link in self.link_id.keys():
            if 'Tarsus' in link or 'Tibia' in link or 'Femur' in link:
                p.setCollisionFilterPair(
                    bodyUniqueIdA=self.animal,
                    bodyUniqueIdB=self.ball_id,
                    linkIndexA=self.link_id[link],
                    linkIndexB=2,
                    enableCollision=1,
                )

        # Disable selected self-collisions
        for (link0, link1) in self.self_collisions:
            #print(link0,link1)
            p.setCollisionFilterPair(
                bodyUniqueIdA=self.animal,
                bodyUniqueIdB=self.animal,
                linkIndexA=self.link_id[link0],
                linkIndexB=self.link_id[link1],
                enableCollision=1,
            )
        """

        ########## ADD GROUND_CONTACTS ##########
        for contact in self.ground_contacts:
            self.add_ground_contact_sensor(contact)
            self.sim_data.ground_contacts.add_parameter(contact)

        ########## ADD MUSCLES ##########
        if self.muscles:
            self.initialize_muscles()

        ########## ADD CONTROLLER ##########
        if self.controller:
            self.controller = NeuralSystem(
                self.controller,
                self.container,
            )

        #: ADD base position parameters
        self.sim_data.base_position.add_parameter('x')
        self.sim_data.base_position.add_parameter('y')
        self.sim_data.base_position.add_parameter('z')

        #: ADD joint paramters
        for name, _ in self.joint_id.items():
            self.sim_data.joint_positions.add_parameter(name)
            self.sim_data.joint_velocities.add_parameter(name)
            self.sim_data.joint_torques.add_parameter(name)

        ########## DISABLE DEFAULT BULLET CONTROLLERS  ##########
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
            self.total_mass += p.getDynamicsInfo(self.animal, j)[0] / self.units.kilograms

        self.bodyweight = -1 * self.total_mass * self.gravity[2]
        #print('Total mass = {}'.format(self.total_mass))
        if self.gui == p.GUI:
            self.rendering(1)

    def set_collisions(self, links, group=0, mask=0):
        """Activate/Deactivate leg collisions"""
        for link in links:
            p.setCollisionFilterGroupMask(
                bodyUniqueId=self.animal,
                linkIndexA=self.link_id[link],
                collisionFilterGroup=group,
                collisionFilterMask=mask
            )

    def initialize_simulation(self):
        """ Initialize simulation. """
        ########## INITIALIZE THE CONTAINER ##########
        self.container.initialize()

        ########## SETUP THE INTEGRATOR ##########
        if self.controller:
            self.controller.setup_integrator()
        if self.muscles:
            self.muscles.setup_integrator()

    def initialize_muscles(self):
        self.muscles = MusculoSkeletalSystem(self.MUSCLE_CONFIG_FILE)

    def initialize_position(self, pose_file=None):
        """Initialize the pose of the animal.
        Parameters:
        pose_file : <selftr>
             File path to the pose data
        """
        if pose_file:
            try:
                with open(pose_file) as stream:
                    self.pose_data = yaml.load(stream, Loader=yaml.SafeLoader)
                    self.pose_data = {k.lower(): v for k, v in self.pose_data.items()}
            except FileNotFoundError:
                print("Pose file {} not found".format(pose_file))
                return
            for joint, _id in self.joint_id.items():
                _pose = np.deg2rad(self.pose_data['joints'].get(joint, 0))
                p.resetJointState(
                    self.animal, _id,
                    targetValue=_pose
                )
        else:
            return None

    def add_ground_contact_sensor(self, link):
        """Add new ground contact sensor

        Parameters
        ----------

        Returns
        -------
        out :

        """
        self.ground_sensors[link] = self.link_id[link]

    def _get_contact_force(self, link_id):
        c = p.getContactPoints(
            self.animal, self.plane,
            link_id, -1)
        self.contact_pos = np.sum(
            [pt[5] for pt in c], axis=0) / len(c) if c else self.ZEROS_3x1
        self.normal_dir = -1 * np.sum(
            [pt[7]for pt in c], axis=0) / len(c) if c else self.ZEROS_3x1
        self.normal = np.sum(
            [pt[9]for pt in c], axis=0) if c else self.ZEROS_3x1
        force = self.normal * self.normal_dir

        return force[2]/self.units.newtons

    def _get_contact_force_ball(self, link_id):
        c = p.getContactPoints(
            self.animal, self.ball_id,
            link_id, 2)
        self.contact_pos = np.sum(
            [pt[5] for pt in c], axis=0) / len(c) if c else self.ZEROS_3x1
        self.normal_dir = 1 * np.sum(
            [pt[7]for pt in c], axis=0) / len(c) if c else self.ZEROS_3x1
        self.normal = np.sum(
            [pt[9]for pt in c], axis=0) if c else self.ZEROS_3x1
        force = self.normal * self.normal_dir
        res_force = np.linalg.norm(force)
        res_dir = np.arctan2(force[2], force[1])
        #print(len(c),self.normal_dir, self.normal,force)
        # return force[2]
        return res_force/self.units.newtons, res_dir

    def get_contact_friction(self, link_id):
        c = p.getContactPoints(
            self.animal, self.plane,
            link_id, -1)

        force1 = np.sum(
            [pt[10]*np.asarray(pt[11]) for pt in c], axis=0) if c else self.ZEROS_3x1
        force2 = np.sum(
            [pt[12]*np.asarray(pt[13]) for pt in c], axis=0) if c else self.ZEROS_3x1
        return force1/self.units.newtons, force2/self.units.newtons

    def is_contact(self, link_name):
        """ Check if link is in contact with floor. """
        return True if p.getContactPoints(
            self.animal, self.plane,
            self.link_id[link_name],
            -1
        ) else False

    def is_contact_ball(self, link_name):
        """ Check if link is in contact with floor. """
        return True if p.getContactPoints(
            self.animal, self.ball_id,
            self.link_id[link_name],
            2
        ) else False

    def get_link_position(self, link_name):
        return (p.getLinkState(self.animal, self.link_id[link_name]))[0]

    def add_ball(self, r):
        # Create Fly ball
        # TODO: Remove the collision for parent sphere completely
        colSphereParent = p.createCollisionShape(p.GEOM_SPHERE, radius=r/100)
        colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=r)

        massParent = 0
        visualShapeId = -1

        if self.behavior == 'walking':
            # basePosition=[-0.025,0.01,0.566] ### Walking ball r= 0.55
            # basePosition=[-0.025,0.005,0.568] ### Walking ball r= 0.55 NEW
            # basePosition=[-0.023, 0.0085, 0.6198] ### Walking ball r= 0.5
            # basePosition=[-0.0225, 0.007, 0.61973] ### Walking ball r= 0.5
            basePosition = np.array(
                [0.2e-3, 0.0e-3,-5.1e-3])*self.units.meters+self.model_offset
        elif self.behavior == 'grooming':
            # basePosition=[0.0,-0.01,0.63] ### Grooming
            basePosition = np.array(
                [0.0e-3, 0.0e-3, -5e-3])*self.units.meters+self.model_offset

        # basePosition=[-0.03,-0.0,0.589] ### Walking ball r= 0.525
        # basePosition=[-0.02,0.0,0.595] ### Walking ball r= 0.52
        # basePosition=[-0.04,-0.005,0.594] ### Walking ball r=0.52
        # basePosition=[-0.04,0.0,0.605] ### Walking ball r=0.51

        baseOrientation = [0, 0, 0, 1]
        #link_Masses = [0.0000005,0.0000005,0.0000005]
        link_Masses = np.array([5e-11, 5e-11, 5e-11])*self.units.kilograms
        linkCollisionShapeIndices = [-1, -1, colSphereId]
        linkVisualShapeIndices = [-1, -1, -1]
        linkPositions = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        linkOrientations = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
        linkInertialFramePositions = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        linkInertialFrameOrientations = [
            [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
        indices = [0, 1, 2]
        jointTypes = [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]
        #axis = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
        axis = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]

        sphereId = p.createMultiBody(
            massParent,
            -1,  # colSphereParent,
            visualShapeId,
            basePosition,
            baseOrientation,
            linkMasses=link_Masses,
            linkCollisionShapeIndices=linkCollisionShapeIndices,
            linkVisualShapeIndices=linkVisualShapeIndices,
            linkPositions=linkPositions,
            linkOrientations=linkOrientations,
            linkInertialFramePositions=linkInertialFramePositions,
            linkInertialFrameOrientations=linkInertialFrameOrientations,
            linkParentIndices=indices,
            linkJointTypes=jointTypes,
            linkJointAxis=axis,
        )

        # p.changeDynamics(sphereId,
        #               -1,
        #               spinningFriction=100,
        #               linearDamping=0.0)
        textureBall = p.loadTexture(
            '../../design/textures/ball/chequered_0048.jpg')
        p.changeVisualShape(
            sphereId, 2, rgbaColor=[225/255, 225/255, 210/255, 1],
            specularColor=[0, 0, 0], textureUniqueId=textureBall
        )
        return sphereId

    @property
    def joint_states(self):
        """ Get all joint states  """
        return p.getJointStates(
            self.animal,
            np.arange(0, p.getNumJoints(self.animal))
        )

    @property
    def ground_reaction_forces(self):
        """Get the ground reaction forces.  """
        return list(
            map(self._get_contact_force, self.ground_sensors.values())
        )

    @property
    def base_position(self):
        """ Get the position of the animal  """
        if self.base_link and self.link_id[self.base_link] != -1:
            link_id = self.link_id[self.base_link]
            return np.asarray(
                (p.getLinkState(self.animal, link_id))[0]
            )/self.units.meters
        else:
            return np.asarray(
                (p.getBasePositionAndOrientation(self.animal))[0]
            )/self.units.meters

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
            state[-1]/self.units.torques for state in p.getJointStates(
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
        return self.base_position[0]

    @property
    def distance_y(self):
        """ Distance the animal has travelled in y-direction. """
        return -self.base_position[1]

    @property
    def distance_z(self):
        """ Distance the animal has travelled in z-direction. """
        return self.base_position[2]

    @property
    def mechanical_work(self):
        """ Mechanical work done by the animal. """
        return np.sum(np.sum(
            np.abs(np.asarray(self.sim_data.joint_torques.log)
                   * np.asarray(self.sim_data.joint_velocities.log))
        ))*self.time_step/self.run_time

    @property
    def thermal_loss(self):
        """ Thermal loss for the animal. """
        return np.sum(np.sum(
            np.asarray(self.sim_data.joint_torques.log)**2
        ))*self.time_step/self.run_time

    def update_logs(self):
        """ Update all the physics logs. """
        #: Update log
        self.sim_data.base_position.values = np.asarray(
            self.base_position)
        self.sim_data.joint_positions.values = np.asarray(
            self.joint_positions)
        self.sim_data.joint_velocities.values = np.asarray(
            self.joint_velocities)
        self.sim_data.joint_torques.values = np.asarray(
            self.joint_torques()
        )
        self.sim_data.ground_contacts.values = np.asarray(
            self.ground_reaction_forces
        )

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
            p.resetDebugVisualizerCamera(
                self.camera_distance,
                -90,
                -10,
                base)

        ######Optimization camera sequence#######
        if self.gui == p.GUI and self.rotate_camera:
            base = np.array(self.base_position) * self.units.meters
            yaw = (t-4500)/4500*360-90
            pitch = -10
            p.resetDebugVisualizerCamera(
                self.camera_distance,
                yaw,
                pitch,
                base)

        #: update the feedback to controller
        self.feedback_to_controller()

        #: Step controller
        if self.controller:
            self.controller.step(self.time_step)

        #: update the controller_to_actuator
        self.controller_to_actuator(t)

        #: Step muscles
        if self.muscles:
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

        if self.slow_down:
            time.sleep(self.sleep_time)

        #: Check if optimization is to be killed
        if optimization:
            optimization_status = self.optimization_check()
            return optimization_status

        return True

    def run(self, optimization=False):
        """ Run the full simulation. """
        for t in tqdm(range(0, int(self.run_time / self.time_step))):
            status = self.step(t, optimization=optimization)
            if t > 10 and not status:
                return False


def main():
    """ Main """

    sim_options = {'headless': False,
                   'model': '../mouse/models/mouse_bullet/sdf/mouse_bullet.sdf',
                   'model_offset': [0., 0., 0.05],
                   'pose': '../mouse/config/mouse_rig_simple_default_pose.yml',
                   'run_time': 10.}

    animal = BulletSimulation(**sim_options)
    animal.run(optimization=True)
    # container = Container.get_instance()
    # animal.control.visualize_network(edge_labels=False)
    # plt.figure()
    # plt.plot(np.sin(container.neural.outputs.log))
    # plt.show()


if __name__ == '__main__':
    main()

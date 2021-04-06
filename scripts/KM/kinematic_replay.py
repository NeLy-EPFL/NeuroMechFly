import pybullet as p
import numpy as np
import pickle

from bullet_simulation_kinematic_replay import BulletSimulation
from NeuroMechFly.container import Container
from NeuroMechFly.sdf.units import SimulationUnitScaling

class DrosophilaSimulation(BulletSimulation):

    def __init__(
        self,
        container,
        sim_options,
        Kp,
        Kv,
        units=SimulationUnitScaling(
            meters=1000,
            kilograms=1000)):
        super().__init__(container, units, **sim_options)

        self.kp = Kp
        self.kv = Kv
        self.angles = self.load_angles(
            './new_angles/walking_converted_joint_angles_smoothed.pkl')
        self.velocities = self.load_angles(
            './new_angles/walking_converted_joint_velocities.pkl')

    def load_angles(self, data_path):
        try:
            with open(data_path, 'rb') as f:
                return pickle.load(f)
        except BaseException:
            FileNotFoundError(f"File {data_path} not found!")

    def controller_to_actuator(self, t):
        """
        Code that glues the controller the actuator in the system.
        If there are muscles then contoller actuates the muscles.
        If not then the controller directly actuates the joints
        """

        joints = [joint for joint in range(self.num_joints)]
        pose = [0] * self.num_joints
        vel = [0] * self.num_joints

        pose[self.joint_id['joint_A3']] = np.deg2rad(-15)
        pose[self.joint_id['joint_A4']] = np.deg2rad(-15)
        pose[self.joint_id['joint_A5']] = np.deg2rad(-15)
        pose[self.joint_id['joint_A6']] = np.deg2rad(-15)

        pose[self.joint_id['joint_LAntenna']] = np.deg2rad(33)
        pose[self.joint_id['joint_RAntenna']] = np.deg2rad(-33)

        pose[self.joint_id['joint_Rostrum']] = np.deg2rad(90)
        pose[self.joint_id['joint_Haustellum']] = np.deg2rad(-60)

        pose[self.joint_id['joint_LWing_roll']] = np.deg2rad(90)
        pose[self.joint_id['joint_LWing_yaw']] = np.deg2rad(-17)
        pose[self.joint_id['joint_RWing_roll']] = np.deg2rad(-90)
        pose[self.joint_id['joint_RWing_yaw']] = np.deg2rad(17)

        pose[self.joint_id['joint_Head']] = np.deg2rad(10)

        ind = t + 1000

        ####### Walk on floor#########
        '''
        init_lim = 25
        if ind<init_lim:
            pose[self.joint_id['prismatic_support_2']] = (1.01*self.MODEL_OFFSET[2]-ind*self.MODEL_OFFSET[2]/init_lim)*self.units.meters
        else:
            pose[self.joint_id['prismatic_support_2']] = 0
        '''
        ####LEFT LEGS#######
        pose[self.joint_id['joint_LFCoxa_roll']
             ] = self.angles['LF_leg']['roll'][ind]
        pose[self.joint_id['joint_LFCoxa_yaw']
             ] = self.angles['LF_leg']['yaw'][ind]
        pose[self.joint_id['joint_LFCoxa']] = self.angles['LF_leg']['pitch'][ind]
        pose[self.joint_id['joint_LFFemur_roll']
             ] = self.angles['LF_leg']['roll_tr'][ind]
        pose[self.joint_id['joint_LFFemur']
             ] = self.angles['LF_leg']['th_fe'][ind]
        pose[self.joint_id['joint_LFTibia']
             ] = self.angles['LF_leg']['th_ti'][ind]
        pose[self.joint_id['joint_LFTarsus1']
             ] = self.angles['LF_leg']['th_ta'][ind]

        pose[self.joint_id['joint_LMCoxa_roll']
             ] = self.angles['LM_leg']['roll'][ind]
        pose[self.joint_id['joint_LMCoxa_yaw']
             ] = self.angles['LM_leg']['yaw'][ind]
        pose[self.joint_id['joint_LMCoxa']] = self.angles['LM_leg']['pitch'][ind]
        pose[self.joint_id['joint_LMFemur_roll']
             ] = self.angles['LM_leg']['roll_tr'][ind]
        pose[self.joint_id['joint_LMFemur']
             ] = self.angles['LM_leg']['th_fe'][ind]
        pose[self.joint_id['joint_LMTibia']
             ] = self.angles['LM_leg']['th_ti'][ind]
        pose[self.joint_id['joint_LMTarsus1']
             ] = self.angles['LM_leg']['th_ta'][ind]

        pose[self.joint_id['joint_LHCoxa_roll']
             ] = self.angles['LH_leg']['roll'][ind]
        pose[self.joint_id['joint_LHCoxa_yaw']
             ] = self.angles['LH_leg']['yaw'][ind]
        pose[self.joint_id['joint_LHCoxa']] = self.angles['LH_leg']['pitch'][ind]
        pose[self.joint_id['joint_LHFemur_roll']
             ] = self.angles['LH_leg']['roll_tr'][ind]
        pose[self.joint_id['joint_LHFemur']
             ] = self.angles['LH_leg']['th_fe'][ind]
        pose[self.joint_id['joint_LHTibia']
             ] = self.angles['LH_leg']['th_ti'][ind]
        pose[self.joint_id['joint_LHTarsus1']
             ] = self.angles['LH_leg']['th_ta'][ind]

        #####RIGHT LEGS######
        pose[self.joint_id['joint_RFCoxa_roll']
             ] = self.angles['RF_leg']['roll'][ind]
        pose[self.joint_id['joint_RFCoxa_yaw']
             ] = self.angles['RF_leg']['yaw'][ind]
        pose[self.joint_id['joint_RFCoxa']] = self.angles['RF_leg']['pitch'][ind]
        pose[self.joint_id['joint_RFFemur_roll']
             ] = self.angles['RF_leg']['roll_tr'][ind]
        pose[self.joint_id['joint_RFFemur']
             ] = self.angles['RF_leg']['th_fe'][ind]
        pose[self.joint_id['joint_RFTibia']
             ] = self.angles['RF_leg']['th_ti'][ind]
        pose[self.joint_id['joint_RFTarsus1']
             ] = self.angles['RF_leg']['th_ta'][ind]

        pose[self.joint_id['joint_RMCoxa_roll']
             ] = self.angles['RM_leg']['roll'][ind]
        pose[self.joint_id['joint_RMCoxa_yaw']
             ] = self.angles['RM_leg']['yaw'][ind]
        pose[self.joint_id['joint_RMCoxa']] = self.angles['RM_leg']['pitch'][ind]
        pose[self.joint_id['joint_RMFemur_roll']
             ] = self.angles['RM_leg']['roll_tr'][ind]
        pose[self.joint_id['joint_RMFemur']
             ] = self.angles['RM_leg']['th_fe'][ind]
        pose[self.joint_id['joint_RMTibia']
             ] = self.angles['RM_leg']['th_ti'][ind]
        pose[self.joint_id['joint_RMTarsus1']
             ] = self.angles['RM_leg']['th_ta'][ind]

        pose[self.joint_id['joint_RHCoxa_roll']
             ] = self.angles['RH_leg']['roll'][ind]
        pose[self.joint_id['joint_RHCoxa_yaw']
             ] = self.angles['RH_leg']['yaw'][ind]
        pose[self.joint_id['joint_RHCoxa']] = self.angles['RH_leg']['pitch'][ind]
        pose[self.joint_id['joint_RHFemur_roll']
             ] = self.angles['RH_leg']['roll_tr'][ind]
        pose[self.joint_id['joint_RHFemur']
             ] = self.angles['RH_leg']['th_fe'][ind]
        pose[self.joint_id['joint_RHTibia']
             ] = self.angles['RH_leg']['th_ti'][ind]
        pose[self.joint_id['joint_RHTarsus1']
             ] = self.angles['RH_leg']['th_ta'][ind]
        #################### VELOCITY SET ####################

        ####LEFT LEGS#######
        vel[self.joint_id['joint_LFCoxa_roll']
            ] = self.velocities['LF_leg']['roll'][ind + 0]
        vel[self.joint_id['joint_LFCoxa_yaw']
            ] = self.velocities['LF_leg']['yaw'][ind + 0]
        vel[self.joint_id['joint_LFCoxa']
            ] = self.velocities['LF_leg']['pitch'][ind + 0]
        vel[self.joint_id['joint_LFFemur_roll']
            ] = self.velocities['LF_leg']['roll_tr'][ind + 0]
        vel[self.joint_id['joint_LFFemur']
            ] = self.velocities['LF_leg']['th_fe'][ind + 0]
        vel[self.joint_id['joint_LFTibia']
            ] = self.velocities['LF_leg']['th_ti'][ind + 0]
        vel[self.joint_id['joint_LFTarsus1']
            ] = self.velocities['LF_leg']['th_ta'][ind + 0]

        vel[self.joint_id['joint_LMCoxa_roll']
            ] = self.velocities['LM_leg']['roll'][ind + 0]
        vel[self.joint_id['joint_LMCoxa_yaw']
            ] = self.velocities['LM_leg']['yaw'][ind + 0]
        vel[self.joint_id['joint_LMCoxa']
            ] = self.velocities['LM_leg']['pitch'][ind + 0]
        vel[self.joint_id['joint_LMFemur_roll']
            ] = self.velocities['LM_leg']['roll_tr'][ind + 0]
        vel[self.joint_id['joint_LMFemur']
            ] = self.velocities['LM_leg']['th_fe'][ind + 0]
        vel[self.joint_id['joint_LMTibia']
            ] = self.velocities['LM_leg']['th_ti'][ind + 0]
        vel[self.joint_id['joint_LMTarsus1']
            ] = self.velocities['LM_leg']['th_ta'][ind + 0]

        vel[self.joint_id['joint_LHCoxa_roll']
            ] = self.velocities['LH_leg']['roll'][ind + 0]
        vel[self.joint_id['joint_LHCoxa_yaw']
            ] = self.velocities['LH_leg']['yaw'][ind + 0]
        vel[self.joint_id['joint_LHCoxa']
            ] = self.velocities['LH_leg']['pitch'][ind + 0]
        vel[self.joint_id['joint_LHFemur_roll']
            ] = self.velocities['LH_leg']['roll_tr'][ind + 0]
        vel[self.joint_id['joint_LHFemur']
            ] = self.velocities['LH_leg']['th_fe'][ind + 0]
        vel[self.joint_id['joint_LHTibia']
            ] = self.velocities['LH_leg']['th_ti'][ind + 0]
        vel[self.joint_id['joint_LHTarsus1']
            ] = self.velocities['LH_leg']['th_ta'][ind + 0]

        #####RIGHT LEGS######
        vel[self.joint_id['joint_RFCoxa_roll']
            ] = self.velocities['RF_leg']['roll'][ind + 0]
        vel[self.joint_id['joint_RFCoxa_yaw']
            ] = self.velocities['RF_leg']['yaw'][ind + 0]
        vel[self.joint_id['joint_RFCoxa']
            ] = self.velocities['RF_leg']['pitch'][ind + 0]
        vel[self.joint_id['joint_RFFemur_roll']
            ] = self.velocities['RF_leg']['roll_tr'][ind + 0]
        vel[self.joint_id['joint_RFFemur']
            ] = self.velocities['RF_leg']['th_fe'][ind + 0]
        vel[self.joint_id['joint_RFTibia']
            ] = self.velocities['RF_leg']['th_ti'][ind + 0]
        vel[self.joint_id['joint_RFTarsus1']
            ] = self.velocities['RF_leg']['th_ta'][ind + 0]

        vel[self.joint_id['joint_RMCoxa_roll']
            ] = self.velocities['RM_leg']['roll'][ind + 0]
        vel[self.joint_id['joint_RMCoxa_yaw']
            ] = self.velocities['RM_leg']['yaw'][ind + 0]
        vel[self.joint_id['joint_RMCoxa']
            ] = self.velocities['RM_leg']['pitch'][ind + 0]
        vel[self.joint_id['joint_RMFemur_roll']
            ] = self.velocities['RM_leg']['roll_tr'][ind + 0]
        vel[self.joint_id['joint_RMFemur']
            ] = self.velocities['RM_leg']['th_fe'][ind + 0]
        vel[self.joint_id['joint_RMTibia']
            ] = self.velocities['RM_leg']['th_ti'][ind + 0]
        vel[self.joint_id['joint_RMTarsus1']
            ] = self.velocities['RM_leg']['th_ta'][ind + 0]

        vel[self.joint_id['joint_RHCoxa_roll']
            ] = self.velocities['RH_leg']['roll'][ind + 0]
        vel[self.joint_id['joint_RHCoxa_yaw']
            ] = self.velocities['RH_leg']['yaw'][ind + 0]
        vel[self.joint_id['joint_RHCoxa']
            ] = self.velocities['RH_leg']['pitch'][ind + 0]
        vel[self.joint_id['joint_RHFemur_roll']
            ] = self.velocities['RH_leg']['roll_tr'][ind + 0]
        vel[self.joint_id['joint_RHFemur']
            ] = self.velocities['RH_leg']['th_fe'][ind + 0]
        vel[self.joint_id['joint_RHTibia']
            ] = self.velocities['RH_leg']['th_ti'][ind + 0]
        vel[self.joint_id['joint_RHTarsus1']
            ] = self.velocities['RH_leg']['th_ta'][ind + 0]

        joint_control_middle = list(
            np.arange(42, 49)) + list(np.arange(81, 88))
        joint_control_front = list(np.arange(17, 23)) + list(np.arange(56, 63))
        joint_control_hind = list(np.arange(28, 35)) + list(np.arange(67, 74))
        joint_control = joint_control_hind + joint_control_middle + joint_control_front

        for joint in range(self.num_joints):
            # if joint!=19 and joint!=58:
            p.setJointMotorControl2(
                self.animal, joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=pose[joint],
                targetVelocity=vel[joint],
                positionGain=self.kp,
                velocityGain=self.kv,
            )


    def feedback_to_controller(self):
        """
        Code that glues the sensors/feedback to controller in the system.
        """
        pass

    def update_parameters(self, params):
        """ Update parameters. """
        pass


def main():
    """ Main """
    run_time = 4.0
    time_step = 0.001
    behavior = 'walking'

    side = ['L', 'R']
    pos = ['F', 'M', 'H']
    leg_segments = ['Tibia'] + ['Tarsus' + str(i) for i in range(1, 6)]
    left_front_leg = ['LF' + name for name in leg_segments]
    right_front_leg = ['RF' + name for name in leg_segments]
    body_segments = [s + b for s in side for b in ['Eye', 'Antenna']]
    col_hind_leg = [
        s +
        'H' +
        leg for s in side for leg in [
            'Coxa',
            'Coxa_roll',
            'Femur',
            'Femur_roll',
            'Tibia']]
    col_body_abd = [
        'prismatic_support_1',
        'prismatic_support_2',
        'A1A2',
        'A3',
        'A4',
        'A5',
        'A6']
    ground_contact = [
        s +
        p +
        name for s in side for p in pos for name in leg_segments if name != 'Tibia']

    self_collision = []
    for link0 in left_front_leg:
        for link1 in right_front_leg:
            self_collision.append([link0, link1])

    for link0 in left_front_leg + right_front_leg:
        for link1 in body_segments:
            if link0[0] == link1[0]:
                self_collision.append([link0, link1])

    for link0 in col_hind_leg:
        for link1 in col_body_abd:
            self_collision.append([link0, link1])

    sim_options = {
        "headless": False,
        "model": "../../design/sdf/neuromechfly_noLimits_noSupport.sdf",
        # "model_offset": [0., -0.1, 1.12],
        "model_offset": [0, 0., 2.0e-3],
        "run_time": run_time,
        "base_link": 'Thorax',
        "ground_contacts": ground_contact,
        "self_collisions": self_collision,
        "draw_collisions": False,
        "record": True,
        'camera_distance': 6.0,
        'track': True,
        'moviename': 'videos/040421_walking_contacterp0.1_noSupport_perturbation.mp4',
        'moviefps': 80,
        'slow_down': False,
        'sleep_time': 0.001,
        'rot_cam': False,
        'behavior': behavior,
        'ground': 'floor'
    }

    container = Container(run_time / time_step)
    animal = DrosophilaSimulation(container, sim_options, Kp=0.4, Kv=0.9)
    animal.run(optimization=False)
    animal.container.dump(
        dump_path="./basepositionrecorded",
        overwrite=True)


if __name__ == '__main__':
    main()

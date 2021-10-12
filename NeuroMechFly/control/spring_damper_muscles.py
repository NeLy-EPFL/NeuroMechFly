""" Spring damper muscles. """
from dataclasses import dataclass
import numpy as np

@dataclass
class Parameters:
    """
    Muscle parameters class.

    Parameters
    ----------
    alpha: <float>
        Gain of the Ekeberg muscle model.
    beta: <float>
        Stiffness gain of the Ekeberg muscle model.
    gamma: <float>
        Tonic stiffness of the Ekeberg muscle model.
    delta: <float>
        Damping coefficient of the Ekeberg muscle model.
    rest_pos: <float>
        Resting position of the Ekeberg muscle model.
    Returns
    -------
    """
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    delta: float = 0.0
    rest_pos: float = 0.0

class SDAntagonistMuscle:
    """Antagonist Spring Damper muscles. """
    def __init__(
            self, container, name, joint_pos,
            joint_vel, rest_pos=0.0,
            flexor_mn=None, extensor_mn=None,
            flexor_amp=None, extensor_amp=None,
            parameters=None
    ):
        super().__init__()
        params = parameters if parameters else Parameters()
        self.name = name
        self.alpha = container.muscle.parameters.add_parameter(
            '{}_alpha'.format(name), params.alpha
        )[0]
        self.beta = container.muscle.parameters.add_parameter(
            '{}_beta'.format(name), params.beta
        )[0]
        self.gamma = container.muscle.parameters.add_parameter(
            '{}_gamma'.format(name), params.gamma
        )[0]
        self.delta = container.muscle.parameters.add_parameter(
            '{}_delta'.format(name), params.delta
        )[0]
        self.rest_pos = container.muscle.parameters.add_parameter(
            '{}_rest_pos'.format(name), params.rest_pos
        )[0]
        self.flexor_act = container.muscle.outputs.add_parameter(
            '{}_flexor_act'.format(name)
        )[0]
        self.extensor_act = container.muscle.outputs.add_parameter(
            '{}_extensor_act'.format(name)
        )[0]
        self.torque = container.muscle.outputs.add_parameter(
            '{}_torque'.format(name)
        )[0]
        self.active_torque = container.muscle.active_torques.add_parameter(
            '{}_active_torque'.format(name)
        )[0]
        self.passive_torque = container.muscle.passive_torques.add_parameter(
            '{}_passive_torque'.format(name)
        )[0]

        self.flexor_mn = flexor_mn
        self.extensor_mn = extensor_mn
        self.r_fmn = flexor_amp
        self.r_emn = extensor_amp
        self.jpos = joint_pos
        self.jvel = joint_vel

    def update_parameters(self, params):
        """ Update muscle parameters for optimization. """
        self.alpha.value = params.alpha
        self.beta.value = params.beta
        self.gamma.value = params.gamma
        self.delta.value = params.delta
        self.rest_pos.value = params.rest_pos

    def compute_torque(self, only_passive=False):
        """ Compute joint torque. """
        #: Passive forces
        _passive_stiff = self.beta.value*self.gamma.value*(
            self.rest_pos.value - self.jpos.value
        )
        _damp = self.delta.value*(self.jvel.value)
        self.passive_torque.value = _passive_stiff - _damp
        if only_passive:
            self.torque.value = self.passive_torque.value
            return self.torque.value
        else:
            #: Active
            self.flexor_act.value = self.r_fmn.value*(
                    1 + np.sin(self.flexor_mn.value)
            )
            self.extensor_act.value = self.r_emn.value*(
                    1 + np.sin(self.extensor_mn.value)
            )
            _co = self.alpha.value*(
                self.flexor_act.value - self.extensor_act.value
            )
            _active_stiff = self.beta.value*(
                self.flexor_act.value + self.extensor_act.value
            )*(self.rest_pos.value - self.jpos.value)
            self.active_torque.value = _co + _active_stiff
            self.torque.value = (
                self.active_torque.value + self.passive_torque.value
            )
            return self.torque.value

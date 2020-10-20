""" Plot results. """

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

nstates = pd.read_hdf('./Results/neural/states.h5')
nout = pd.read_hdf('./Results/neural/outputs.h5')

#: Plot Neural states
nstates_arr = nstates.to_numpy()

neurons = []
for key in nout.keys():
    neurons.append(key.split('_', maxsplit=1)[-1])

########## NEURAL ##########
trot1 = ('LFCoxa', 'RMCoxa_roll', 'LHCoxa_roll')
trot2 = ('RFCoxa', 'LMCoxa_roll', 'RHCoxa_roll')
RFLeg = ('RFCoxa', 'RFFemur', 'RFTibia')

plt.figure()
plt.subplot(211)
plt.title('neural-base flexion')
for elem in trot1:
    name = '{}_{}_{}'.format('joint', elem, 'flexion')
    plt.plot(nstates["amp_{}".format(name)]*np.sin(
        nstates["phase_{}".format(name)])
    )
plt.legend(trot1)
plt.subplot(212)
for elem in trot2:
    name = '{}_{}_{}'.format('joint', elem, 'flexion')
    plt.plot(nstates["amp_{}".format(name)]*np.sin(
        nstates["phase_{}".format(name)])
    )
plt.legend(trot2)

plt.figure()
plt.subplot(211)
plt.title('neural-base right front leg')
for elem in RFLeg:
    name = '{}_{}_{}'.format('joint', elem, 'flexion')
    plt.plot(nstates["amp_{}".format(name)]*np.sin(
        nstates["phase_{}".format(name)])
    )
plt.legend(RFLeg)




leg = ('Femur', 'Tibia')
plt.figure()
name = 'joint_LFFemur_flexion'
plt.plot(nstates["amp_{}".format(name)]*np.sin(
    nstates["phase_{}".format(name)])
)
name = 'joint_RMFemur_flexion'
plt.plot(nstates["amp_{}".format(name)]*np.sin(
    nstates["phase_{}".format(name)])
)

plt.figure()
name = 'joint_LFFemur_flexion'
plt.plot(nstates["amp_{}".format(name)]*np.sin(
    nstates["phase_{}".format(name)])
)
name = 'joint_RFFemur_flexion'
plt.plot(nstates["amp_{}".format(name)]*np.sin(
    nstates["phase_{}".format(name)])
)

plt.show()

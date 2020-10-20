# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: overflowcheck=False
# cython: optimize.unpack_method_calls=True
# cython: np_pythran=False

"""Oscillator model"""
from libc.stdio cimport printf
import farms_pylog as pylog
from libc.math cimport exp
from libc.math cimport M_PI
from libc.math cimport sin as csin
import numpy as np
cimport numpy as cnp


cdef class Oscillator(Neuron):

    def __init__(self, n_id, num_inputs, neural_container, **kwargs):
        """Initialize.

        Parameters
        ----------
        n_id: str
            Unique ID for the neuron in the network.
        """
        super(Oscillator, self).__init__('leaky')

        #: Neuron ID
        self.n_id = n_id
        #: Initialize parameters
        self.f = neural_container.parameters.add_parameter(
            'freq_' + self.n_id, kwargs.get('f', 0.1))[0]

        self.R = neural_container.parameters.add_parameter(
            'R_' + self.n_id, kwargs.get('R', 0.1))[0]

        self.a = neural_container.parameters.add_parameter(
            'a_' + self.n_id, kwargs.get('a', 0.1))[0]

        #: Initialize states
        self.phase = neural_container.states.add_parameter(
            'phase_' + self.n_id, kwargs.get('phase0', 0.0))[0]
        self.amp = neural_container.states.add_parameter(
            'amp_' + self.n_id, kwargs.get('amp0', 0.0))[0]
        
        #: External inputs
        self.ext_in = neural_container.inputs.add_parameter(
            'ext_in_' + self.n_id)[0]

        #: ODE RHS
        self.phase_dot = neural_container.dstates.add_parameter(
            'phase_dot_' + self.n_id, 0.0)[0]
        self.amp_dot = neural_container.dstates.add_parameter(
            'amp_dot_' + self.n_id, 0.0)[0]

        #: Output
        self.nout = neural_container.outputs.add_parameter(
            'nout_' + self.n_id, 0.0)[0]

        #: Neuron inputs
        self.neuron_inputs = cnp.ndarray((num_inputs,),
                                         dtype=[('neuron_idx', 'i'),
                                                ('weight_idx', 'i'),
                                                ('phi_idx', 'i')])

        self.num_inputs = num_inputs

    def add_ode_input(self,int idx, neuron, neural_container, **kwargs):
        """ Add relevant external inputs to the ode."""
        #: Create a struct to store the inputs and weights to the neuron
        cdef OscillatorNeuronInput n
        #: Get the neuron parameter
        neuron_idx = neural_container.outputs.get_parameter_index(
            'nout_'+neuron.n_id)

        #: Add the weight parameter
        weight = neural_container.weights.add_parameter(
            'w_' + neuron.n_id + '_to_' + self.n_id,
            kwargs.get('weight', 0.0))[0]
        phi = neural_container.parameters.add_parameter(
            'phi_' + neuron.n_id + '_to_' + self.n_id,
            kwargs.get('phi', 0.0))[0]

        weight_idx = neural_container.weights.get_parameter_index(
            'w_' + neuron.n_id + '_to_' + self.n_id)
        phi_idx = neural_container.parameters.get_parameter_index(
            'phi_' + neuron.n_id + '_to_' + self.n_id)

        n.neuron_idx = neuron_idx
        n.weight_idx = weight_idx
        n.phi_idx = phi_idx

        #: Append the struct to the list
        self.neuron_inputs[idx] = n

    def output(self):
        """Neuron activation function.
        Parameters
        ----------
        m_potential: float
            Neuron membrane potential
        """
        return self.c_output()

    def ode_rhs(self, y, w, p):
        """ Python interface to the ode_rhs computation."""
        self.c_ode_rhs(y, w, p)

    #################### C-FUNCTIONS ####################
    cdef void c_ode_rhs(self, double[:] _y, double[:] _w, double[:] _p) nogil:
        """ Compute the ODE. Internal Setup Function."""

        #: Current state
        cdef double _phase = self.phase.c_get_value()
        cdef double _amp = self.amp.c_get_value()

        #: Neuron inputs
        cdef double _sum = 0.0
        cdef unsigned int j
        cdef double _neuron_out
        cdef double _weight
        cdef double _phi

        for j in range(self.num_inputs):
            _neuron_out = _y[self.neuron_inputs[j].neuron_idx]
            _weight = _w[self.neuron_inputs[j].weight_idx]
            _phi = _p[self.neuron_inputs[j].phi_idx]
            _sum += self.c_neuron_inputs_eval(_neuron_out,
                                              _weight, _phi, _phase, _amp)

        #: phidot : phase_dot
        self.phase_dot.c_set_value(2*M_PI*self.f.c_get_value() + _sum)

        #: ampdot
        self.amp_dot.c_set_value(
            self.a.c_get_value()*(self.R.c_get_value() - _amp)
        )

    cdef void c_output(self) nogil:
        """ Neuron output. """
        self.nout.c_set_value(self.phase.c_get_value())

    cdef double c_neuron_inputs_eval(
            self, double _neuron_out, double _weight, double _phi,
            double _phase, double _amp) nogil:
        """ Evaluate neuron inputs."""
        return _weight*_amp*csin(_neuron_out - _phase - _phi)

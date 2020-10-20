"""Oscillator model."""

from farms_container.parameter cimport Parameter
from farms_network.neuron cimport Neuron

cdef struct OscillatorNeuronInput:
    int neuron_idx
    int weight_idx
    int phi_idx

cdef class Oscillator(Neuron):
    cdef:
        readonly str n_id

        unsigned int num_inputs

        #: parameters
        #: constants
        Parameter f
        Parameter R
        Parameter a

        #: states
        Parameter phase
        Parameter amp

        #: inputs
        Parameter ext_in

        #: ode
        Parameter phase_dot
        Parameter amp_dot

        #: Ouputs
        Parameter nout

        #: neuron connenctions
        OscillatorNeuronInput[:] neuron_inputs

    cdef:
        void c_ode_rhs(self, double[:] _y, double[:] _w, double[:] _p) nogil
        void c_output(self) nogil
        cdef double c_neuron_inputs_eval(
            self, double _neuron_out, double _weight, double _phi,
            double _phase, double _amp) nogil

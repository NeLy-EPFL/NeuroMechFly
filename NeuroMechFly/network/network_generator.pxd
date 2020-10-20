from farms_network.leaky_integrator cimport LeakyIntegrator
from farms_container.table cimport Table
from farms_network.neuron cimport Neuron


cdef class NetworkGenerator(object):
    cdef:
        dict __dict__
        Neuron[:] c_neurons
        Table states
        Table dstates
        Table constants
        Table inputs
        Table weights
        Table parameters
        Table outputs

        unsigned int num_neurons
    cdef:
        # void c_step(self, double[:] inputs)
        double[:] c_ode(self, double t, double[:] state)

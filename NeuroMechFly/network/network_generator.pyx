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

""" Generate neural network. """
from libc.stdio cimport printf
from farms_network.neuron cimport Neuron
from farms_container.table cimport Table
from cython.parallel import prange
from farms_network.leaky_integrator cimport LeakyIntegrator
import farms_pylog as pylog
from farms_network.neuron_factory import NeuronFactory
from collections import OrderedDict
import numpy as np
cimport numpy as cnp
cimport cython
pylog.set_level('debug')


cdef class NetworkGenerator:
    """ Generate Neural Network.
    """

    def __init__(self, graph, neural_container):
        """Initialize.

        Parameters
        ----------
        graph_file_path: <str>
            File path to the graphml structure.
        """
        super(NetworkGenerator, self).__init__()

        #: Attributes
        self.neurons = OrderedDict()  #: Neurons in the network
        self.states = <Table > neural_container.add_table('states')
        self.dstates = <Table > neural_container.add_table('dstates')
        self.constants = <Table > neural_container.add_table(
            'constants', table_type='CONSTANT')
        self.inputs = <Table > neural_container.add_table('inputs')
        self.weights = <Table > neural_container.add_table('weights')
        self.parameters = <Table > neural_container.add_table('parameters')
        self.outputs = <Table > neural_container.add_table('outputs')

        self.odes = []

        self.fin = {}
        self.integrator = {}

        #:  Read the graph
        self.graph = graph

        #: Get the number of neurons in the model
        self.num_neurons = len(self.graph)

        self.c_neurons = np.ndarray((self.num_neurons,), dtype=Neuron)
        self.generate_neurons(neural_container)
        self.generate_network(neural_container)

    def generate_neurons(self, neural_container):
        """Generate the complete neural network.
        Instatiate a neuron model for each node in the graph

        Returns
        -------
        out : <bool>
            Return true if successfully created the neurons
        """
        cdef int j
        for j, (name, neuron) in enumerate(sorted(self.graph.nodes.items())):
            #: Add neuron to list
            pylog.debug(
                'Generating neuron model : {} of type {}'.format(
                    name, neuron['model']))
            #: Generate Neuron Models
            _neuron = NeuronFactory.gen_neuron(neuron['model'])
            self.neurons[name] = _neuron(
                name, self.graph.in_degree(name),
                neural_container,
                **neuron
            )
            self.c_neurons[j] = <Neuron > self.neurons[name]

    def generate_network(self, neural_container):
        """
        Generate the network.
        """
        for name, neuron in list(self.neurons.items()):
            pylog.debug(
                'Establishing neuron {} network connections'.format(
                    name))
            for j, pred in enumerate(self.graph.predecessors(name)):
                pylog.debug(('{} -> {}'.format(pred, name)))
                #: Set the weight of the parameter
                neuron.add_ode_input(
                    j,
                    self.neurons[pred],
                    neural_container,
                    **self.graph[pred][name])

    def ode(self, t, cnp.ndarray[double, ndim=1] state):
        return self.c_ode(t, state)

    #################### C-FUNCTIONS ####################

    cdef double[:] c_ode(self, double t, double[:] state):
        self.states.c_set_values(state)
        cdef unsigned int j

        for j in range(self.num_neurons):
            (<Neuron>self.c_neurons[j]).c_output()

        for j in range(self.num_neurons):
            (<Neuron>self.c_neurons[j]).c_ode_rhs(
                self.outputs.c_get_values(),
                self.weights.c_get_values(),
                self.parameters.c_get_values()
            )
        return self.dstates.c_get_values()

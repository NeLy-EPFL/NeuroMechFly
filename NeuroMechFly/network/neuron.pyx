import os


cdef class Neuron:
    """Base neuron class.
    """

    def __init__(self, model_type):
        super(Neuron, self).__init__()
        self._model_type = model_type  # : Type of neuron  @property

    def add_ode_input(self, neuron, neural_container, **kwargs):
        """Add relevant external inputs to the ode.
        Parameters
        ----------
        neuron : <LIF_Danner>
            Neuron model from which the input is received.
        kwargs : <dict>
             Contains the weight/synaptic information from the receiving neuron.
        """
        print(
            'add_ode_input : Method not implemented in Neuron child class')
        raise NotImplementedError()

    def ode_rhs(self, y, w, p):
        """ ODE RHS.
        Returns
        ----------
        ode_rhs: <list>
            List containing the rhs equations of the ode states in the system
        """
        print('ode_rhs : Method not implemented in Neuron child class')
        raise NotImplementedError()

    def output(self):
        """ Output of the neuron model.
        Returns
        ----------
        out: <cas.SX.sym>
            Output of the neuron  model
        """
        print('output : Method not implemented in Neuron child class')
        raise NotImplementedError()

    #################### PROPERTIES ####################
    @property
    def model_type(self):
        """Neuron type.  """
        return self._model_type

    @model_type.setter
    def model_type(self, value):
        """
        Parameters
        ----------
        value : <str>
            Type of neuron model
        """
        self._model_type = value

    #################### C-FUNCTIONS ####################
    cdef void c_ode_rhs(self, double[:] _y, double[:] _w, double[:] _p) nogil:
        pass

    cdef void c_output(self) nogil:
        pass

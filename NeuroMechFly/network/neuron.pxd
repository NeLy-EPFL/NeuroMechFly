""" Header for Neuron Base Class. """

cdef class Neuron:
    """Base neuron class.
    """

    cdef:
        str _model_type

    cdef:
        void c_ode_rhs(self, double[:] _y, double[:] _w, double[:] _p) nogil
        void c_output(self) nogil

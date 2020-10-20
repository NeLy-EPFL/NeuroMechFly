# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: overflowcheck=False


import numpy as np

DTYPE = np.float64

cdef class Parameter(object):
    """ Wrapper for parameters in the system. """

    def __init__(self, name, value, idx):
        """ Initialization. """
        super(Parameter, self).__init__()

        #: Update the value to the list
        self._val = value

        #: Name
        self._name = name

        #: IDX
        self._idx = idx

    #################### PROPERTIES ####################

    @property
    def value(self):
        """ Get the value  """
        return self.c_get_value()

    @value.setter
    def value(self, data):
        """
        Set the value
        """
        self.c_set_value(data)

    @property
    def prev_value(self):
        """ Get the value  """
        return self.c_get_prev_value()

    @property
    def name(self):
        """ Get the name of the attribute  """
        return self._name

    @property
    def idx(self):
        """ Get the index of the attribute in the data table. """
        return self._idx

    #################### C-FUNCTIONS ####################

    cdef void c_set_memory_view(self, double[:, :] data_table,
                                long int * curr_index):
        """ Set the memory view for the data object. """
        self.data = data_table[:, self._idx]
        self.curr_index = curr_index
        #: Set the value in the table
        self.value = self._val

    cdef inline double c_get_value(self) nogil:
        return self.data[self.curr_index[0]]

    cdef double c_set_value(self, double value) nogil:
        self.data[self.curr_index[0]] = value

    cdef double c_get_prev_value(self) nogil:
        cdef long int idx
        if self.curr_index[0] == 0:
            idx = 0
        else:
            idx = self.curr_index[0] - 1
        return self.data[idx]

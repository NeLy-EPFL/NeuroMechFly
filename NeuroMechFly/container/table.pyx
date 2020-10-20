# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: overflowcheck=False

"""Parameters book keeping class"""

from collections import OrderedDict
import numpy as np
import farms_pylog as pylog
from .parameter cimport Parameter

DTYPE = np.float64

cdef class Table(list):
    """Table for storing parameters"""

    def __init__(self, name, table_type, max_iterations):
        """ Initialization. """
        super(Table, self).__init__()
        self.name = name
        self.table_type = table_type
        if table_type == 'VARIABLE':
            self.max_iterations = max_iterations
        elif table_type == 'CONSTANT':
            self.max_iterations = 1
        else:
            pylog.error('Unkown table of type {}'.format(table_type))
            raise TypeError
        self._name_to_idx = {}
        self._names = []
        self.current_idx = 0

    def initialize_table(self):
         """ Initialization of the parameters array. """         
         self.c_initialize_table()

    def add_parameter(self, name, value=0.0):
        _idx = len(self)
        cdef Parameter parameter = Parameter(name, value, _idx)
        self.append(parameter)
        self._name_to_idx[name] = _idx
        self._names.append(name)
        return parameter, value

    def get_parameter_value(self, name):
        """ Get the value of the parameter in the table. """
        return (self.get_parameter(name)).value

    def get_parameter_index(self, name):
        """ Get the index of the parameter in the table. """
        return self._name_to_idx.get(name, None)

    def get_parameter(self, name):
        """ Get the access to the param. """
        return self[self.get_parameter_index(name)]

    def update_log(self):
        """ Update the curr index. """
        #: TO DO : ADD CHECK FOR MAX ITERATIONS
        self.c_update_log()
        if self.buffer_full:
            pylog.debug(
                "Logging Buffer exceeded for {}!".format(self.name))

    #################### PROPERTIES ####################

    @property
    def values(self):
        """ Get the values of the parameters  """
        return self.c_get_values()

    @values.setter
    def values(self, values):
        """ Set the values of the param data  """
        self.c_set_values(values)

    @property
    def log(self):
        """Get the complete buffer  """
        return np.array(self.c_get_buffer())

    @property
    def name_index(self):
        """ Get all the parameter names and their indices. """
        return self._name_to_idx

    @property
    def names(self):
        """ Get all the parameter names. """
        return self._names

    #################### C-FUNCTIONS ####################
    cdef void c_initialize_table(self):
        """ Initialization table of parameters. """
        #: Create a table of size [num_iterations, num_parameters]
        cdef int array_len = len(self)
        #:
        if array_len == 0:
            pylog.warning(
                "No parameters of type : {}!!!".format(self.name))

        if self.max_iterations == 1:
            self.data_table = (
                np.zeros((1, array_len), dtype=DTYPE))
        else:
            self.data_table = (
                np.zeros((self.max_iterations, array_len), dtype=DTYPE))
        cdef Parameter p
        for p in self:
            p.c_set_memory_view(self.data_table, & self.current_idx)

    cdef double[:] c_get_values(self) nogil:
        """ Get the values. """
        return self.data_table[self.current_idx, :]

    cdef void c_set_values(self, double[:] values) nogil:
        """ Set the values. """
        self.data_table[self.current_idx, :] = values[:]

    cdef double[:, :] c_get_buffer(self):
        """Get teh complete buffer."""
        return self.data_table[:, :]

    cdef void c_update_log(self) nogil:
        if self.max_iterations == 1:
            self.current_idx = 0
        else:
            if self.current_idx < self.max_iterations-1:
                self.current_idx += 1
            else:
                self.buffer_full = 1
            self.data_table[self.current_idx,
                            :] = self.data_table[self.current_idx-1, :]

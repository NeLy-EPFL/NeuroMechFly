#: Types
ctypedef double real

############### PARAMETER ###############

cdef class Parameter(object):
    cdef:
        double _val
        str _name
        readonly unsigned int _idx
        double[:] data
        long int * curr_index

    cdef:
        inline double c_get_value(self) nogil
        double c_set_value(self, double value) nogil
        double c_get_prev_value(self) nogil
        void c_set_memory_view(self, double[:, :] data_table,
                               long int * curr_index)

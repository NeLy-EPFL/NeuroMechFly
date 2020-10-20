"""Table to keep parameters"""

############### PARAMETERS ###############

cdef class Table(list):
    cdef:
        str name
        dict _name_to_idx
        list _names
        str table_type
        long int max_iterations
        long int current_idx
        int buffer_full
        double[:, :] data_table

    cdef:
        void c_initialize_table(self)
        double[:] c_get_values(self) nogil
        void c_set_values(self, double[:] values) nogil
        double[:, :] c_get_buffer(self)
        void c_update_log(self) nogil

from farms_container.table cimport Table

cdef class ContactSensors:
    """ Contact sensors """
    cdef Table contact_normal_force
    cdef Table contact_lateral_force
    cdef Table contact_position

    cdef public tuple contact_ids

    cdef public double imeters
    cdef public double inewtons

    cpdef void update(self)

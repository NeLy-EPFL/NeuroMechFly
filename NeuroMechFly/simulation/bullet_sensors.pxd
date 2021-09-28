from farms_container.table cimport Table


cdef struct ARRAY3:
    double x
    double y
    double z


cdef inline ARRAY3 tuple_to_struct(tuple data):
    """ Convert tuple to struct object for forces """
    cdef ARRAY3 struct_data
    struct_data.x = data[0]
    struct_data.y = data[1]
    struct_data.z = data[2]
    return struct_data


cdef class ContactSensors:
    """ Contact sensors """
    cdef Table contact_flag
    cdef Table contact_normal_force
    cdef Table contact_lateral_force
    cdef Table contact_position

    cdef public tuple contact_ids
    cdef public tuple ground_contact_indices
    cdef public tuple self_contact_indices

    cdef public double imeters
    cdef public double inewtons

    cpdef void update(self)


cdef class COMSensor:
    """ Center of mass sensor"""
    cdef Table center_of_mass
    cdef unsigned int model_id
    cdef unsigned int num_links
    cdef double imeters
    cdef double ikilograms

    cdef readonly double total_mass
    cdef readonly double[:] link_masses

    cpdef void update(self)

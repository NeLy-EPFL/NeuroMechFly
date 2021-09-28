""" Helper functions for bullet to speed up code """


import numpy as np
import pybullet as p

cimport cython
cimport numpy as np


cdef class ContactSensors:
    """ Contact sensors """

    def __init__(
            self, sim_data, contact_ids, # self_collisions_ids,
            meters=1, newtons=1
    ):
        self.contact_ids = contact_ids
        self.ground_contact_indices = tuple([
            index
            for index, contact_id in enumerate(contact_ids)
            if contact_id[0] != contact_id[1]
        ])
        self.self_contact_indices = tuple([
            index
            for index, contact_id in enumerate(contact_ids)
            if contact_id[0] == contact_id[1]
        ])
        self.contact_flag = sim_data.contact_flag
        self.contact_position = sim_data.contact_position
        self.contact_normal_force = sim_data.contact_normal_force
        self.contact_lateral_force = sim_data.contact_lateral_force
        # Units
        self.imeters = 1./meters
        self.inewtons = 1./newtons

    @cython.nonecheck(False)
    cpdef void update(self):
        """ Update contacts """
        cdef double inewtons = self.inewtons
        cdef double imeters = self.imeters
        cdef tuple contact
        cdef tuple contacts
        cdef unsigned int stride = 3
        cdef unsigned int sensor_index = 0
        cdef double rx, ry, rz, fx, fy, fz, px, py, pz
        cdef double rx_tot, ry_tot, rz_tot, fx_tot, fy_tot, fz_tot
        cdef int model_A, model_B, link_A, link_B

        cdef double[:] contact_flag_data
        cdef double[:] contact_position_data
        cdef double[:] contact_normal_data
        cdef double[:] contact_lateral_data

        cdef ARRAY3 normal_force_vec
        cdef double normal_force

        cdef ARRAY3 lateral_force_vec1
        cdef double lateral_force1

        cdef ARRAY3 lateral_force_vec2
        cdef double lateral_force2

        cdef ARRAY3 position

        for model_A, model_B, link_A, link_B in self.contact_ids:
            px = 0.0
            py = 0.0
            pz = 0.0
            rx_tot = 0.0
            ry_tot = 0.0
            rz_tot = 0.0
            fx_tot = 0.0
            fy_tot = 0.0
            fz_tot = 0.0
            contacts = p.getContactPoints(model_A, model_B, link_A, link_B)
            for contact in contacts:
                # Normal reaction
                normal_force_vec = tuple_to_struct(contact[7])
                normal_force = contact[9]
                rx = normal_force*normal_force_vec.x*inewtons
                ry = normal_force*normal_force_vec.y*inewtons
                rz = normal_force*normal_force_vec.z*inewtons
                rx_tot += rx
                ry_tot += ry
                rz_tot += rz
                # Lateral friction dir 1 + Lateral friction dir 2
                lateral_force_vec1 = tuple_to_struct(contact[11])
                lateral_force1 = contact[10]
                lateral_force_vec2 = tuple_to_struct(contact[13])
                lateral_force2 = contact[12]
                fx = (lateral_force1*lateral_force_vec1.x + lateral_force2*lateral_force_vec2.x)*inewtons
                fy = (lateral_force1*lateral_force_vec1.y + lateral_force2*lateral_force_vec2.y)*inewtons
                fz = (lateral_force1*lateral_force_vec1.z + lateral_force2*lateral_force_vec2.z)*inewtons
                fx_tot += fx
                fy_tot += fy
                fz_tot += fz
                # TODO: Check this computation
                # Position
                position = tuple_to_struct(contact[5])
                px += (rx+fx)*position.x*imeters
                py += (ry+fy)*position.y*imeters
                pz += (rz+fz)*position.z*imeters
            # Update contact flag
            contact_flag_data = self.contact_flag.c_get_values()
            if len(contacts) > 0:
                contact_flag_data[sensor_index] = 1.0
            else:
                contact_flag_data[sensor_index] = 0.0
            # Update data
            contact_normal_data = (
                self.contact_normal_force.c_get_values()[
                    stride*sensor_index:(stride*sensor_index)+stride
                ]
            )
            contact_lateral_data = (
                self.contact_lateral_force.c_get_values()[
                    stride*sensor_index:(stride*sensor_index)+stride
                ]
            )
            contact_position_data = (
                self.contact_position.c_get_values()[
                    stride*sensor_index:(stride*sensor_index)+stride
                ]
            )
            # normal
            contact_normal_data[0] = rx_tot
            contact_normal_data[1] = ry_tot
            contact_normal_data[2] = rz_tot
            # lateral
            contact_lateral_data[0] = fx_tot
            contact_lateral_data[1] = fy_tot
            contact_lateral_data[2] = fz_tot
            # position
            contact_position_data[0] = px
            contact_position_data[1] = py
            contact_position_data[2] = pz
            # Update sensor index
            sensor_index += 1


cdef class COMSensor:
    """ Center of Mass sensor """

    def __init__(self, model_id, sim_data, meters=1, kilograms=1):
        # Model id
        self.model_id = model_id
        # Units
        self.imeters = 1./meters
        self.ikilograms = 1./kilograms
        # Number of links in the model
        self.num_links = p.getNumJoints(self.model_id)
        # Initialize masses
        cdef int link_index
        self.link_masses = np.asarray(
            [p.getDynamicsInfo(self.model_id, link_index)[0]
             for link_index in np.arange(-1, self.num_links)],
            dtype=np.double
        )
        self.total_mass = np.sum(self.link_masses)*self.ikilograms
        # Data
        self.center_of_mass = sim_data.center_of_mass

    cpdef void update(self):
        """ Update center of mass """
        link_states = p.getLinkStates(
            self.model_id, range(self.num_links)
        )
        # Declarations
        cdef ARRAY3 com
        cdef double mass
        cdef double[:] center_of_mass_data
        # Base
        cdef ARRAY3 base_position = tuple_to_struct(
            p.getBasePositionAndOrientation(self.model_id)[0]
        )
        com.x = base_position.x*self.link_masses[0]
        com.y = base_position.y*self.link_masses[0]
        com.z = base_position.z*self.link_masses[0]
        # Links
        cdef int link_id
        cdef ARRAY3 link_com
        for link_id, state in enumerate(link_states):
            mass = self.link_masses[link_id+1]
            link_com = tuple_to_struct(state[0])
            com.x += link_com.x*mass
            com.y += link_com.y*mass
            com.z += link_com.z*mass
        # Update the data
        center_of_mass_data = self.center_of_mass.c_get_values()
        center_of_mass_data[0] = com.x*self.imeters/self.total_mass
        center_of_mass_data[1] = com.y*self.imeters/self.total_mass
        center_of_mass_data[2] = com.z*self.imeters/self.total_mass

# cython: linetrace=True

""" Helper functions for bullet to speed up code """


import numpy as np
import pybullet as p

cimport cython
cimport numpy as np

DTYPE = np.double

ctypedef np.double_t DTYPE_t


cdef class ContactSensors:
    """ Contact sensors """

    def __init__(
            self, sim_data, contact_ids, # self_collisions_ids,
            meters=1, newtons=1
    ):
        self.contact_ids = contact_ids
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
        cdef unsigned int stride = 3
        cdef unsigned int sensor_index = 0
        cdef double rx, ry, rz, fx, fy, fz, px, py, pz
        cdef double rx_tot, ry_tot, rz_tot, fx_tot, fy_tot, fz_tot
        cdef int model_A, model_B, link_A, link_B

        cdef double[:] contact_position_data
        cdef double[:] contact_normal_data
        cdef double[:] contact_lateral_data

        for model_A, model_B, link_A, link_B in self.contact_ids:
            px = 0
            py = 0
            pz = 0
            rx_tot = 0
            ry_tot = 0
            rz_tot = 0
            fx_tot = 0
            fy_tot = 0
            fz_tot = 0
            for contact in p.getContactPoints(
                    model_A, model_B, link_A, link_B
            ):
                # Normal reaction
                rx = contact[9]*contact[7][0]*inewtons
                ry = contact[9]*contact[7][1]*inewtons
                rz = contact[9]*contact[7][2]*inewtons
                rx_tot += rx
                ry_tot += ry
                rz_tot += rz
                # Lateral friction dir 1 + Lateral friction dir 2
                fx = (contact[10]*contact[11][0] + contact[12]*contact[13][0])*inewtons
                fy = (contact[10]*contact[11][1] + contact[12]*contact[13][1])*inewtons
                fz = (contact[10]*contact[11][2] + contact[12]*contact[13][2])*inewtons
                fx_tot += fx
                fy_tot += fy
                fz_tot += fz
                # TODO: Check this computation
                # Position
                px += (rx+fx)*contact[5][0]*imeters
                py += (ry+fy)*contact[5][1]*imeters
                pz += (rz+fz)*contact[5][2]*imeters
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

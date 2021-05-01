"""
-----------------------------------------------------------------------
Copyright 2018-2020 Jonathan Arreguit, Shravan Tata Ramalingasetty
Copyright 2018 BioRobotics Laboratory, École polytechnique fédérale de Lausanne

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-----------------------------------------------------------------------

Module to apply units

"""

from ..utils.options import Options


class SimulationUnitScaling(Options):
    """Simulation scaling

    1 [m] in reality = self.meterss [m] in simulation
    1 [s] in reality = self.seconds [s] in simulation
    1 [kg] in reality = self.kilograms [kg] in simulation

    """

    def __init__(self, meters=1, seconds=1, kilograms=1):
        super(SimulationUnitScaling, self).__init__()
        self.meters = meters
        self.seconds = seconds
        self.kilograms = kilograms

    @property
    def hertz(self):
        """Hertz (frequency)

        Scaled as self.hertz = 1/self.seconds

        """
        return 1./self.seconds

    @property
    def newtons(self):
        """Newtons

        Scaled as self.newtons = self.kilograms*self.meters/self.time**2

        """
        return self.kilograms*self.acceleration

    @property
    def torques(self):
        """Torques

        Scaled as self.torques = self.kilograms*self.meters**2/self.time**2

        """
        return self.newtons*self.meters

    @property
    def velocity(self):
        """Velocity

        Scaled as self.velocities = self.meters/self.seconds

        """
        return self.meters/self.seconds

    @property
    def acceleration(self):
        """Acceleration

        Scaled as self.gravity = self.meters/self.seconds**2

        """
        return self.velocity/self.seconds

    @property
    def gravity(self):
        """Gravity

        Scaled as self.gravity = self.meters/self.seconds**2

        """
        return self.acceleration

    @property
    def volume(self):
        """Volume

        Scaled as self.volume = self.meters**3

        """
        return self.meters**3

    @property
    def density(self):
        """Density

        Scaled as self.density = self.kilograms/self.meters**3

        """
        return self.kilograms/self.volume

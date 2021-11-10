""" This scripts takes the default NeuroMechFly template SDF and replaces the links with cylinders.

To make further changes, simply replace the link name and provide the size and name of the new geometric shape.
"""

from pathlib import Path

from NeuroMechFly.sdf.sdf import ModelSDF, Link, Collision, Visual
from NeuroMechFly.sdf.units import SimulationUnitScaling
from NeuroMechFly.sdf import utils


# Global config paths
SCRIPT_PATH = Path(__file__).parent.absolute()
DATA_PATH = SCRIPT_PATH.joinpath("..", "..", "data")
CONFIG_PATH = DATA_PATH.joinpath("config")
POSE_CONFIG_PATH = CONFIG_PATH.joinpath("pose")
SDF_MODEL_PATH = DATA_PATH.joinpath("design", "sdf")

units = SimulationUnitScaling(
    meters=1e0,
    seconds=1e0,
    kilograms=1e0
)


def generate_new_sdf(model, radius, new_model_name, **kwargs):
    """ Replaces forelegs and antennae with cylinders.

    Parameters
    ----------
    model : ModelSDF.read
        SDF to be changed.
    radius : float
        Radius of the cylinders that will replace forelegs
    new_model_name : str
        Name of the new SDF file.
    """
    link_index = utils.link_name_to_index(model)
    joint_index = utils.joint_name_to_index(model)

    leg_segments = {
        'Coxa': 0.425,
        'Femur': 0.706,
        'Tibia': 0.518,
        'Tarsus1': 0.663,
        'Tarsus4': 0,
        'Tarsus3': 0,
        'Tarsus2': 0,
        'Tarsus5': 0
    }

    length_antenna = 0.297
    radius_antenna = kwargs.get('radius_antenna', 0.06)

    for segment_name, length in leg_segments.items():
        for side in ('R', 'L'):
            link_name = f'{side}F{segment_name}'
            if length == 0:
                model.links[link_index[link_name]].visuals = [
                    Link.empty(
                        name=link_name,
                        pose=[0, 0, 0, 0, 0, 0],
                        units=units
                    )
                ]
                model.links[link_index[link_name]].collisions = [
                    Link.empty(
                        name=link_name,
                        pose=[0, 0, 0, 0, 0, 0],
                        units=units
                    )
                ]
            else:
                model.links[link_index[link_name]].visuals = [
                    Visual.cylinder(
                        name=link_name,
                        radius=radius,
                        length=length,
                        pose=[0, 0, -length / 2, 0, 0, 0],
                        units=units
                    )
                ]

                model.links[link_index[link_name]].collisions = [
                    Collision.cylinder(
                        name=link_name,
                        radius=radius,
                        length=length,
                        pose=[0, 0, -length / 2, 0, 0, 0],
                        units=units
                    )
                ]

            model.links[link_index[f'{side}Antenna']].visuals = [
                Visual.cylinder(
                    name=f'{side}Antenna',
                    radius=radius_antenna,
                    length=length_antenna,
                    pose=[0, 0, -length_antenna / 2, 0, 0, 0],
                    units=units
                )
            ]

            model.links[link_index[f'{side}Antenna']].collisions = [
                Collision.cylinder(
                    name=f'{side}Antenna',
                    radius=radius_antenna,
                    length=length_antenna,
                    pose=[0, 0, -length_antenna / 2, 0, 0, 0],
                    units=units
                )
            ]

    model.write(SDF_MODEL_PATH.joinpath(new_model_name))


if __name__ == '__main__':
    """ Main. """

    # Read the sdf model from template
    model = ModelSDF.read(SDF_MODEL_PATH.joinpath(
        "neuromechfly_noLimits.sdf"))[0]

    # New model name
    new_model_name = "neuromechfly_frontleg_cylinder.sdf"

    # Set the radius of the cylinders that will replace the forelimbs
    radius = 0.096
    generate_new_sdf(model, radius, new_model_name)

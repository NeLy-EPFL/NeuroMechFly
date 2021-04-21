""" Generate joints figure """

import os
import pathlib
import sys

import numpy as np
import yaml

import bpy
import mathutils
from farms_blender.core.camera import create_multiview_camera
from farms_blender.core.collections import add_object_to_collection
from farms_blender.core.display import display_farms_types
from farms_blender.core.freestyle import (configure_lineset, create_lineset,
                                          default_lineset_config,
                                          enable_freestyle, remove_lineset,
                                          set_freestyle_setting)
from farms_blender.core.materials import farms_material, flat_shade_material
from farms_blender.core.objects import objs_of_farms_types
from farms_blender.core.materials import create_material
from farms_blender.core.pose import set_model_pose
from farms_blender.core.primitives import create_sphere
from farms_blender.core.resources import get_resource_object
from farms_blender.core.scene import set_scene_settings
from farms_blender.core.sdf import get_base_link, load_sdf
from farms_blender.core.utils import clear_world
from farms_models.utils import get_sdf_path

sys.path.append(bpy.data.filepath)
SCRIPT_PATH = pathlib.Path(__file__).parent.absolute()


def add_floor(**kwargs):
    """Add floor"""
    floor_offset = kwargs.pop('floor_offset', [-0.05, 0.0, 0.0])
    name, links_map, joints_map, objs = load_sdf(
        get_sdf_path(
            name='arena_flat',
            version='v0',
        ),
        position=kwargs.pop('position', floor_offset),
        texture='BIOROB2_blue.png',
        texture_collision=True,
    )
    scale = kwargs.pop('scale', 1.)
    objs[0].scale = [scale]*3


def add_lights(**kwargs):
    """ Add lights """
    #: Add sun
    sun = bpy.data.lights.new(name="sun", type='SUN')
    sun.energy = kwargs.pop('sun_energy', 4.0)
    sun.angle = kwargs.pop('sun_angle', 3.14)
    sun_obj = bpy.data.objects.new(name="sun", object_data=sun)
    bpy.context.collection.objects.link(sun_obj)


def configure_freestyle(objs=None, **kwargs):
    """ Configure line style settings. """
    #: Line thickness
    bpy.data.scenes['Scene'].render.line_thickness = kwargs.pop(
        'line_thickness', 1.0
    )

    #: Create collections for visual objects
    if objs:
        for obj in objs:
            add_object_to_collection(obj, collection='freestyle', create=True)
    else:
        for obj, _ in objs_of_farms_types(visual=True, link=True):
            add_object_to_collection(obj, collection='freestyle', create=True)

    #: Genric settings
    set_freestyle_setting('crease_angle', 0)
    #: Remove default lineset
    remove_lineset('LineSet')
    #: Create line sets for each elem
    linesets = {
        name: create_lineset(name)
        for name in ["visuals", ]
    }
    lineset_config = default_lineset_config()
    lineset_config['select_by_visibility'] = kwargs.pop(
        "select_by_visibility", True
    )
    lineset_config['select_by_collection'] = kwargs.pop(
        "select_by_collection", True
    )
    lineset_config['select_contour'] = kwargs.pop(
        "select_contour", True
    )
    lineset_config['select_silhouette'] = kwargs.pop(
        "select_silhouette", False
    )
    lineset_config['select_crease'] = kwargs.pop(
        "select_crease", False
    )
    lineset_config["select_border"] = kwargs.pop(
        "select_border", False
    )
    lineset_config['select_external_contour'] = kwargs.pop(
        "select_external_contour", True
    )

    for name, lineset in linesets.items():
        lineset_config["collection"] = bpy.data.collections['freestyle']
        configure_lineset(lineset, **lineset_config)


def scene(**kwargs):
    """ scene """
    if kwargs.pop("clear_world", True):
        #: Clear world
        clear_world()
    if kwargs.pop("add_floor", True):
        #: Add floor
        add_floor(**kwargs)
    if kwargs.pop("add_lights", True):
        #: Add light
        add_lights(**kwargs)
    #: Set render settings
    set_scene_settings(renderer='BLENDER_EEVEE')
    #: background light
    bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[1].default_value = 1.0
    #: Render size
    bpy.context.scene.render.resolution_x = kwargs.pop(
        "resolution_x", 1920
    )
    bpy.context.scene.render.resolution_y = kwargs.pop(
        "resolution_y", 1080
    )
    #: Render samples
    bpy.context.scene.eevee.taa_render_samples = kwargs.pop(
        "render_samples", 64
    )
    #: Disable background
    bpy.data.scenes['Scene'].render.film_transparent = kwargs.pop(
        "film_transparent", True
    )


def load_fly(**kwargs):
    """Load fly model

    Parameters
    ----------
    **kwargs :

    Returns
    -------
    out : model_name, objs

    """
    model_offset = kwargs.pop('model_offset', (0.0, 0.0, 0.0))
    # FIXME: Remove the hard-coded path
    model_name, *_, objs = load_sdf(
        path="../sdf/neuromechfly_limitsFromData.sdf",
        **kwargs
    )
    base_link = get_base_link(model_name)
    base_link.location = model_offset
    return model_name, objs


def apply_materials():
    """ Apply default materials to the model."""

    wings = create_material(name="wings", color=(0.4, 0.4, 0.4, 0.4))
    eyes = create_material(name="eyes", color=(0.47, 0.035, 0.015, 1.0))
    body = create_material(name="body", color=(0.41, 0.2, 0.033, 1.0))
    legs = create_material(name="legs", color=(0.55, 0.3, 0.08, 1.0))

    # Get visual objects
    objs = {
        obj.name: obj
        for obj, _ in objs_of_farms_types(visual=True)
    }

    for name, obj in objs.items():
        if 'Wing' in name:
            obj.active_material = wings
        elif 'Eye' in name:
            obj.active_material = eyes
        elif any([link in name for link in ('Tarsus', 'Tibia', 'Femur', 'Coxa')]):
            obj.active_material = legs
        else:
            obj.active_material = body


def configure_scene(**kwargs):
    """ Configure scene. """
    #: Disable background
    bpy.data.scenes['Scene'].render.film_transparent = True
    #: Enable freestyle
    if kwargs.pop("freestyle", False):
        enable_freestyle()


def add_cameras():
    """Add cameras """
    camera_options = {
        "loc": (12.4, -0.165, -0.035), "rot": (np.pi/2, 0., np.pi/2.),
        "type": 'PERSP', "lens": 50, "scale": 1.0
    }
    #: Create camera 0
    camera_front = create_multiview_camera(
        0, camera_options
    )

    #: Create camera 1
    camera_options['loc'] = (-0.43, 10.78, -0.035)
    camera_options['rot'] = (np.pi/2, 0., np.pi)
    camera_side = create_multiview_camera(
        1, camera_options
    )
    return (camera_front, camera_side)


def render_leg(side='R', leg='F', cameras=None, camera_options=None):
    """ Render leg """

    leg_segments = [
        'Coxa', 'Femur', 'Tibia', 'Tarsus1', 'Tarsus2', 'Tarsus3',
        'Tarsus4', 'Tarsus5'
    ]

    objs = {
        obj.name: obj
        for obj, _ in objs_of_farms_types(visual=True, joint_axis=True)
        if f"{side}{leg}" in obj.name
    }

    display_farms_types(objs=objs.values(), visual=True, joint_axis=True)

    for camera in cameras:
        #: Choose camera
        bpy.data.scenes['Scene'].camera = camera

        # render settings
        bpy.context.scene.render.filepath = os.path.join(
            SCRIPT_PATH,
            f"./leg_joints_{camera.name}.png"
        )
        bpy.ops.render.render(write_still=1)


def render_whole_body(cameras=None, camera_options=None):
    """ Render leg """

    objs = {
        obj.name: obj
        for obj, _ in objs_of_farms_types(visual=True, joint_axis=True)
    }

    display_farms_types(objs=objs.values(), visual=True, joint_axis=True)

    # Hide prismatic joints
    for name, obj in objs.items():
        if 'support' in name:
            obj.hide_set(False)
            obj.hide_render = False

    for camera in cameras:
        #: Choose camera
        bpy.data.scenes['Scene'].camera = camera

        # render settings
        bpy.context.scene.render.filepath = os.path.join(
            SCRIPT_PATH,
            f"./whole_body_{camera.name}.png"
        )
        bpy.ops.render.render(write_still=1)


def main():
    """ main """
    #: Load default scene
    scene(
        add_floor=False, scale=1e-2, resolution_x=1920, resolution_y=1080,
        render_samples=16
    )
    #: Configure scene
    configure_scene(objs=[
        obj
        for obj, _ in objs_of_farms_types(joint_axis=True)
    ])
    #: Load model
    model_name, objs = load_fly(
        model_offset=(0.0, 0.0, 0.0), resources_scale=0.2
    )
    #: add cameras
    cameras = add_cameras()
    #: Add resource object
    world_axis = get_resource_object(resource_object='link', name="world")
    world_axis.scale = [0.25]*3
    world_axis.location = (0.71, -0.73, -1.13)
    #: Configure freestyle
    # configure_freestyle(
    #     objs=[
    #         world_axis, *[
    #             obj
    #             for obj, _ in objs_of_farms_types(joint_axis=True)
    #         ]
    #     ]
    # )
    #: Display
    display = {
        'link': False,
        'visual': False,
        'collision': False,
        'inertial': False,
        'com': False,
        'muscle': False,
        'joint': False,
        'joint_axis': False,
    }
    #: Apply materials and color
    apply_materials()
    #: Hide all
    display_farms_types(objs=objs, **display)
    #: Render leg
    render_whole_body(cameras=cameras)


if __name__ == '__main__':
    main()

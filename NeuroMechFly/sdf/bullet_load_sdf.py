"""SDF"""
import farms_pylog as pylog
import os
import numpy as np
import pybullet
from NeuroMechFly.sdf.sdf import (
    ModelSDF,
    Plane,
    Box,
    Sphere,
    Cylinder,
    Capsule,
    Mesh,
    # Heightmap,
    Collision,
)


def rot_quat(rot):
    """Quaternion from Euler"""
    return pybullet.getQuaternionFromEuler(rot)


def rot_matrix(rot):
    """Matrix from Quaternion"""
    return np.array(pybullet.getMatrixFromQuaternion(rot)).reshape([3, 3])


def rot_invert(rot):
    """Invert rot"""
    return pybullet.invertTransform(
        [0, 0, 0],
        rot,
    )[1]


def rot_mult(rot0, rot1):
    """Rotation Multiplication"""
    return pybullet.multiplyTransforms([0, 0, 0], rot0, [0, 0, 0], rot1)[1]


def rot_diff(rot0, rot1):
    """Rotation difference"""
    return pybullet.getDifferenceQuaternion(rot0, rot1)


def pybullet_options_from_shape(shape, path='', force_concave=False):
    """Pybullet shape"""
    options = {}
    collision = isinstance(shape, Collision)
    if collision:
        options['collisionFramePosition'] = shape.pose[:3]
        options['collisionFrameOrientation'] = rot_quat(shape.pose[3:])
    else:
        options['visualFramePosition'] = shape.pose[:3]
        options['visualFrameOrientation'] = rot_quat(shape.pose[3:])
        options['rgbaColor'] = shape.diffuse
        options['specularColor'] = shape.specular
    if isinstance(shape.geometry, Plane):
        options['shapeType'] = pybullet.GEOM_PLANE
        options['planeNormal'] = shape.geometry.normal
    elif isinstance(shape.geometry, Box):
        options['shapeType'] = pybullet.GEOM_BOX
        options['halfExtents'] = 0.5*np.array(shape.geometry.size)
    elif isinstance(shape.geometry, Sphere):
        options['shapeType'] = pybullet.GEOM_SPHERE
        options['radius'] = shape.geometry.radius
    elif isinstance(shape.geometry, Cylinder):
        options['shapeType'] = pybullet.GEOM_CYLINDER
        options['radius'] = shape.geometry.radius
        options['height' if collision else 'length'] = shape.geometry.length
    elif isinstance(shape.geometry, Capsule):
        options['shapeType'] = pybullet.GEOM_CAPSULE
        options['radius'] = shape.geometry.radius
        options['height' if collision else 'length'] = shape.geometry.length
    elif isinstance(shape.geometry, Mesh):
        options['shapeType'] = pybullet.GEOM_MESH
        options['fileName'] = os.path.join(path, shape.geometry.uri)
        options['meshScale'] = shape.geometry.scale
        if force_concave:
            options['flags'] = pybullet.GEOM_FORCE_CONCAVE_TRIMESH
    elif isinstance(shape.geometry, Heightmap):
        options['shapeType'] = pybullet.GEOM_HEIGHTMAP
    else:
        raise Exception('Unknown type {}'.format(type(shape.geometry)))
    return options


def find_joint(sdf_model, link):
    """Find joint"""
    for joint in sdf_model.joints:
        if joint.child == link.name:
            return joint
    return None


def joint_pybullet_type(joint):
    """Find joint"""
    if joint.type == 'revolute':
        return pybullet.JOINT_REVOLUTE
    if joint.type == 'continuous':
        return pybullet.JOINT_REVOLUTE
    if joint.type == 'prismatic':
        return pybullet.JOINT_PRISMATIC
    return pybullet.JOINT_FIXED


def reset_controllers(identity):
    """Reset controllers"""
    n_joints = pybullet.getNumJoints(identity)
    joints = np.arange(n_joints)
    zeros = np.zeros_like(joints)
    pybullet.setJointMotorControlArray(
        identity,
        joints,
        pybullet.POSITION_CONTROL,
        targetPositions=zeros,
        targetVelocities=zeros,
        forces=zeros
    )
    pybullet.setJointMotorControlArray(
        identity,
        joints,
        pybullet.VELOCITY_CONTROL,
        targetVelocities=zeros,
        forces=zeros,
    )
    pybullet.setJointMotorControlArray(
        identity,
        joints,
        pybullet.TORQUE_CONTROL,
        forces=zeros
    )


def rearange_base_link_list(table, base_link_index):
    """Rarange base link to beginning of table"""
    value = table[base_link_index]
    del table[base_link_index]
    table.insert(0, value)
    return table


def rearange_base_link_dict(dictionary, base_link_index):
    """Rarange base link to beginning of table"""
    new_dict = {}
    for key, value in dictionary.items():
        new_dict[key] = (
            0 if value == base_link_index
            else value + 1 if value < base_link_index
            else value
        )
    return new_dict


def load_sdf(
        sdf_path,
        force_concave=False,
        reset_control=False,
        verbose=False,
        links_options=None,
):
    """Load SDF"""
    sdf = ModelSDF.read(sdf_path)[0]
    folder = os.path.dirname(sdf_path)
    links_names = []
    visuals = []
    collisions = []
    joint_types = []
    joints_names = []
    link_index = {}
    joints_axis = []
    link_pos = []
    link_ori = []
    link_masses = []
    link_com = []
    link_i = 0
    base_link_index = None
    parenting = {joint.child: joint.parent for joint in sdf.joints}
    for link in sdf.links:
        
        # Number of visuals/collisions in link
        n_links = max(1, len(link.visuals), len(link.collisions))

        # Joint information
        joint = find_joint(sdf, link)

        if 'Antenna' in link.name:
            force_concave=True
        else:
            force_concave=False

        # Visual and collisions
        for i in range(n_links):
            # Visuals
            visuals.append(
                pybullet.createVisualShape(
                    **pybullet_options_from_shape(
                        link.visuals[i],
                        path=folder,
                        force_concave=force_concave,
                    )
                ) if i < len(link.visuals) else -1
            )
            # Collisions
            collisions.append(
                pybullet.createCollisionShape(
                    **pybullet_options_from_shape(
                        link.collisions[i],
                        path=folder,
                        force_concave=force_concave,
                    )
                ) if i < len(link.collisions) else -1
            )
            if i > 0:
                # Dummy links to support multiple visuals and collisions
                link_name = '{}_dummy_{}'.format(link.name, i-1)
                link_index[link_name] = link_i
                links_names.append(link_name)
                parenting[link_name] = link_index[parenting[link.name]]
                link_pos.append([0, 0, 0])
                link_ori.append([0, 0, 0])
                # parenting[link_name] = link_index[parenting[link.name]]
                # link_pos.append(link.pose[:3])
                # link_ori.append(link.pose[3:])
                link_masses.append(0)
                link_com.append(link.pose[:3])
                # link_com.append([0, 0, 0])
                joint_types.append(pybullet.JOINT_FIXED)
                joints_names.append('{}_dummy_{}'.format(joint.name, i-1))
                joints_axis.append([0.0, 0.0, 1.0])
            else:
                # Link information
                link_index[link.name] = link_i
                links_names.append(link.name)
                link_pos.append(link.pose[:3])
                link_ori.append(link.pose[3:])
                link_masses.append(link.inertial.mass)
                link_com.append(link.inertial.pose[:3])
                # link_com.append([0, 0, 0])
                # Joint information
                if joint is None:
                    # Base link
                    assert base_link_index is None, 'Found two base links'
                    base_link_index = link_i
                else:
                    joint_types.append(joint_pybullet_type(joint))
                    joints_names.append(joint.name)
                    joints_axis.append(
                        joint.axis.xyz
                        if joint.axis is not None
                        else [0.0, 0.0, 1.0]
                    )
            link_i += 1
    n_links = link_i

    # Rearange base link at beginning
    links_names = rearange_base_link_list(links_names, base_link_index)
    link_pos = rearange_base_link_list(link_pos, base_link_index)
    link_ori = rearange_base_link_list(link_ori, base_link_index)
    link_com = rearange_base_link_list(link_com, base_link_index)
    link_masses = rearange_base_link_list(link_masses, base_link_index)
    visuals = rearange_base_link_list(visuals, base_link_index)
    collisions = rearange_base_link_list(collisions, base_link_index)
    link_index = rearange_base_link_dict(link_index, base_link_index)
    link_parent_indices = [
        link_index[parenting[name]]
        for link_i, name in enumerate(links_names[1:])
    ]

    if links_options:
        # Modify masses
        mass_multiplier_map = {
            link.name: link.mass_multiplier
            for link in links_options
        }
        link_masses = [
            mass_multiplier_map[link_name]*link_mass
            if link_name in mass_multiplier_map
            else link_mass
            for link_name, link_mass in zip(links_names, link_masses)
        ]

    # Local information
    link_local_positions = []
    link_local_orientations = []
    for pos, name, ori in zip(link_pos, links_names, link_ori):
        if name in parenting:
            link_local_positions.append(
                pybullet.multiplyTransforms(
                    [0, 0, 0],
                    rot_invert(
                        rot_quat(link_ori[link_index[parenting[name]]])),
                    (
                        np.array(pos)
                        - np.array(link_pos[link_index[parenting[name]]])
                    ),
                    rot_quat([0, 0, 0]),
                )[0]
            )
            link_local_orientations.append(
                rot_mult(
                    rot_invert(
                        rot_quat(link_ori[link_index[parenting[name]]])),
                    rot_quat(ori),
                )
            )

    # Model information
    if verbose:
        pylog.debug('\n'.join(
            [
                '0 (Base link): {} - index: {} - mass: {:.4f} [kg]'.format(
                    name,
                    link_i,
                    link_masses[link_i],
                )
                if link_i == 0
                else (
                    '{: >3} {: <15}'
                    ' - parent: {: <15} ({: >2})'
                    ' - mass: {:.4f} [kg]'
                    ' - joint: {: <15} - axis: {}'
                ).format(
                    '{}:'.format(link_i),
                    name,
                    parenting[name],
                    link_index[parenting[name]],
                    link_masses[link_i],
                    joints_names[link_i-1],
                    joints_axis[link_i-1],
                )
                for link_i, name in enumerate(links_names)
            ] + [
                '\nTotal mass: {:.4f} [kg]'.format(sum(link_masses))
            ]
        ))
        pylog.debug('Spawning model')

    # Spawn model
    model = pybullet.createMultiBody(
        baseMass=link_masses[0],
        basePosition=link_pos[0],
        baseOrientation=rot_quat(link_ori[0]),
        baseVisualShapeIndex=visuals[0],
        baseCollisionShapeIndex=collisions[0],
        baseInertialFramePosition=link_com[0],
        baseInertialFrameOrientation=[0, 0, 0, 1],
        linkMasses=link_masses[1:],
        linkPositions=link_local_positions,
        linkOrientations=link_local_orientations,
        linkInertialFramePositions=link_com[1:],
        linkInertialFrameOrientations=[[0, 0, 0, 1]]*(len(collisions)-1),
        linkVisualShapeIndices=visuals[1:],
        linkCollisionShapeIndices=collisions[1:],
        linkParentIndices=link_parent_indices,
        linkJointTypes=joint_types,
        linkJointAxis=joints_axis,
        flags=(
            pybullet.URDF_USE_SELF_COLLISION
            # | pybullet.URDF_MERGE_FIXED_LINKS
            # | pybullet.URDF_MAINTAIN_LINK_ORDER  # Removes certain links?
            # | pybullet.URDF_ENABLE_SLEEPING
            # | pybullet.URDF_USE_INERTIA_FROM_FILE
            # | pybullet.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
            # | pybullet.URDF_PRINT_URDF_INFO
            # | pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL
        ),
    )
    if verbose:
        pylog.debug('Spawned model (Identity={})'.format(model))
        pylog.debug(
            '\n'.join([
                '{: <15} mass: {:.4f} [kg] - inertia: {} {} {}'.format(
                    link_name+':',
                    *np.array(
                        pybullet.getDynamicsInfo(model, link_i-1)
                    )[[0, 2, 3, 4]],
                )
                for link_i, link_name in enumerate(links_names)
            ])
        )
    if reset_control:
        reset_controllers(model)
    links, joints = {}, {}
    links[links_names[0]] = -1
    for joint_i in range(pybullet.getNumJoints(model)):
        joint_info = pybullet.getJointInfo(model, joint_i)
        joint_name = joint_info[1].decode('UTF-8')
        joint_number = int(joint_name.replace('joint', ''))-1
        joints[joints_names[joint_number]] = joint_i
        link_name = joint_info[12].decode('UTF-8')
        link_number = int(link_name.replace('link', ''))
        links[links_names[link_number]] = joint_i
    return model, links, joints


def load_sdf_pybullet(sdf_path, index=0, morphology_links=None):
    """Original way of loading SDF - Deprecated"""
    links, joints = {}, {}
    identity = pybullet.loadSDF(
        sdf_path,
        useMaximalCoordinates=0,
        globalScaling=1,
    )[index]
    for joint_i in range(pybullet.getNumJoints(identity)):
        joint_info = pybullet.getJointInfo(identity, joint_i)
        links[joint_info[12].decode('UTF-8')] = joint_i
        joints[joint_info[1].decode('UTF-8')] = joint_i
    if morphology_links is not None:
        for link in morphology_links:
            if link not in links:
                links[link] = -1
                break
        for link in morphology_links:
            assert link in links, 'Link {} not in {}'.format(
                link,
                links,
            )
    return identity, links, joints

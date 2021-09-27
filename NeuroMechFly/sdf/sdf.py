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

Farms SDF

"""

import xml.dom.minidom
import xml.etree.ElementTree as ET
import numpy as np
import trimesh as tri
from ..utils.options import Options
from .units import SimulationUnitScaling
from .utils import replace_file_name_in_path


class ModelSDF(Options):
    """Class to import SDF file in a PyBullet compatible way. """
    def __init__(self, name, pose, units, **kwargs):
        super(ModelSDF, self).__init__()
        self.name = name
        self.pose = pose
        self.units = units
        self.links = kwargs.pop("links", [])
        self.joints = kwargs.pop("joints", [])

    def xml(self):
        """xml"""
        sdf = ET.Element("sdf", version="1.6")
        model = ET.SubElement(sdf, "model", name=self.name)
        if self.pose is not None:
            pose = ET.SubElement(model, "pose")
            pose.text = " ".join([
                str(element*(self.units.meters if i < 3 else 1))
                for i, element in enumerate(self.pose)
            ])
        for link in self.links:
            link.xml(model)
        for joint in self.joints:
            joint.xml(model)
        return sdf

    def xml_str(self):
        """xml string"""
        sdf = self.xml()
        xml_str = ET.tostring(
            sdf,
            encoding='utf8',
            method='xml'
        ).decode('utf8')
        # dom = xml.dom.minidom.parse(xml_fname)
        dom = xml.dom.minidom.parseString(xml_str)
        return dom.toprettyxml(indent=2*" ")

    def write(self, filename="animat.sdf"):
        """Write SDF to file"""
        # ET.ElementTree(self.xml()).write(filename)
        with open(filename, "w+") as sdf_file:
            sdf_file.write(self.xml_str())

    @classmethod
    def create_model(cls, data):
        """ Create ModelSDF from parsed sdf model data. """
        pose = (
            [float(p) for p in data.find('pose').text.split(' ')]
            if data.find('pose') is not None
            else np.zeros(6)
        )
        links = []
        joints = []
        for link in data.findall('link'):
            links.append(Link.from_xml(link))
        for joint in data.findall('joint'):
            joints.append(Joint.from_xml(joint))
        return cls(
            name=data.attrib['name'],
            pose=pose,
            units=SimulationUnitScaling(),
            **{
                'links': links,
                'joints': joints
            }
        )

    @classmethod
    def read(cls, filename):
        """ Read from an SDF FILE. """
        tree = ET.parse(filename)
        root = tree.getroot()
        if root.find("world"):
            world = root.find("world")
        else:
            world = root
        return [
            cls.create_model(model) for model in world.findall("model")
        ]

    def change_units(self, units):
        """ Change the units of all elements in the model. """
        self.units = units
        #: Change units of links
        for link in self.links:
            link.units = units
            link.inertial.units = units
            for collision in link.collisions:
                collision.units = units
                collision.geometry.units = units
            for visual in link.visuals:
                visual.units = units
                visual.geometry.units = units
        #: Change units of joints
        for joint in self.joints:
            pass


class Link(Options):
    """Link"""

    def __init__(self, name, pose, units, **kwargs):
        super(Link, self).__init__()
        self.name = name
        self.pose = pose
        self.inertial = kwargs.pop('inertial', None)
        self.collisions = kwargs.pop('collisions', [])
        self.visuals = kwargs.pop('visuals', [])
        self.units = units

    @classmethod
    def empty(cls, name, pose, units, **kwargs):
        """Empty"""
        return cls(
            name,
            pose=pose,
            inertial=kwargs.pop("inertial", Inertial.empty(units)),
            collisions=[],
            visuals=[],
            units=units
        )

    @classmethod
    def plane(cls, name, pose, units, **kwargs):
        """Plane"""
        visual_kwargs = {}
        if "color" in kwargs:
            visual_kwargs["color"] = kwargs.pop("color", None)
        # inertial_pose = kwargs.pop("inertial_pose", np.zeros(6))
        shape_pose = kwargs.pop("shape_pose", np.zeros(6))
        return cls(
            name,
            pose=pose,
            collisions=[Collision.plane(
                name,
                pose=shape_pose,
                units=units,
                **kwargs
            )],
            visuals=[Visual.plane(
                name,
                pose=shape_pose,
                units=units,
                **visual_kwargs,
                **kwargs
            )],
            units=units
        )

    @classmethod
    def box(cls, name, pose, units, **kwargs):
        """Box"""
        visual_kwargs = {}
        if "color" in kwargs:
            visual_kwargs["color"] = kwargs.pop("color", None)
        # inertial_pose = kwargs.pop("inertial_pose", np.zeros(6))
        shape_pose = kwargs.pop("shape_pose", np.zeros(6))
        return cls(
            name,
            pose=pose,
            inertial=Inertial.box(
                pose=shape_pose,
                units=units,
                **kwargs
            ),
            collisions=[Collision.box(
                name,
                pose=shape_pose,
                units=units,
                **kwargs
            )],
            visuals=[Visual.box(
                name,
                pose=shape_pose,
                units=units,
                **visual_kwargs,
                **kwargs
            )],
            units=units
        )

    @classmethod
    def sphere(cls, name, pose, units, **kwargs):
        """Sphere"""
        visual_kwargs = {}
        if "color" in kwargs:
            visual_kwargs["color"] = kwargs.pop("color", None)
        # inertial_pose = kwargs.pop("inertial_pose", np.zeros(6))
        shape_pose = kwargs.pop("shape_pose", np.zeros(6))
        return cls(
            name,
            pose=pose,
            inertial=Inertial.sphere(
                pose=shape_pose,
                units=units,
                **kwargs,
            ),
            collisions=[Collision.sphere(
                name,
                pose=shape_pose,
                units=units,
                **kwargs
            )],
            visuals=[Visual.sphere(
                name,
                pose=shape_pose,
                units=units,
                **visual_kwargs,
                **kwargs,
            )],
            units=units
        )

    @classmethod
    def cylinder(cls, name, pose, units, **kwargs):
        """Cylinder"""
        visual_kwargs = {}
        if "color" in kwargs:
            visual_kwargs["color"] = kwargs.pop("color", None)
        # inertial_pose = kwargs.pop("inertial_pose", np.zeros(6))
        shape_pose = kwargs.pop("shape_pose", np.zeros(6))
        return cls(
            name,
            pose=pose,
            inertial=Inertial.cylinder(
                pose=shape_pose,
                units=units,
                **kwargs
            ),
            collisions=[Collision.cylinder(
                name,
                pose=shape_pose,
                units=units,
                **kwargs
            )],
            visuals=[Visual.cylinder(
                name,
                pose=shape_pose,
                units=units,
                **visual_kwargs,
                **kwargs
            )],
            units=units
        )

    @classmethod
    def capsule(cls, name, pose, units, **kwargs):
        """Capsule"""
        visual_kwargs = {}
        if "color" in kwargs:
            visual_kwargs["color"] = kwargs.pop("color", None)
        # inertial_pose = kwargs.pop("inertial_pose", np.zeros(6))
        shape_pose = kwargs.pop("shape_pose", np.zeros(6))
        return cls(
            name,
            pose=pose,
            inertial=Inertial.capsule(
                pose=shape_pose,
                units=units,
                **kwargs
            ),
            collisions=[Collision.capsule(
                name,
                pose=shape_pose,
                units=units,
                **kwargs
            )],
            visuals=[Visual.capsule(
                name,
                pose=shape_pose,
                units=units,
                **visual_kwargs,
                **kwargs
            )],
            units=units
        )

    @classmethod
    def from_mesh(cls, name, mesh, pose, units, **kwargs):
        """From mesh"""
        inertial_kwargs = {}
        for element in ['mass', 'density']:
            if element in kwargs:
                inertial_kwargs[element] = kwargs.pop(element)
        visual_kwargs = {}
        if "color" in kwargs:
            visual_kwargs["color"] = kwargs.pop("color", None)
        # inertial_pose = kwargs.pop("inertial_pose", np.zeros(6))
        inertial_from_bounding = kwargs.pop("inertial_from_bounding", False)
        shape_pose = kwargs.pop("shape_pose", np.zeros(6))
        scale = kwargs.pop("scale", 1)
        compute_inertial = kwargs.pop('compute_inertial', True)
        assert not kwargs, kwargs
        return cls(
            name,
            pose=pose,
            inertial=(
                Inertial.from_mesh(
                    mesh,
                    pose=shape_pose,
                    scale=scale,
                    units=units,
                    mesh_bounding_box=inertial_from_bounding,
                    **inertial_kwargs,
                ) if compute_inertial else Inertial(
                    pose=np.zeros(6),
                    mass=0,
                    inertias=np.zeros(6),
                    units=units,
                )
            ),
            collisions=[Collision.from_mesh(
                name,
                mesh,
                pose=shape_pose,
                scale=(scale*np.ones(3)).tolist(),
                units=units
            )],
            visuals=[Visual.from_mesh(
                name,
                mesh,
                pose=shape_pose,
                scale=(scale*np.ones(3)).tolist(),
                units=units,
                **visual_kwargs
            )],
            units=units
        )

    @classmethod
    def heightmap(cls, name, uri, pose, size, pos, units, **kwargs):
        """Heightmap"""
        visual_kwargs = {}
        if 'color' in kwargs:
            visual_kwargs['color'] = kwargs.pop('color', None)
        shape_pose = kwargs.pop('shape_pose', np.zeros(6))
        size = kwargs.pop('size', np.ones(3))
        pos = kwargs.pop('pos', np.zeros(3))
        assert not kwargs, kwargs
        return cls(
            name,
            pose=pose,
            inertial=Inertial(
                pose=np.zeros(6),
                mass=0,
                inertias=np.zeros(6),
                units=units,
            ),
            collisions=[Collision.heightmap(
                name,
                uri,
                pose=shape_pose,
                size=size,
                pos=pos,
                units=units
            )],
            visuals=[Visual.heightmap(
                name,
                uri,
                pose=shape_pose,
                size=size,
                pos=pos,
                units=units,
                **visual_kwargs
            )],
            units=units
        )

    def xml(self, model):
        """xml"""
        link = ET.SubElement(model, "link", name=self.name)
        if self.pose is not None:
            pose = ET.SubElement(link, "pose")
            pose.text = " ".join([
                str(element*(self.units.meters if i < 3 else 1))
                for i, element in enumerate(self.pose)
            ])
        if self.inertial is not None:
            self.inertial.xml(link)
        for collision in self.collisions:
            collision.xml(link)
        for visual in self.visuals:
            visual.xml(link)

    @classmethod
    def from_xml(cls, data):
        """ Create link object from parsed xml data. """
        pose = (
            [float(p) for p in data.find('pose').text.split(' ')]
            if data.find('pose') is not None
            else np.zeros(6)
        )
        return cls(
            data.attrib['name'],
            pose=pose,
            inertial=(
                Inertial.from_xml(data.find('inertial'))
                if data.find('inertial') is not None
                else None
            ),
            collisions=(
                [
                    Collision.from_xml(collision)
                    for collision in data.findall('collision')
                ]
                if data.find('collision') is not None
                else []
            ),
            visuals=(
                [
                    Visual.from_xml(visual)
                    for visual in data.findall('visual')
                ]
                if data.find('visual') is not None
                else []
            ),
            units=SimulationUnitScaling()
        )


class Inertial(Options):
    """Inertial"""

    def __init__(self, pose, mass, volume, inertias, units):
        super(Inertial, self).__init__()
        self.mass = mass
        self.volume = volume
        self.inertias = inertias
        self.units = units
        self.pose = pose

    @classmethod
    def empty(cls, units):
        """Empty"""
        return cls(
            pose=[0]*6,
            mass=0,
            volume = 0,
            inertias=[0]*6,
            units=units
        )

    @classmethod
    def box(cls, size, pose, units, **kwargs):
        """Box"""
        density = kwargs.pop("density", 1000)
        volume = size[0]*size[1]*size[2]
        mass = volume*density
        return cls(
            pose=np.asarray(pose),
            mass=mass,
            inertias=[
                1/12*mass*(size[1]**2 + size[2]**2),
                0,
                0,
                1/12*mass*(size[0]**2 + size[2]**2),
                0,
                1/12*mass*(size[0]**2 + size[1]**2)
            ],
            units=units
        )

    @classmethod
    def sphere(cls, radius, pose, units, **kwargs):
        """Sphere"""
        density = kwargs.pop("density", 1000)
        volume = 4/3*np.pi*radius**3
        mass = volume*density
        return cls(
            pose=np.asarray(pose),
            mass=mass,
            inertias=[
                2/5*mass*radius**2,
                0,
                0,
                2/5*mass*radius**2,
                0,
                2/5*mass*radius**2
            ],
            units=units
        )

    @classmethod
    def cylinder(cls, length, radius, pose, units, **kwargs):
        """Cylinder"""
        density = kwargs.pop("density", 1000)
        volume = np.pi*radius**2*length
        mass = volume*density
        return cls(
            pose=np.asarray(pose),
            mass=mass,
            inertias=[
                1/12*mass*(3*radius**2 + length**2),
                0,
                0,
                1/12*mass*(3*radius**2 + length**2),
                0,
                1/2*mass*(radius**2)
            ],
            units=units
        )

    @classmethod
    def capsule(cls, length, radius, pose, units, **kwargs):
        """Capsule"""
        density = kwargs.pop("density", 1000)
        volume_sphere = 4/3*np.pi*radius**3
        volume_cylinder = np.pi*radius**2*length
        volume = volume_sphere + volume_cylinder
        mass = volume*density
        return cls(
            pose=np.asarray(pose),
            mass=mass,
            # TODO: This is Cylinder inertia!!
            inertias=[
                1/12*mass*(3*radius**2 + length**2),
                0,
                0,
                1/12*mass*(3*radius**2 + length**2),
                0,
                1/2*mass*(radius**2)
            ],
            units=units
        )

    @classmethod
    def from_mesh(cls, mesh, scale, pose, units, **kwargs):
        """From mesh"""
        _from_bounding_box = kwargs.pop('mesh_bounding_box', False)
        _mesh = tri.load_mesh(mesh)
        if _from_bounding_box:
            _mesh = _mesh.bounding_box
        _mesh.apply_transform(tri.transformations.scale_matrix(scale))
        # : Set density for the mesh
        if 'mass' in kwargs:
            if isinstance(_mesh, tri.Scene):
                values = list(_mesh.geometry.values())
                if len(values) > 1:
                    _mesh = _mesh.convex_hull
                else:
                    _mesh = values[0]
            mass = kwargs.pop('mass')
            mesh_mass = _mesh.mass
            _mesh.density *= mass/mesh_mass
        else:
            _mesh.density = kwargs.pop('density', 1000)
            if isinstance(_mesh, tri.Scene):
                meshes = _mesh.geometry.values()
                assert len(meshes) == 1, (
                    'Wrong number of meshes in {}'.format(mesh)
                )
                _mesh = meshes[0]
            mass = _mesh.mass
            volume = _mesh.volume
        inertia = _mesh.moment_inertia
        if not Inertial.valid_mass(mass) or not Inertial.valid_inertia(inertia):
            raise ValueError(
                'Mesh {} has inappropriate mass {} or inertia {}'.format(
                    mesh,
                    mass,
                    inertia,
                )
            )
        return cls(
            pose=np.concatenate([
                (_mesh.center_mass+np.asarray(pose[:3])),
                pose[3:]
            ]),
            mass=mass,
            volume=volume,
            inertias=[
                inertia[0, 0],
                inertia[0, 1],
                inertia[0, 2],
                inertia[1, 1],
                inertia[1, 2],
                inertia[2, 2]
            ],
            units=units,
        )

    @staticmethod
    def valid_mass(mass):
        """ Check if mass is positive, bounded and non zero. """
        if mass <= 0.0 or np.isnan(mass):
            return False
        return True

    @staticmethod
    def valid_inertia(inertia):
        """ Check if inertia matrix is positive, bounded and non zero. """
        ixx = inertia[0, 0]
        iyy = inertia[1, 1]
        izz = inertia[2, 2]
        (ixx, iyy, izz) = np.linalg.eigvals(inertia)
        positive_definite = np.all([i > 0.0 for i in (ixx, iyy, izz)])
        ineqaulity = (
            (ixx + iyy > izz) and (ixx + izz > iyy) and (iyy + izz > ixx))
        if not ineqaulity or not positive_definite:
            return False
        return True

    def xml(self, link):
        """xml"""
        inertial = ET.SubElement(link, "inertial")
        if self.pose is not None:
            pose = ET.SubElement(inertial, "pose")
            pose.text = " ".join([
                str(element*(self.units.meters if i < 3 else 1))
                for i, element in enumerate(self.pose)
            ])
        if self.mass is not None:
            mass = ET.SubElement(inertial, "mass")
            mass.text = str(self.mass*self.units.kilograms)
        if self.inertias is not None:
            inertia = ET.SubElement(inertial, "inertia")
            inertias = [
                ET.SubElement(inertia, name)
                for name in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]
            ]
            for i, inertia in enumerate(inertias):
                inertia.text = str(
                    self.inertias[i]*self.units.kilograms*self.units.meters**2
                )

    @classmethod
    def from_xml(cls, data):
        """ Create Inertial object from xml data.

        Parameters
        ----------
        cls : <cls>
            Class

        data : <ET.ElemenTree>
            Inertial data from the sdf

        Returns
        -------
        out : <Inertial>
            Inertial object from xml

        """
        pose = (
            [float(p) for p in data.find('pose').text.split(' ')]
            if data.find('pose') is not None
            else np.zeros(6)
        )
        mass = (
            float(data.find('mass').text)
            if data.find('mass') is not None
            else None
        )
        volume = (
            float(data.find('volume').text)
            if data.find('volume') is not None
            else None
        )
        inertias = (
            [float(i.text) for i in data.find('inertia').getchildren()]
            if data.find('inertia') is not None
            else None
        )
        return cls(
            mass=mass,
            volume=volume,
            inertias=inertias,
            pose=pose,
            units=SimulationUnitScaling()
        )


class Shape(Options):
    """Shape"""

    def __init__(self, name, geometry, suffix, units, **kwargs):
        super(Shape, self).__init__()
        if suffix not in name:
            self.name = "{}_{}".format(name, suffix)
        else:
            self.name = name
        self.geometry = geometry
        self.suffix = suffix
        self.pose = kwargs.pop('pose', np.zeros(6))
        assert self.pose is not None
        self.units = units

    @classmethod
    def plane(cls, name, normal, size, units, **kwargs):
        """Plane"""
        return cls(
            name=name,
            geometry=Plane(normal, size, units),
            units=units,
            **kwargs
        )

    @classmethod
    def box(cls, name, size, units, **kwargs):
        """Box"""
        return cls(
            name=name,
            geometry=Box(size, units),
            units=units,
            **kwargs
        )

    @classmethod
    def sphere(cls, name, radius, units, **kwargs):
        """Box"""
        return cls(
            name=name,
            geometry=Sphere(radius, units),
            units=units,
            **kwargs
        )

    @classmethod
    def cylinder(cls, name, radius, length, units, **kwargs):
        """Cylinder"""
        return cls(
            name=name,
            geometry=Cylinder(radius, length, units),
            units=units,
            **kwargs
        )

    @classmethod
    def capsule(cls, name, radius, length, units, **kwargs):
        """Box"""
        return cls(
            name=name,
            geometry=Capsule(radius, length, units),
            units=units,
            **kwargs
        )

    @classmethod
    def from_mesh(cls, name, mesh, scale, units, **kwargs):
        """From mesh"""
        return cls(
            name=name,
            geometry=Mesh(mesh, scale, units),
            units=units,
            **kwargs
        )

    @classmethod
    def bounding_from_mesh(cls, name, mesh, scale, units, **kwargs):
        """ Create bounding shape from mesh."""
        bounding_shape = (kwargs.get('bounding_shape', 'box')).lower()
        use_primitive = kwargs.get('use_primitive', False)
        #: Read the original mesh
        mesh_obj = tri.load(mesh)
        if bounding_shape == 'box':
            box = mesh_obj.bounding_box
            extents = box.extents
            if use_primitive:
                return cls(
                    name=name,
                    geometry=Box(extents, units),
                    units=units,
                    **kwargs
                )
            else:
                #: Export mesh
                new_mesh_path = replace_file_name_in_path(
                    mesh, name.replace('_'+cls.SUFFIX, '')+'_bounding_box'
                )
                box.export(new_mesh_path)
                return cls(
                    name=name,
                    geometry=Mesh(new_mesh_path, scale, units),
                    units=units,
                    **kwargs
                )
        elif bounding_shape == 'sphere':
            sphere = mesh_obj.bounding_sphere
            radius = sphere.primitive.radius
            if use_primitive:
                return cls(
                    name=name,
                    geometry=Sphere(radius, units),
                    units=units,
                    **kwargs
                )
            else:
                #: Export mesh
                new_mesh_path = replace_file_name_in_path(
                    mesh, name.replace('_'+cls.SUFFIX, '')+'_bounding_sphere'
                )
                box.export(new_mesh_path)
                return cls(
                    name=name,
                    geometry=Mesh(new_mesh_path, scale, units),
                    units=units,
                    **kwargs
                )
        elif bounding_shape == 'cylinder':
            cylinder = mesh_obj.bounding_cylinder
            radius = cylinder.primitive.radius
            length = cylinder.primitive.height
            if use_primitive:
                return cls(
                    name=name,
                    geometry=Cylinder(radius, length, units),
                    units=units,
                    **kwargs
                )
            else:
                #: Export mesh
                new_mesh_path = replace_file_name_in_path(
                    mesh, name.replace('_'+cls.SUFFIX, '')+'_bounding_cylinder'
                )
                box.export(new_mesh_path)
                return cls(
                    name=name,
                    geometry=Mesh(new_mesh_path, scale, units),
                    units=units,
                    **kwargs
                )
        elif bounding_shape == 'convex_hull':
            convex_hull = mesh_obj.convex_hull
            #: Export mesh
            new_mesh_path = replace_file_name_in_path(
                mesh, name.replace('_'+cls.SUFFIX, '')+'_bounding_convex_hull'
            )
            convex_hull.export(new_mesh_path)
            return cls(
                name=name,
                geometry=Mesh(new_mesh_path, scale, units),
                units=units,
                **kwargs
            )
        else:
            return cls(
                name=name,
                geometry=Mesh(mesh, scale, units),
                units=units,
                **kwargs
            )

    @classmethod
    def heightmap(cls, name, uri, size, pos, units, **kwargs):
        """Heightmap"""
        return cls(
            name=name,
            geometry=Heightmap(uri, size, pos, units),
            units=units,
            **kwargs
        )

    def xml(self, link):
        """xml"""
        shape = ET.SubElement(
            link,
            self.suffix,
            name=self.name
        )
        if self.pose is not None:
            pose = ET.SubElement(shape, "pose")
            pose.text = " ".join([
                str(element*(self.units.meters if i < 3 else 1))
                for i, element in enumerate(self.pose)
            ])
        self.geometry.xml(shape)
        return shape


class Collision(Shape):
    """Collision"""

    SUFFIX = "collision"

    def __init__(self, name, **kwargs):
        super(Collision, self).__init__(
            name=name,
            suffix=self.SUFFIX,
            **kwargs
        )

    @classmethod
    def from_xml(cls, data):
        """Generate collision shape model from xml.

        Parameters
        ----------
        cls : <Shape>
            Shape class data
        data : <ET.ElementTree>
            Visual/Collision object data

        Returns
        -------
        out : <Shape>
            Shape model
        """
        geometry_types = {
            'plane': Plane,
            'box': Box,
            'sphere': Sphere,
            'cylinder': Cylinder,
            'capsule': Capsule,
            'mesh': Mesh,
            'heightmap': Heightmap,
        }
        pose = (
            [float(p) for p in data.find('pose').text.split(' ')]
            if data.find('pose') is not None
            else np.zeros(6)
        )
        shape_data = {
            'geometry': geometry_types[
                data.find('geometry')[0].tag
            ].from_xml(data.find('geometry')[0]),
            'pose': pose,
            'units': SimulationUnitScaling()
        }
        #: Remove the suffix
        name = data.attrib['name']
        if '_collision' in name:
            name = name.replace('_collision', '')
        return cls(
            name=name,
            **shape_data
        )


class Visual(Shape):
    """Visual"""

    SUFFIX = "visual"

    def __init__(self, name, **kwargs):
        self.color = kwargs.pop("color", None)
        self.ambient = self.color
        self.diffuse = self.color
        self.specular = self.color
        self.emissive = self.color
        super(Visual, self).__init__(name=name, suffix=self.SUFFIX, **kwargs)

    def xml(self, link):
        """xml"""
        shape = super(Visual, self).xml(link)
        material = ET.SubElement(shape, "material")
        # script = ET.SubElement(material, "script")
        # uri = ET.SubElement(script, "uri")
        # uri.text = "skin.material"
        # name = ET.SubElement(script, "name")
        # name.text = "Skin"
        if self.color is not None:
            # color = ET.SubElement(material, "color")
            # color.text = " ".join([str(element) for element in self.color])
            ambient = ET.SubElement(material, "ambient")
            ambient.text = " ".join(
                [str(element) for element in self.ambient]
            )
            diffuse = ET.SubElement(material, "diffuse")
            diffuse.text = " ".join(
                [str(element) for element in self.diffuse]
            )
            specular = ET.SubElement(material, "specular")
            specular.text = " ".join(
                [str(element) for element in self.specular]
            )
            emissive = ET.SubElement(material, "emissive")
            emissive.text = " ".join(
                [str(element) for element in self.emissive]
            )

    @classmethod
    def from_xml(cls, data):
        """Generate visual shape model from xml.

        Parameters
        ----------
        cls : <Shape>
            Shape class data
        data : <ET.ElementTree>
            Visual/Collision object data

        Returns
        -------
        out : <Shape>
            Shape model
        """
        geometry_types = {
            'plane': Plane,
            'box': Box,
            'sphere': Sphere,
            'cylinder': Cylinder,
            'capsule': Capsule,
            'mesh': Mesh,
            'heightmap': Heightmap,
        }
        material = data.find('material')
        color = (
            [float(c) for c in material.find('diffuse').text.split(' ')]
            if material
            else None
        )
        pose = (
            [float(p) for p in data.find('pose').text.split(' ')]
            if data.find('pose') is not None
            else np.zeros(6)
        )
        shape_data = {
            'geometry': geometry_types[
                data.find('geometry')[0].tag
            ].from_xml(data.find('geometry')[0]),
            'pose': pose,
            'color': color,
            'units': SimulationUnitScaling()
        }
        #: Remove the suffix
        name = data.attrib['name']
        if '_visual' in name:
            name = name.replace('_visual', '')
        return cls(
            name=name,
            **shape_data
        )


class Plane(Options):
    """Plane"""

    def __init__(self, normal, size, units):
        super(Plane, self).__init__()
        self.normal = normal
        self.size = size
        self.units = units

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, "geometry")
        plane = ET.SubElement(geometry, "plane")
        normal = ET.SubElement(plane, "normal")
        normal.text = " ".join([
            str(element)
            for element in self.normal
        ])
        size = ET.SubElement(plane, "size")
        size.text = " ".join([
            str(element*self.units.meters)
            for element in self.size
        ])

    @classmethod
    def from_xml(cls, data):
        """Generate Plane shape from xml data.

        Parameters
        ----------
        cls : <Plane>
            Plane class data
        data : <ET.ElementTree>
            Plane object data

        Returns
        -------
        out : <Plane>
            Plane model
        """
        return cls(
            normal=[float(s) for s in data.find('normal').text.split(' ')],
            size=[float(s) for s in data.find('size').text.split(' ')],
            units=SimulationUnitScaling()
        )


class Box(Options):
    """Box"""

    def __init__(self, size, units):
        super(Box, self).__init__()
        self.size = size
        self.units = units

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, "geometry")
        box = ET.SubElement(geometry, "box")
        size = ET.SubElement(box, "size")
        size.text = " ".join([
            str(element*self.units.meters)
            for element in self.size
        ])

    @classmethod
    def from_xml(cls, data):
        """Generate Box shape from xml data.

        Parameters
        ----------
        cls : <Box>
            Box class data
        data : <ET.ElementTree>
            Box object data

        Returns
        -------
        out : <Box>
            Box model
        """
        return cls(
            size=[float(s) for s in data.find('size').text.split(' ')],
            units=SimulationUnitScaling()
        )


class Sphere(Options):
    """Sphere"""

    def __init__(self, radius, units):
        super(Sphere, self).__init__()
        self.radius = radius
        self.units = units

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, "geometry")
        sphere = ET.SubElement(geometry, "sphere")
        radius = ET.SubElement(sphere, "radius")
        radius.text = str(self.radius*self.units.meters)

    @classmethod
    def from_xml(cls, data):
        """Generate Sphere shape from xml data.

        Parameters
        ----------
        cls : <Sphere>
            Sphere class data
        data : <ET.ElementTree>
            Sphere object data

        Returns
        -------
        out : <Sphere>
            Sphere model
        """
        return cls(
            radius=float(data.find('radius').text),
            units=SimulationUnitScaling()
        )


class Cylinder(Options):
    """Cylinder"""

    def __init__(self, radius, length, units):
        super(Cylinder, self).__init__()
        self.radius = radius
        self.length = length
        self.units = units

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, "geometry")
        cylinder = ET.SubElement(geometry, "cylinder")
        radius = ET.SubElement(cylinder, "radius")
        radius.text = str(self.radius*self.units.meters)
        length = ET.SubElement(cylinder, "length")
        length.text = str(self.length*self.units.meters)

    @classmethod
    def from_xml(cls, data):
        """Generate Cylinder shape from xml data.

        Parameters
        ----------
        cls : <Cylinder>
            Cylinder class data
        data : <ET.ElementTree>
            Cylinder object data

        Returns
        -------
        out : <Cylinder>
            Cylinder model
        """
        return cls(
            radius=float(data.find('radius').text),
            length=float(data.find('length').text),
            units=SimulationUnitScaling()
        )


class Capsule(Options):
    """Capsule"""

    def __init__(self, radius, length, units):
        super(Capsule, self).__init__()
        self.radius = radius
        self.length = length
        self.units = units

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, "geometry")
        capsule = ET.SubElement(geometry, "capsule")
        length = ET.SubElement(capsule, "length")
        length.text = str(self.length*self.units.meters)
        radius = ET.SubElement(capsule, "radius")
        radius.text = str(self.radius*self.units.meters)

    @classmethod
    def from_xml(cls, data):
        """Generate Capsule shape from xml data.

        Parameters
        ----------
        cls : <Capsule>
            Capsule class data
        data : <ET.ElementTree>
            Capsule object data

        Returns
        -------
        out : <Capsule>
            Capsule model
        """
        return cls(
            length=float(data.find('length').text),
            radius=float(data.find('radius').text),
            units=SimulationUnitScaling()
        )


class Mesh(Options):
    """Mesh"""

    def __init__(self, uri, scale, units):
        super(Mesh, self).__init__()
        self.uri = uri
        self.scale = scale
        self.units = units

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, "geometry")
        mesh = ET.SubElement(geometry, "mesh")
        uri = ET.SubElement(mesh, "uri")
        uri.text = self.uri
        if self.scale is not None:
            scale = ET.SubElement(mesh, "scale")
            scale.text = " ".join(
                [str(s*self.units.meters) for s in self.scale]
            )

    @classmethod
    def from_xml(cls, data):
        """Generate Mesh shape from xml data.

        Parameters
        ----------
        cls : <Mesh>
            Mesh class data
        data : <ET.ElementTree>
            Mesh object data

        Returns
        -------
        out : <Mesh>
            Mesh model
        """
        scale = (
            [float(s) for s in data.find('scale').text.split(' ')]
            if data.find('scale') is not None
            else [1.0, 1.0, 1.0]
        )
        return cls(
            uri=data.find('uri').text,
            scale=scale,
            units=SimulationUnitScaling()
        )


class Heightmap(Options):
    """Heightmap"""

    def __init__(self, uri, size, pos, units):
        super(Heightmap, self).__init__()
        self.uri = uri
        self.size = size
        self.pos = pos
        self.units = units

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, "geometry")
        heightmap = ET.SubElement(geometry, "heightmap")
        uri = ET.SubElement(heightmap, "uri")
        uri.text = self.uri
        if self.size is not None:
            size = ET.SubElement(heightmap, "size")
            size.text = " ".join([
                str(element*self.units.meters)
                for element in self.size
            ])
        if self.pos is not None:
            pos = ET.SubElement(heightmap, "pos")
            pos.text = " ".join([
                str(element*self.units.meters)
                for element in self.pos
            ])

    @classmethod
    def from_xml(cls, data):
        """Generate Heightmap shape from xml data.

        Parameters
        ----------
        cls : <Heightmap>
            Heightmap class data
        data : <ET.ElementTree>
            Heightmap object data

        Returns
        -------
        out : <Heightmap>
            Heightmap model
        """
        size = (
            [float(s) for s in data.find('size').text.split(' ')]
            if data.find('size') is not None
            else 1.0
        )
        pos = (
            [float(s) for s in data.find('pos').text.split(' ')]
            if data.find('pos') is not None
            else 1.0
        )
        return cls(
            uri=data.find('uri').text,
            size=size,
            pos=pos,
            units=SimulationUnitScaling()
        )


class Joint(Options):
    """Joint"""

    def __init__(self, name, joint_type, parent, child, **kwargs):
        super(Joint, self).__init__()
        self.name = name
        self.type = joint_type
        self.parent = parent.name
        self.child = child.name
        self.pose = kwargs.pop("pose", np.zeros(6))
        if kwargs.get('xyz', None) is not None:
            self.axis = Axis(**kwargs)
        else:
            self.axis = None

    def xml(self, model):
        """xml"""
        joint = ET.SubElement(model, "joint", name=self.name, type=self.type)
        parent = ET.SubElement(joint, "parent")
        parent.text = self.parent
        child = ET.SubElement(joint, "child")
        child.text = self.child
        if self.pose is not None:
            pose = ET.SubElement(joint, "pose")
            pose.text = " ".join([str(element) for element in self.pose])
        if self.axis is not None:
            self.axis.xml(joint)

    @classmethod
    def from_xml(cls, data):
        """
        Generate joint object from xml data

        Parameters
        ----------
        cls : <cls>
            Class

        data : <ET.ElemenTree>
            Joint data from the sdf

        Returns
        -------
        out : <Joint>
            Joint object from xml
        """
        pose = (
            [float(p) for p in data.find('pose').text.split(' ')]
            if data.find('pose') is not None
            else np.zeros(6)
        )
        axis_data = (
            Axis.from_xml(data.find('axis'))
            if data.find('axis') is not None
            else None
        )
        if axis_data is None:
            return cls(
                name=data.attrib['name'],
                joint_type=data.attrib['type'],
                parent=Link.empty(data.find('parent').text, [], None),
                child=Link.empty(data.find('child').text, [], None),
                **{'pose': pose}
            )
        return cls(
            name=data.attrib['name'],
            joint_type=data.attrib['type'],
            parent=Link.empty(data.find('parent').text, [], None),
            child=Link.empty(data.find('child').text, [], None),
            **{
                'pose': pose,
                **axis_data
            }
        )


class Axis(Options):
    """Axis"""

    def __init__(self, **kwargs):
        super(Axis, self).__init__()
        self.initial_position = kwargs.pop("initial_position", None)
        self.xyz = kwargs.pop("xyz", [0, 0, 0])
        self.limits = kwargs.pop("limits", None)
        self.dynamics = kwargs.pop("dynamics", None)

    def xml(self, joint):
        """xml"""
        axis = ET.SubElement(joint, "axis")
        if self.initial_position:
            initial_position = ET.SubElement(axis, "initial_position")
            initial_position.text = str(self.initial_position)
        xyz = ET.SubElement(axis, "xyz")
        xyz.text = " ".join([str(element) for element in self.xyz])
        if self.limits is not None:
            limit = ET.SubElement(axis, "limit")
            lower = ET.SubElement(limit, "lower")
            lower.text = str(self.limits[0])
            upper = ET.SubElement(limit, "upper")
            upper.text = str(self.limits[1])
            effort = ET.SubElement(limit, "effort")
            effort.text = str(self.limits[2])
            velocity = ET.SubElement(limit, "velocity")
            velocity.text = str(self.limits[3])
        if self.dynamics is not None:
            dynamics = ET.SubElement(axis, "dynamics")
            damping = ET.SubElement(dynamics, "damping")
            damping.text = str(self.dynamics[0])
            friction = ET.SubElement(dynamics, "friction")
            friction.text = str(self.dynamics[1])
            spring_reference = ET.SubElement(dynamics, "spring_reference")
            spring_reference.text = str(self.dynamics[2])
            spring_stiffness = ET.SubElement(dynamics, "spring_stiffness")
            spring_stiffness.text = str(self.dynamics[3])

    @classmethod
    def from_xml(cls, data):
        """
        Generate axis object from xml data

        Parameters
        ----------
        cls : <cls>
            Class

        data : <ET.ElemenTree>
            Axis data from the sdf

        Returns
        -------
        out : <Axis>
            Axis object from xml
        """
        xyz = [
            float(p) for p in data.find('xyz').text.split(' ')
        ]
        initial_position = (
            float(data.find('initial_position').text)
            if data.find('initial_position') is not None
            else None
        )
        limits = None
        if data.find('limit') is not None:
            limits = [-1]*4
            limits[0] = float(data.find('limit/lower').text)
            limits[1] = float(data.find('limit/upper').text)
            limits[2] = (
                float(data.find('limit/effort').text)
                if data.find('limit/effort') is not None
                else -1
            )
            limits[3] = (
                float(data.find('limit/velocity').text)
                if data.find('limit/velocity') is not None
                else -1
            )
        dynamics = None
        if data.find('dynamics') is not None:
            dynamics = [0]*4
            dynamics[0] = (
                float(data.find('damping').text)
                if data.find('damping') is not None
                else 0.0
            ),
            dynamics[1] = (
                float(data.find('friction').text)
                if data.find('friction') is not None
                else 0.0
            ),
            dynamics[2] = float(data.find('spring_reference').text)
            dynamics[3] = float(data.find('spring_stiffness').text)
        axis_data = {
            'initial_position': initial_position,
            'xyz': xyz,
            'limits': limits,
            'dynamics': dynamics
        }
        return axis_data

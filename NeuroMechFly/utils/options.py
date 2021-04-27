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

Options

"""

from enum import IntEnum
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def pyobject2yaml(filename, pyobject, mode='w+'):
    """Pyobject to yaml"""
    with open(filename, mode) as yaml_file:
        yaml.dump(
            pyobject,
            yaml_file,
            default_flow_style=False,
            sort_keys=False,
            Dumper=Dumper,
        )


def yaml2pyobject(filename):
    """Pyobject to yaml"""
    with open(filename, 'r') as yaml_file:
        options = yaml.load(yaml_file, Loader=Loader)
    return options


class Options(dict):
    """Options"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __getstate__(self):
        """Get state"""
        return self

    def __setstate__(self, value):
        """Get state"""
        for item in value:
            self[item] = value[item]

    def to_dict(self):
        """To dictionary"""
        return {
            key: (
                value.to_dict() if isinstance(value, Options)
                else int(value) if isinstance(value, IntEnum)
                else [
                    val.to_dict() if isinstance(val, Options) else val
                    for val in value
                ] if isinstance(value, list)
                else value
            )
            for key, value in self.items()
        }

    @classmethod
    def load(cls, filename):
        """Load from file"""
        return cls(**yaml2pyobject(filename))

    def save(self, filename):
        """Save to file"""
        pyobject2yaml(filename, self.to_dict())

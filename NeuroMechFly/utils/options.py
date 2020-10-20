"""Options"""

from enum import IntEnum
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
    import farms_pylog as pylog
    pylog.warning(
        'YAML CLoader and CDumper not available'
        ', switching to Python implementation'
        '\nThis will run slower than the C alternative'
    )



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

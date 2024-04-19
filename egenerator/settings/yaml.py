"""The YAML loader and dumper for the egenerator package.

This module defines the YAML loader and dumper for the egenerator package. It
also registers all classes that can be loaded from YAML files.
"""

from ruamel.yaml import YAML

from egenerator.misc import load_class


REGISTERTED_CLASSES = []

# define yaml dumper
yaml_dumper = YAML(typ="safe", pure=True)

# define yaml loader and register all classes
yaml_loader = YAML(typ="safe", pure=True)

for class_name in REGISTERTED_CLASSES:
    yaml_loader.register_class(load_class(class_name))
    yaml_dumper.register_class(load_class(class_name))

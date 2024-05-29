__author__ = "Mirco Huennefeld"
__author_email__ = "mirco.huennefeld@tu-dortmund.de"
__description__ = "IceCube Event-Generator"
__url__ = "https://github.com/icecube/event-generator"

__version_major__ = 1
__version_minor__ = 1
__version_patch__ = 0
__version_info__ = "-dev"

__version__ = "{}.{}.{}{}".format(
    __version_major__,
    __version_minor__,
    __version_patch__,
    __version_info__,
)

# A dictionary of changes that are not backwards compatible
# with previous versions. The keys are the versions that
# contain the breaking changes and the values contain
# information on the changes that were made.
# Mandatory keys are:
#    "type": "global" or "local"
#        "global" means that the change affects all components
#        "local" means that the change affects only specific components
#                and the key "affected_components" must be present.
#
# Example:
# __version_compatibility__ = {
#     "1.0.0": {
#         "Description": "Description of the breaking change",
#         "type": "global",
#     },
#     "1.0.1": {
#         "Description": "Description of the breaking change",
#         "type": "local",
#         "affected_components": ["class_string1", "class_string2"],
#     },
# }
__version_compatibility__ = {}

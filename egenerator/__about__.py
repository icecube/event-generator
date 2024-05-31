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
__version_compatibility__ = {
    "1.1.0": {
        "Description": (
            "Bugfix: Fixed a bug in the re-normalization for time "
            "window exclusions. The bug re-normalized the individual "
            "mixture model components instead of the whole mixture. "
            "This bug thus lead to changes in the shape of the pulse "
            "arrival time PDF when exclusions were used. This is now "
            "fixed to instead re-normalize the whole mixture. "
            "Older models will have compensated for this effect if "
            "already trained as a mixture of multiple sources. Thus, "
            "introducing this bugfix will lead to incompatibilities "
            "with older models."
        ),
        "type": "global",
    },
}

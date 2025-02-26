__author__ = "Mirco Huennefeld"
__author_email__ = "mirco.huennefeld@tu-dortmund.de"
__description__ = "IceCube Event-Generator"
__url__ = "https://github.com/icecube/event-generator"

__version_major__ = 2
__version_minor__ = 0
__version_patch__ = 0
__version_info__ = ""

__version__ = "{}.{}.{}{}".format(
    __version_major__,
    __version_minor__,
    __version_patch__,
    __version_info__,
)

# A dictionary of changes that are not backwards compatible
# with previous versions. The keys are the versions that
# contain the breaking changes and the values contain
# a list of dictinoaries with information on
# each of the changes that were made.
# Mandatory keys are:
#    "type": "global" or "local"
#        "global" means that the change affects all components
#        "local" means that the change affects only specific components
#                and the key "affected_components" must be present.
#
# Example:
# __version_compatibility__ = {
#     "1.0.0": [{
#         "Description": "Description of the breaking change",
#         "type": "global",
#     }],
#     "1.0.1": [{
#         "Description": "Description of the breaking change",
#         "type": "local",
#         "affected_components": ["class_string1", "class_string2"],
#     }],
# }
__version_compatibility__ = {
    "2.0.0": [
        {
            "Description": (
                "Major code restructuring and refactoring. "
                "Introduction of LatentToPDF mixture models for charge "
                "and time distributions. Overall code cleanup. "
                "This is not backwards compatible with previous versions."
            ),
            "type": "global",
        },
        {
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
        {
            "Description": (
                "The shift_cascade_vertex option was removed from the "
                "label modules. It is now part of the cascade models. "
                "In addition, the float_precision must now be set and "
                "the previous parameter estimate_charge_distribution "
                "was renamed to charge_distribution_type. These changes "
                "are not backwards compatible with previous versions."
            ),
            "type": "global",
        },
        {
            "Description": (
                "The cascade default model was modified to change the "
                "charge scaling. The asymmetric Gaussian components "
                "are now offset by the time it takes for the first photon"
                "to reach the DOM from the cascade vertex. This change is "
                "not backwards compatible with previous versions. "
                "In addition, a number of naming changes were made to "
                "various modules and components in event-generator. "
            ),
            "type": "global",
        },
        {
            "Description": (
                "The label modules were restructured to reduce code "
                "duplication. This change is not backwards compatible "
                "with previous versions."
            ),
            "type": "global",
        },
        {
            "Description": (
                "The x_pulses_ids now also includes the pulse number at "
                "the given DOM. This allows for easy selection of the "
                "first pulses at a DOM, for instance. "
                "A result of this change is that the tensor changed "
                "its shape from [n_pulses, 3] to [n_pulses, 4]. This "
                "breaks backwards compatibility and may also require "
                "updates in dependen user code."
            ),
            "type": "global",
        },
    ],
}

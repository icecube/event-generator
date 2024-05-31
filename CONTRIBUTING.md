# Contributing

Welcome and thanks for considering to contribute to this repository!

## Pre-commit Hooks

When contributing to this project, please utilize the pre-commit hooks. If not installed yet, you will need to add the python package `pre-commit`:

    pip install pre-commit

Once the package is installed, simply install the pre-commit hooks defined in the repository by executing:

    pre-commit install

from within the repository directory.

The pre-commit hooks will now automatically run when invoking `git commit`. Note, however, that this requires an active shell that has `pre-commit` installed.
You can also manually run the pre-commit on single files or on all files via:

    pre-commit run --all-files

If you need to commit something even though there are errors (this should not have to be done!), then you can add the flag `--no-verify` to the `git commit` command. This will bypass the pre-commit hooks.

Additional information is provided here: https://pre-commit.com/


## Non-backward compatible changes

New contributions to this repository should aim to maintain backwards compatibility, such that
models trained with earlier version of the software may still be run in later software versions.
However, this is not always possible. In such cases where breaking changes are required, these
should be documented in the `__version_compatibility__` dictionary in the `egenerator.__about__`
file. When loading saved components from disk, this dictionary is utilized to verify compatibility
of a previously saved model and the current software version. Breaking changes may either be of
type `global` if they affect all components and trained models of the software, or of type
`local` in case only certain components are affected. In the latter case, a list of
`affected_components` must be provided in the corresponding dictionary entry.
This list contains a list of class strings of each affected component in the event-generator
software.

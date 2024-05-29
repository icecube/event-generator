| Testing | Coverage | Documentation | DOI |
| :-----: | :------: | :-----------: | :-----: |
| [![Unit Tests](https://github.com/icecube/event-generator/actions/workflows/test_suite.yml/badge.svg)](https://github.com/icecube/event-generator/actions/workflows/test_suite.yml) | [![Coverage Status](https://codecov.io/gh/icecube/event-generator/branch/master/graph/badge.svg)](https://codecov.io/gh/icecube/event-generator/branch/master) | [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://user-web.icecube.wisc.edu/~mhuennefeld/docs/event_generator/html/) | [![DOI](https://zenodo.org/badge/244745589.svg)](https://zenodo.org/badge/latestdoi/244745589) |

# event-generator
event-generator is a package designed for the IceCube neutrino telescope. It can be used to generate and reconstruct arbitrary event
hypotheses.

## Documentation

The documentation for the package can be found here: <https://user-web.icecube.wisc.edu/~mhuennefeld/docs/event_generator/html/>

More example code and examples will be added as time permits.

## Package structure

`deps` contains C/C++ file dependencies of the packages which are compiled when
the package is installed.

`docs` contains Sphinx documentation which is automatically build from in-code
comments and deployed

`egenerator` contains the python source code of the package.

`test` contains unit tests which can be run locally

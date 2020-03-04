| Testing | Coverage | Documentation |
| :-----: | :------: | :-----------: |
| [![Build Status](https://travis-ci.org/mhuen/event-generator.svg?branch=master)](https://travis-ci.org/mhuen/event-generator) | [![Coverage Status](https://codecov.io/gh/mhuen/event-generator/branch/master/graph/badge.svg)](https://codecov.io/gh/mhuen/event-generator/branch/master) | [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://mhuen.github.io/event-generator) |

# event-generator
event-generator is a package designed for the IceCube neutrino telescope. It can be used to generate and reconstruct arbitrary event
hypotheses.

## Documentation

The documentation for the package can be found here: <https://mhuen.github.io/event-generator/>

More example code and examples will be added as time permits.

## Package structure

`deps` contains C/C++ file dependencies of the packages which are compiled when
the package is installed.

`docs` contains Sphinx documentation which is automatically build from in-code 
comments and deployed

`egenerator` contains the python source code of the package.

`test` contains unit tests which can be run locally


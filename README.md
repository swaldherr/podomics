# README #

podomics is a Python package to do proper orthogonal decomposition (POD) on large-scale timeseries datasets such as from timeseries omics experiments.

### Installation instructions ###

Currently there is no installation method.
Put the `podomics` directory into the working directory of your scripts / notebooks.

### Documentation ###

Documentation is currently accessible at:

https://swaldherr.github.io/podomics/podomics

### Developer guidelines ###

Running tests:

	nosetests -v --with-coverage --cover-package=podomics --with-doctest podomics

Compiling documentation:

	pdoc3 --html -o docs --force podomics

### Author ###

Steffen Waldherr <steffen.waldherr@univie.ac.at>

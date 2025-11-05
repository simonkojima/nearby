# nearby

## Installation

- Requirements
    - Python >= 3.11
    - numpy
    - scipy
    - pandas
    - mne
    - tslearn
    - autograd
    - pymanopt[autograd]

You can install all the dependencies with the following pip command.

```
pip install numpy scipy pandas mne tslearn autograd pymanopt[autograd]
```

### Installation from source

```
git clone https://github.com/simonkojima/nearby.git
cd ./nearby
pip install .
```

## Usage

Check out the documentation [here](https://simonkojima.github.io/nearby-docs/).

## docs generation

### Requirements for docs generation

```
pip install git+https://github.com/NeuroTechX/moabb.git#egg=moabb
pip install filelock sphinx pydata-sphinx-theme sphinx-multiversion sphinx-gallery numpydoc tslearn autograd pymanopt[autograd]
```

1. Run following in docs

```

sphinx-apidoc -f -o ../docs/source ../nearby && make html

```

2. Run following in nearby root

```

sphinx-build -b html docs/source ~/git/nearby-docs/latest

```
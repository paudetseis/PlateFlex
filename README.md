# PlateFlex: Software for mapping the effective elastic thickness of the lithosphere

<!-- ![](./plateflex/examples/picture/tws_logo.png)
 -->
The flexure of elastic plates is a central concept in the theory of plate tectonics,
where the Earth's lithosphere (crust and uppermost mantle) reacts to applied loads 
by bending, a process referred to as flexural isostasy. The plate elasticity is 
parameterized by the *flexural rigidity*, which is proportional to the product of 
Young's modulus with the cube of the elastic plate thickness. Estimating the *effective* 
elastic thickness (<i>T<sub>e</sub></i>) of the lithosphere (thickness 
of an equivalent ideal elastic plate) gives important clues on the rheology of the 
lithosphere and its thermal state. 

Estimating <i>T<sub>e</sub></i> can be done by modeling the cross-spectral properties 
(admittance and coherence) between topography and gravity anomaly data, 
which are proxies for the distribution of flexurally compensated surface and subsurface 
loads. These spectral properties can be calculated using different spectral
estimation techniques - however, to map <i>T<sub>e</sub></i> variations it is 
important to use analysis windows small enough for good spatial resolution, but 
large enough to capture the effect of flexure at long wavelengths. The wavelet 
transform is particularly well suited for this analysis because it avoids splitting
the grids into small windows and can therefore produce cross-spectral functions
at each point of the input grid.

This package contains `python` and `fortran` modules to calculate the wavelet spectral
and cross-spectral quantities of 2D gridded data of topography and gravity anomalies.
Once obtained, the wavelet cross-spectral quantities (admittance and coherence) are
used to determine the parameters of the effectively elastic plate, such as the 
effective elastic thickness (<i>T<sub>e</sub></i>), the initial subsurface-to-surface
load ratio (<i>F</i>) and optionally the initial phase difference between
surface and subsurface loads (<i>alpha</i>). The software uses the analytical
functions with *uniform F and alpha* to fit the admittance and/or coherence functions 
using a probabilistic inference method. 

The analysis can be done using either the Bouguer or Free air gravity anomalies, and
over land or ocean areas. Common computational 
workflows are covered in the Jupyter notebooks bundled with this package.

.. note::
    
    The cross-spectral quantities calculated here are the real-valued admittance and real-squared-coherency, as discussed in the [references](#references)

## Installation

### Dependencies

The current version was developed using **Python3.7**
Also, the following packages are required:

- [`gfortran`](https://gcc.gnu.org/wiki/GFortran) (or any Fortran compiler)
- [`numpy`](https://numpy.org)
- [`pymc3`](https://docs.pymc.io)
- [`matplotlib`](https://matplotlib.org)
- [`seaborn`](https://seaborn.pydata.org)

### Installing using pip

You can install `plateflex` using the [pip package manager](https://pypi.org/project/pip/):

```bash
pip install plateflex
```
All the dependencies will be automatically installed by `pip`.

### Installing with conda

You can install `plateflex` using the [conda package manager](https://conda.io).
Its required dependencies can be easily installed with:

```bash
conda install numpy pymc3 matplotlib -c conda-forge
```

Then `plateflex` can be installed with `pip`:

```bash
pip install plateflex
```

#### Conda environment

We recommend creating a custom 
[conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)
where `plateflex` can be installed along with its dependencies. 

- Create a environment called `pflex` and install all dependencies:

```bash
conda create -n pflex python=3.7 numpy pymc3 matplotlib seaborn -c conda-forge
```

- or create it from the `pflex_env.yml` file by first checking out the repository:

```bash
git checkout https://github.com/paudetseis/PlateFlex.git
cd PlateFlex
conda env create -f pflex_env.yml
```

Activate the newly created environment:

```bash
conda activate pflex
```

Install `plateflex` with `pip`:

```bash
pip install plateflex
```

### Installing from source

Download or clone the repository:
```bash
git clone https://github.com/paudetseis/PlateFlex.git
cd PlateFlex
```

Next we recommend following the steps for creating a `conda` environment (see [above](#conda-environment)). Then install using `pip`:

```bash
pip install .
``` 

---
**NOTE**

If you are actively working on the code, or making frequent edits, it is advisable to perform 
installation from source with the `-e` flag: 

```bash
pip install -e .
```

This enables an editable installation, where symbolic links are used rather than straight 
copies. This means that any changes made in the local folders will be reflected in the 
package available on the system.

---

## Usage 

### Jupyter Notebooks

Included in this package is a set of Jupyter Notebooks, which give examples on how to call the various routines 
<!-- and obtain plane wave seismograms and receiver functions. The Notebooks describe how to reproduce published examples of synthetic data from [Audet (2016)](#references) and [Porter et al. (2011)](#references).

- [sim_obs_Audet2016.ipynb](./plateflex/examples/Notebooks/sim_obs_Audet2016.ipynb): Example plane wave seismograms and P receiver functions for OBS data from [Audet (2016)](#Audet).
- [sim_Prfs_Porter2011.ipynb](./plateflex/examples/Notebooks/sim_Prfs_Porter2011.ipynb): Example P receiver functions from [Porter et al. (2011)](#Porter)
- [sim_SKS.ipynb](./plateflex/examples/Notebooks/sim_SKS.ipynb): Example plane wave seismograms for SKS splitting studies.
 -->
After [installing `plateflex`](#installation), these notebooks can be locally installed (i.e., in a local folder `Notebooks`) from the package by running:

```python
from plateflex import doc
doc.install_doc(path='Notebooks')
```

To run the notebooks you will have to further install `jupyter`:

```bash
conda install jupyter
```

Then ```cd Notebooks``` and type:

```bash
jupyter notebook
```

You can then save the notebooks as `python` scripts, check out the model files and you should be good to go!

### Testing

A series of tests are located in the ``tests`` subdirectory. In order to perform these tests, clone the repository and run `pytest` (`conda install pytest` if needed):

```bash
git checkout https://github.com/paudetseis/PlateFlex.git
cd PlateFlex
pytest -v
```

### API Documentation

The API for all functions in `plateflex` can be accessed from https://paudetseis.github.io/PlateFlex/.

## References

- Audet, P. (2014). Toward mapping the effective elastic thickness of planetary lithospheres
from a spherical wavelet analysis of gravity and topography. Physics of the Earth and Planetary Interiors, 226, 48-82. https://doi.org/10.1016/j.pepi.2013.09.011

- Kirby, J.F. (2014). Estimation of the effective elastic thickness of the lithosphere using inverse spectral methods: The state of the art. Tectonophysics, 631, 87-116. https://doi.org/10.1016/j.tecto.2014.04.021


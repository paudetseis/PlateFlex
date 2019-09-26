# PlateFlex: Software for mapping the effective elastic thickness of the lithosphere

![](./plateflex/examples/picture/plateflex_logo.png)

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
functions with *uniform F and alpha* to fit the admittance and/or coherence functions. 
The estimation can be done using non-linear least-squares or probabilistic (i.e., bayesian)
inference methods. 

The analysis can be done using either the Bouguer or Free air gravity anomalies, and
over land or ocean areas. Computational workflows are covered in the Jupyter 
notebooks bundled with this package. The software contains methods to make beautiful and
insightful plots using the `seaborn` package.
    
> **_NOTE:_**  The cross-spectral quantities calculated here are the real-valued admittance and real-squared-coherency, as discussed in the [references](#references)

## Installation

### Dependencies

The current version was developed using **Python3.7**
Also, the following packages are required:

- [`gfortran`](https://gcc.gnu.org/wiki/GFortran) (or any Fortran compiler)
- [`numpy`](https://numpy.org)
- [`pymc3`](https://docs.pymc.io)
- [`seaborn`](https://seaborn.pydata.org)
- [`skimage`](https://scikit-image.org)

<!-- ### Installing using pip

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
 -->
#### Conda environment

We recommend creating a custom 
[conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)
where `plateflex` can be installed along with its dependencies. 

<!-- - Create a environment called `pflex` and install all dependencies:
 -->
```bash
conda create -n flex python=3.7 numpy pymc3 matplotlib seaborn scikit-image -c conda-forge
```

<!-- - or create it from the `flex_env.yml` file by first checking out the repository:

```bash
git checkout https://github.com/paudetseis/PlateFlex.git
cd PlateFlex
conda env create -f pflex_env.yml
```
 -->
Activate the newly created environment:

```bash
conda activate flex
```

<!-- Install `plateflex` with `pip`:

```bash
pip install plateflex
```
 -->
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

<!-- ---
**_NOTE_**

If you are actively working on the code, or making frequent edits, it is advisable to perform 
installation from source with the `-e` flag: 

```bash
pip install -e .
```

This enables an editable installation, where symbolic links are used rather than straight 
copies. This means that any changes made in the local folders will be reflected in the 
package available on the system.

---
 -->
## Usage 

### Documentation

The documentation for all classes and functions in `plateflex` can be accessed from https://paudetseis.github.io/PlateFlex/.

### Jupyter Notebooks

Included in this package is a set of Jupyter Notebooks, which give examples on how to create `Grid` objects and estimate
the flexural parameters over whole grids. The Notebooks describe how to produce pulication quality results that closely
match those published in [Audet (2014)](#references) and [Kirby and Swain. (2009)](#references) for North America, as well
as those of [Kalnins and Watts (2009)](#references) for the NW Pacific.

- [Ex1_making_grids.ipynb](./plateflex/examples/Notebooks/Ex1_making_grids.ipynb): Exploring the basic class of `PlateFlex`.
- [Ex2_wavelet_analysis.ipynb](./plateflex/examples/Notebooks/Ex2_wavelet_analysis.ipynb): Performing a wavelet analysis of gridded data
- [Ex3_admittance_coherence.ipynb](./plateflex/examples/Notebooks/Ex3_admittance_coherence.ipynb): Calculating the wavelet admittance and coherence of two `Grid` objects
- [Ex4_estimate_flex_parameters_cell.ipynb](./plateflex/examples/Notebooks/Ex4_estimate_flex_parameters_cell.ipynb): Estimating the flexural parameters at a single grid cell location.
- [Ex5_estimate_flex_parameters_grid.ipynb](./plateflex/examples/Notebooks/Ex5_estimate_flex_parameters_grid.ipynb): Estimating the flexural parameters over the whole grid.
- [Ex6_full_suite_North_America.ipynb](./plateflex/examples/Notebooks/Ex6_full_suite_North_America.ipynb): Carrying out the full suite at once for North America with new `Grid` objects to improve modeling.
- [Ex7_full_suite_NW_Pacific.ipynb](./plateflex/examples/Notebooks/Ex7_full_suite_NW_Pacific.ipynb): Same as Example 6 but for the NW Pacific.

After [installing `plateflex`](#installation), these notebooks can be locally installed (i.e., in a local folder `Examples`) from the package by running:

```python
from plateflex import doc
doc.install_doc(path='Examples')
```

To run the notebooks you will have to further install `jupyter`:

```bash
conda install jupyter
```

Then:

```bash
unzip data.zip
cd Examples
jupyter notebook
```

You can then save the notebooks as `python` scripts and you should be good to go!

### How to make new gridded data sets

Although the examples above work as advertised, making new grids for your own project can be a daunting task. In the [wiki](https://github.com/paudetseis/PlateFlex/wiki/How-to-make-gridded-data-sets-to-use-with-PlateFlex) page we provide examples of how to reproduce the data sets used in the Jupyter notebooks from publicly available topography and gravity models. 


<!-- ### Testing

A series of tests are located in the ``tests`` subdirectory. In order to perform these tests, clone the repository and run `pytest` (`conda install pytest` if needed):

```bash
git checkout https://github.com/paudetseis/PlateFlex.git
cd PlateFlex
pytest -v
```
 -->
## References

- Audet, P. (2014). Toward mapping the effective elastic thickness of planetary lithospheres
from a spherical wavelet analysis of gravity and topography. Physics of the Earth and Planetary Interiors, 226, 48-82. https://doi.org/10.1016/j.pepi.2013.09.011

- Kalnins, L.M., and Watts, A.B. (2009). Spatial variations in effective elastic thickness in the Western Pacific Ocean and their implications for Mesozoic volcanism. Earth and Planetary Science Letters, 286, 89-100. https://doi.org/10.1016/j.epsl.2009.06.018

- Kirby, J.F., and Swain, C.J. (2009). A reassessment of spectral Te estimation in continental interiors: The case of North America. Journal of Geophysical Research, 114, B08401. https://doi.org/10.1029/2009JB006356

- Kirby, J.F. (2014). Estimation of the effective elastic thickness of the lithosphere using inverse spectral methods: The state of the art. Tectonophysics, 631, 87-116. https://doi.org/10.1016/j.tecto.2014.04.021


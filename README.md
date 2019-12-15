# PlateFlex: Software for mapping the effective elastic thickness of the lithosphere

![](./plateflex/examples/picture/logo_plateflex.png)

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
    
> **_NOTE:_**  The cross-spectral quantities calculated here are the real-valued admittance and squared-real coherency, as discussed in the [references](#references)


[![Build Status](https://travis-ci.com/paudetseis/PlateFlex.svg?branch=master)](https://travis-ci.com/paudetseis/PlateFlex)

## Usage 

### Documentation

Installation, Usage and API documentation are described at https://paudetseis.github.io/OBStools/

### How to make new gridded data sets

Although the examples above work as advertised, making new grids for your own project can be a daunting task. In the [wiki](https://github.com/paudetseis/PlateFlex/wiki/How-to-make-gridded-data-sets-to-use-with-PlateFlex) page we provide examples of how to reproduce the data sets used in the Jupyter notebooks from publicly available topography and gravity models. 


## References

- Audet, P. (2014). Toward mapping the effective elastic thickness of planetary lithospheres
from a spherical wavelet analysis of gravity and topography. Physics of the Earth and Planetary Interiors, 226, 48-82. https://doi.org/10.1016/j.pepi.2013.09.011

- Kalnins, L.M., and Watts, A.B. (2009). Spatial variations in effective elastic thickness in the Western Pacific Ocean and their implications for Mesozoic volcanism. Earth and Planetary Science Letters, 286, 89-100. https://doi.org/10.1016/j.epsl.2009.06.018

- Kirby, J.F., and Swain, C.J. (2009). A reassessment of spectral Te estimation in continental interiors: The case of North America. Journal of Geophysical Research, 114, B08401. https://doi.org/10.1029/2009JB006356

- Kirby, J.F. (2014). Estimation of the effective elastic thickness of the lithosphere using inverse spectral methods: The state of the art. Tectonophysics, 631, 87-116. https://doi.org/10.1016/j.tecto.2014.04.021


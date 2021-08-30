
.. figure:: ../plateflex/examples/picture/logo_plateflex.png
   :align: center

Documentation
=============

The flexure of elastic plates is a central concept in the theory of plate tectonics,
where the Earth's lithosphere (crust and uppermost mantle) reacts to transverse applied loads 
by bending, a process referred to as flexural isostasy. Estimating the *effective* 
elastic thickness (:math:`T_e`) of the lithosphere (thickness 
of an equivalent ideal elastic plate) gives important clues on the rheology of the 
lithosphere and its thermal state. Estimating :math:`T_e` is typically done by 
modeling the cross-spectral properties (admittance and coherence) between 
topography and gravity anomaly data, which are proxies for the distribution of 
flexurally compensated surface and subsurface loads. 

This package contains ``python`` and ``fortran`` modules to calculate the wavelet spectral
and cross-spectral quantities of 2D gridded data of topography and gravity anomalies.
Once obtained, the wavelet cross-spectral quantities (admittance and coherence) are
used to determine the parameters of the effectively elastic plate, such as the 
effective elastic thickness (:math:`T_e`), the initial subsurface-to-surface
load ratio (:math:`F`) and optionally the initial phase difference between
surface and subsurface loads (:math:`\alpha`).  
The estimation can be done using non-linear least-squares or probabilistic (i.e., bayesian)
inference methods. The analysis can be done using either the Bouguer or Free air gravity anomalies, and
over land or ocean areas. Computational workflows are covered in the Jupyter 
notebooks bundled with this package. The software contains methods to make 
publication-quality plots using the `seaborn` package.
    
.. note:: 

   The cross-spectral quantities calculated here are the real-valued 
   admittance and squared-real coherency, as discussed in the references

.. image:: https://zenodo.org/badge/206867590.svg
   :target: https://zenodo.org/badge/latestdoi/206867590
.. image:: https://github.com/paudetseis/PlateFlex/workflows/Build/badge.svg
    :target: https://github.com/paudetseis/PlateFlex/actions


.. toctree::
   :maxdepth: 1
   :caption: Quick Links

   links

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started

.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   conf
   classes
   estimate
   plotting

.. toctree::
   :maxdepth: 1
   :caption: Jupyter Notebooks

   Example 1: Making grids <https://github.com/paudetseis/PlateFlex/blob/master/plateflex/examples/Notebooks/Ex1_making_grids.ipynb>
   Example 2: Wavelet analysis <https://github.com/paudetseis/PlateFlex/blob/master/plateflex/examples/Notebooks/Ex2_wavelet_analysis.ipynb>
   Example 3: Admittance and coherence <https://github.com/paudetseis/PlateFlex/blob/master/plateflex/examples/Notebooks/Ex3_admittance_coherence.ipynb>
   Example 4: Flexural parameters at single grid cell <https://github.com/paudetseis/PlateFlex/blob/master/plateflex/examples/Notebooks/Ex4_estimate_flex_parameters_cell.ipynb>
   Example 5: Flexural parameters mapped over grid <https://github.com/paudetseis/PlateFlex/blob/master/plateflex/examples/Notebooks/Ex5_estimate_flex_parameters_grid.ipynb>
   Example 6: Full suite for North America <https://github.com/paudetseis/PlateFlex/blob/master/plateflex/examples/Notebooks/Ex6_full_suite_North_America.ipynb>
   Example 7: Full suite for NW Pacific <https://github.com/paudetseis/PlateFlex/blob/master/plateflex/examples/Notebooks/Ex7_full_suite_NW_Pacific.ipynb>

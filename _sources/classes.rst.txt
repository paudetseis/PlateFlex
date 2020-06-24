
Grid Classes
++++++++++++

:mod:`~plateflex` defines the following ``Grid`` classes:

- :class:`~plateflex.classes.Grid`
- :class:`~plateflex.classes.TopoGrid`
- :class:`~plateflex.classes.GravGrid`
- :class:`~plateflex.classes.BougGrid`
- :class:`~plateflex.classes.FairGrid`
- :class:`~plateflex.classes.RhocGrid`
- :class:`~plateflex.classes.ZcGrid`

These classes can be initialized with a grid of the corresponding data
type, and contain methods for the following functionality:

- Extracting contours at some level of the grid
- Performing a wavelet transform using a Morlet wavelet
- Obtaining the wavelet scalogram from the wavelet transform
- Plotting the input grids, wavelet transform components, and scalograms

Grid
----

.. autoclass:: plateflex.classes.Grid
   :members: 

TopoGrid
--------

.. autoclass:: plateflex.classes.TopoGrid
   :members: 

GravGrid
--------

.. autoclass:: plateflex.classes.GravGrid
   :members: 

BougGrid
--------

.. autoclass:: plateflex.classes.BougGrid
   :members: 

FairGrid
--------

.. autoclass:: plateflex.classes.FairGrid
   :members: 

RhocGrid
--------

.. autoclass:: plateflex.classes.RhocGrid
   :members: 

ZcGrid
--------

.. autoclass:: plateflex.classes.ZcGrid
   :members: 

Project Class
+++++++++++++

This module further contains the class :class:`~plateflex.classes.Project`, 
which itself is a container of :class:`~plateflex.classes.Grid` objects 
(at least one each of :class:`~plateflex.classes.TopoGrid` and 
:class:`~plateflex.classes.GravGrid`). Methods are available to:

- Add :class:`~plateflex.classes.Grid` or :class:`~plateflex.classes.Project` objects to the project
- Iterate over :class:`~plateflex.classes.Grid` objects
- Initialize the project
- Perform the wavelet admittance and coherence between topography (:class:`~plateflex.classes.TopoGrid` object) and gravity anomalies (:class:`~plateflex.classes.GravGrid` object)
- Plot the wavelet admnittance and coherence spectra
- Estimate model parameters at single grid cell
- Estimate model parameters at every (or decimated) grid cell
- Plot the statistics of the estimated parameters at single grid cell
- Plot the observed and predicted admittance and coherence functions at single grid cell
- Plot the final grids of model parameters
- Save the results to disk as .csv files

Project
--------

.. autoclass:: plateflex.classes.Project
   :members: 


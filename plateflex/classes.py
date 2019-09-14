# Copyright 2019 Pascal Audet
#
# This file is part of PlateFlex.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This PlateFlex module contains the following ``FlexGrid`` classes, with inheritance in parentheses:

- ``plateflex.classes.FlexGrid``
- ``plateflex.classes.TopoGrid(FlexGrid)``
- ``plateflex.classes.GravGriv(FlexGrid)``
- ``plateflex.classes.BougGrid(GravGrid)``
- ``plateflex.classes.FairGrid(GravGrid)``

These classes can be initiatlized with a grid of topography/bathymetry or gravity
anomaly (Bouguer/Free-air) data. These classes contain methods for the following functionality:

- Performing a wavelet transform using a Morlet wavelet
- Obtaining the wavelet scalogram from the wavelet transform
- Plotting the grids, wavelet transform components, and scalograms

This module further contains the class ``plateflex.classes.Project``, which itself is a container
of ``FlexGrid`` objects (two at most and one each of ``TopoGrid`` and ``GravGrid``). Methods are available to:

- Add ``FlexGrid`` objects to the project
- Iterate over ``FlexGrid`` objects
- Perform the wavelet admittance and coherence between topography (``TopoGrid`` object) and gravity anomalies (``GravGrid`` object)
- Plot the wavelet admnittance and coherence spectra

"""

# -*- coding: utf-8 -*-
import numpy as np
import pymc3 as pm
from plateflex.cpwt import cpwt
from plateflex import conf as cf
from plateflex import plotting
from plateflex import estimate
import seaborn as sns
sns.set()


class FlexGrid(object):
    """Basic grid class of ``plateflex`` with useful methods for wavelet analysis

    Accepts a 2D array and Cartesian coordinates specifying the
    bounding box of the array. Contains methods to calculate the wavelet transform, 
    wavelet scalogram and to plot those quantities at a specified wavenumber index.

    Grid must be projected in km.

    Attributes:
        data (np.ndarray): 2D array of topography/gravity data
        xmin (float): minimum x bound in km
        xmax (float): maximum x bound in km
        ymin (float): minimum y bound in km
        ymax (float): maximum y bound in km
        dx (float): grid spacing in the x-direction in km
        dy (float): grid spacing in the y-direction in km
        nx (int): number of nodes in the x-direction
        ny (int): number of nodes in the y-direction
        xcoords (np.ndarray): 1D array of coordinates in the x-direction
        ycoords (np.ndarray): 1D array of coordinates in the y-direction
        units (str): units of data set
        ns (int): number of wavenumber samples
        k (np.ndarray): 1D array of wavenumbers

    Notes:
        In all instances `x` indicates eastings in metres and `y` indicates northings.
        Using a grid of longitude / latitudinal coordinates (degrees) will result
        in incorrect calculations.

    Examples:
        >>> import numpy as np
        >>> from plateflex import FlexGrid
        >>> # Create zero-valued square grid
        >>> nn = 100; dd = 10.
        >>> x = y = np.linspace(0., nn*dd, nn)
        >>> xmin, xmax = x.min(), x.max()
        >>> ymin, ymax = y.min(), y.max()
        >>> grid = np.zeros((nn, nn))
        >>> flexgrid = FlexGrid(grid, xmin, xmax, ymin, ymax)
        >>> flexgrid
        <plateflex.grids.FlexGrid object at 0x10613fe10>

    """

    def __init__(self, grid, xmin, xmax, ymin, ymax):
        """
        Args:
            grid (np.ndarray): 2D array of topography/gravity data
            xmin (float): minimum x bound in km
            xmax (float): maximum x bound in km
            ymin (float): minimum y bound in km
            ymax (float): maximum y bound in km
        """

        if np.any(np.isnan(np.array(grid))):
            raise(Exception('grid contains NaN values: abort'))
        else:
            self.data = np.array(grid)
            
        ny, nx = self.data.shape
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.xcoords, dx = np.linspace(xmin, xmax, nx, retstep=True)
        self.ycoords, dy = np.linspace(ymin, ymax, ny, retstep=True)
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.units = None
        self.ns, self.k = _lam2k(nx,ny,dx,dy)


    def wlet_transform(self):
        """Calculates the wavelet transform of grid.

        This method uses the module ``plateflex.cpwt.cpwt`` to calculate the wavelet transform
        of the grid. By default the method pads the array with zeros to the next power of 2.
        The wavelet transform is stored as an attribute of the object.

        Attributes:
            wl_trans (np.ndarray): Wavelet transform of the grid (shape (`nx,ny,na,ns`))
        
        Examples:
            >>> import numpy as np
            >>> from plateflex import FlexGrid
            >>> # Create zero-valued square grid
            >>> nn = 100; dd = 1.
            >>> x = y = np.linspace(0., nn*dd, nn)
            >>> xmin, xmax = x.min(), x.max()
            >>> ymin, ymax = y.min(), y.max()
            >>> grid = np.zeros((nn, nn))
            >>> flexgrid = FlexGrid(grid, xmin, xmax, ymin, ymax)
            >>> flexgrid.wlet_transform()
             #loops = 13:  1  2  3  4  5  6  7  8  9 10 11 12 13
            >>> flexgrid.wl_trans.shape
            (100, 100, 11, 13)

        """

        nnx, nny = _npow2(self.nx), _npow2(self.ny)

        wl_trans = cpwt.wlet_transform(self.data, nnx, nny, self.dx, self.dy, self.k)
        self.wl_trans = wl_trans

        return

    def wlet_scalogram(self):
        """Calculates the wavelet scalogram of grid.

        This method uses the module ``plateflex.cpwt.cpwt`` to calculate the wavelet scalogram
        of the grid. If the attribute ``wl_trans`` cannot be found, the method automatically
        calculates the wavelet transform first. The wavelet scalogram is stored as an 
        attribute of the object.

        Attributes:
            wl_sg (np.ndarray): Wavelet scalogram of the grid (shape (`nx,ny,ns`))

        Examples:
            >>> import numpy as np
            >>> from plateflex import FlexGrid
            >>> # Create random-valued square grid
            >>> nn = 200; dd = 10.
            >>> x = y = np.linspace(0., nn*dd, nn)
            >>> xmin, xmax = x.min(), x.max()
            >>> ymin, ymax = y.min(), y.max()
            >>> # set random seed
            >>> np.random.seed(0)
            >>> grid = 100.*np.random.randn(nn, nn)
            >>> flexgrid = FlexGrid(grid, xmin, xmax, ymin, ymax)
            >>> flexgrid.wlet_scalogram()
             #loops = 13:  1  2  3  4  5  6  7  8  9 10 11 12 13
            >>> flexgrid.wl_sg.shape
            (100, 100, 13)

            >>> # Perform wavelet transform first
            >>> flexgrid_2 = FlexGrid(grid, xmin, xmax, ymin, ymax)
            >>> flexgrid_2.wlet_transform()
              #loops = 13:  1  2  3  4  5  6  7  8  9 10 11 12 13
            >>> flexgrid_2.wlet_scalogram()
            >>> np.allclose(flexgrid.wl_sg, flexgrid_2.wl_sg)
            False
            >>> # FIGURE THIS OUT!!! - ok, define cf_f.k0 at _init__

       """

        nnx, nny = 2**(self.nx-1).bit_length(), 2**(self.ny-1).bit_length()

        try:
            wl_sg, ewl_sg = cpwt.wlet_scalogram(self.wl_trans)
        except:
            self.wlet_transform()
            wl_sg, ewl_sg = cpwt.wlet_scalogram(self.wl_trans)

        self.wl_sg = wl_sg
        self.ewl_sg = ewl_sg

        return

    def plot_transform(self, kindex=None, aindex=None, log=False, mask=None, title='Wavelet transform', save=None, clabel=None):
        """Plot the real and imaginary components of the wavelet transform

        This method plots the real and imaginary components of the wavelet transform of a
        ``FlexGrid`` object at wavenumber and angle indices (int). Raises ``Exception`` for the 
        cases where:

        - no wavenumber OR angle index is specified (kindex and aindex)
        - wavenumber index is lower than 0 or larger than self.ns
        - angle index is lower than 0 or larger than 11 (hardcoded)

        If no ``FlexGrid.wl_trans`` attribute is found, the method automatically calculates
        the wavelet transform first.

        """

        if kindex is None or aindex is None:
            raise(Exception('Specify index of wavenumber and angle to plot the transform'))

        if kindex>self.ns or kindex<0:
            raise(Exception('Invalid index: should be between 0 and '+str(ns)))

        if aindex>10 or aindex<0:
            raise(Exception('Invalid index: should be between 0 and 10'))

        try:
            rdata = np.real(self.wl_trans[:,:,aindex,kindex])
            idata = np.imag(self.wl_trans[:,:,aindex,kindex])
        except:
            print('Calculating the transform first')
            self.wlet_transform()
            rdata = np.real(self.wl_trans[:,:,aindex,kindex])
            idata = np.imag(self.wl_trans[:,:,aindex,kindex])

        plotting.plot_real_grid(rdata, title=title, mask=mask, save=save, clabel=self.units)
        plotting.plot_real_grid(idata, title=title, mask=mask, save=save, clabel=self.units)


    def plot_scalogram(self, kindex=None, log=True, mask=None, title='Wavelet scalogram', save=None, clabel=None):
        """Plot the wavelet scalogram

        This method plots the wavelet scalogram of a
        ``FlexGrid`` object at a wavenumber index (int). Raises ``Exception`` for the 
        cases where:

        - no wavenumber index is specified (kindex)
        - wavenumber index is lower than 0 or larger than self.ns

        If no ``FlexGrid.wl_sg`` attribute is found, the method automatically calculates
        the wavelet scalogram (and maybe also the wavelet transform) first.
        
        """

        if kindex is None:
            raise(Exception('Specify index of wavenumber for plotting'))

        if kindex>self.ns or kindex<0:
            raise(Exception('Invalid index: should be between 0 and '+str(self.ns)))

        try:
            data = self.wl_sg[:,:,kindex]
        except:
            print('Calculating the scalogram first')
            self.wlet_scalogram()
            data = self.wl_sg[:,:,kindex]

        if isinstance(self, TopoGrid):
            if log:
                units = r'log(m$^2$)'
            else:
                units = r'm$^2$'
        elif isinstance(self, GravGrid):
            if log:
                units = r'log(mGal$^2$)'
            else:
                units = r'mGal$^2$'

        plotting.plot_real_grid(data, log=log, mask=mask, title=title, save=save, clabel=units)


class GravGrid(FlexGrid):
    """Basic grid class of ``plateflex`` for gravity data that inherits from FlexGrid

    """

    def __init__(self, grid, xmin, xmax, ymin, ymax):

        FlexGrid.__init__(self, grid, xmin, xmax, ymin, ymax)

class BougGrid(GravGrid):
    """Basic grid class of ``plateflex`` for Bouguer gravity data that inherits from GravGrid

    Contains method to plot the grid data with default title and units using module 
    ``plateflex.plotting.plot_real_grid``.
    """

    def __init__(self, grid, xmin, xmax, ymin, ymax):

        GravGrid.__init__(self, grid, xmin, xmax, ymin, ymax)

        self.units = 'mGal'

    def plot(self, mask=None, title='Bouguer anomaly', save=None, clabel=None):

        plotting.plot_real_grid(self.data, title=title, mask=mask, save=save, clabel=self.units)

class FairGrid(GravGrid):
    """Basic grid class of ``plateflex`` for Free-air gravity data that inherits from GravGrid

    Contains method to plot the grid data with default title and units using module 
    ``plateflex.plotting.plot_real_grid``.
    """

    def __init__(self, grid, xmin, xmax, ymin, ymax):

        GravGrid.__init__(self, grid, xmin, xmax, ymin, ymax)

        self.units = 'mGal'

    def plot(self, mask=None, title='Free air anomaly', save=None, clabel=None):

        plotting.plot_real_grid(self.data, title=title, mask=mask, save=save, clabel=self.units)


class TopoGrid(FlexGrid):
    """Basic grid class of ``plateflex`` for Topography data that inherits from FlexGrid

    Contains method to plot the grid data with default title and units using module 
    ``plateflex.plotting.plot_real_grid``.
    """

    def __init__(self, grid, xmin, xmax, ymin, ymax):

        FlexGrid.__init__(self, grid, xmin, xmax, ymin, ymax)

        if np.std(self.data) < 20.:
            self.data *= 1.e3

        self.units = 'meters'

    def plot(self, mask=None, title='Topography', save=None, clabel=None):

        plotting.plot_real_grid(self.data, title=title, mask=mask, save=save, clabel=self.units)

class Project(object):
    """List like object of multiple ``FlexGrid`` objects.

    Contains methods to calculate the wavelet admittance and coherence and 
    estimate flexural model parameters and plot various results. 

    Attributes:
        grids (List): List of ``FlexGrid`` objects

    Note:
        Can hold a list of any length with any type of ``FlexGrid`` objects - 
        however the wavelet calculations will not execute unless the project holds exactly 2 
        ``FlexGrid`` objects in the list, one each of ``GravGrid`` and ``TopoGrid``. 

    Examples:
        >>> import numpy as np
        >>> from plateflex import FlexGrid
        >>> # Create zero-valued square grid
        >>> nn = 100; dd = 10.
        >>> x = y = np.linspace(0., nn*dd, nn)
        >>> xmin, xmax = x.min(), x.max()
        >>> ymin, ymax = y.min(), y.max()
        >>> grid = np.zeros((nn, nn))
        >>> flexgrid = FlexGrid(grid, xmin, xmax, ymin, ymax)
        >>> flexgrid
        <plateflex.grids.FlexGrid object at 0x10613fe10>

    """

    def __init__(self, grids=None):

        self.grids = []

        if isinstance(grids, FlexGrid):
            grids = [grids]
        if grids:
            self.grids.extend(grids)

    def __add__(self, other):
        """
        Add two ``FlexGrid`` objects or a ``Project`` object with a single grid.

        """
        if isinstance(other, FlexGrid):
            other = Project([other])
        if not isinstance(other, Project):
            raise TypeError
        grids = self.grids + other.grids
        return self.__class__(grids=grids)

    def __iter__(self):
        """
        Return a robust iterator for ``FlexGrid`` objects

        """
        return list(self.grids).__iter__()

    def extend(self, grid_list):

        if isinstance(grid_list, list):
            for _i in grid_list:
                # Make sure each item in the list is a FlexGrid object.
                if not isinstance(_i, FlexGrid):
                    msg = 'Extend only accepts a list of FlexGrid objects.'
                    raise TypeError(msg)
            self.grids.extend(grid_list)
        elif isinstance(grid_list, Project):
            self.grids.extend(grid_list.grids)
        else:
            msg = 'Extend only supports a list of FlexGrid objects as argument.'
            raise TypeError(msg)
        return self

    def wlet_admit_coh(self):
        """Calculates the wavelet admittance and coherence of two ``FlexGrid`` objects.

        This method uses the module ``plateflex.cpwt.cpwt`` to calculate the wavelet admittance and
        coherence. The object needs to contain exactly two ``FlexGrid`` objects, one of each
        of ``TopoGrid`` and ``GravGrid`` objects. If the wavelet transforms attributes
        don't exist, they are calculated first.

        Stores the wavelet admittance, coherence and their error as attributes

        Attributes:
            wl_admit (np.ndarray): Wavelet admittance (shape (`nx,ny,ns`))
            wl_eadmit (np.ndarray): Error of wavelet admittance (shape (`nx,ny,ns`))
            wl_coh (np.ndarray): Wavelet coherence (shape (`nx,ny,ns`))
            wl_ecoh (np.ndarray): Error of wavelet coherence (shape (`nx,ny,ns`))

        """

        # Method will not execute unless there are exactly two FlexGrid objects in list
        if len(self.grids)!=2:
            raise(Exception('There needs to be exactly two FlexGrid objects in Project'))

        # Method will fail if there is no ``TopoGrid`` object in list
        if not any(isinstance(g, TopoGrid) for g in self.grids):
            raise(Exception('There needs to be one TopoGrid object in Project'))

        # Method will fail if there is no ``GravGrid`` object in list
        if not any(isinstance(g, GravGrid) for g in self.grids):
            raise(Exception('There needs to be one GravGrid object in Project'))

        self.k = self.grids[0].k
        self.ns = self.grids[0].ns
        self.nx = self.grids[0].nx
        self.ny = self.grids[0].ny

        # Identify the ``FlexGrid`` types for proper calculation of admittance and coherence 
        for grid in self.grids:
            if isinstance(grid, TopoGrid):
                try:
                    wl_trans_topo = grid.wl_trans
                except:
                    grid.wlet_transform()
                    wl_trans_topo = grid.wl_trans
            elif isinstance(grid, GravGrid):
                try:
                    wl_trans_grav = grid.wl_trans
                except:
                    grid.wlet_transform()
                    wl_trans_grav = grid.wl_trans

        # Calculate wavelet admittance and coherence by calling function wlet_admit_coh
        # from module ``plateflex.cpwt.cpwt``
        wl_admit, ewl_admit, wl_coh, ewl_coh = \
            cpwt.wlet_admit_coh(wl_trans_topo, wl_trans_grav)

        # Store the admittance, coherence and error grids as attributes
        self.wl_admit = wl_admit
        self.ewl_admit = ewl_admit
        self.wl_coh = wl_coh
        self.ewl_coh = ewl_coh

        return 

    def plot_admit_coh(self, kindex=None, log=False, mask=None, title=None, save=None, clabel=None):

        if kindex is None:
            raise(Exception('Specify index of wavenumber for plotting'))

        if kindex>self.ns or kindex<0:
            raise(Exception('Invalid index: should be between 0 and '+str(self.ns)))

        try:
            adm = self.wl_admit[:,:,kindex]
            coh = self.wl_coh[:,:,kindex]
        except:
            print('Calculating the admittance and coherence first')
            self.wlet_admit_coh()
            adm = self.wl_admit[:,:,kindex]
            coh = self.wl_coh[:,:,kindex]

        plotting.plot_real_grid(adm, log=log, mask=mask, title=title, save=save, clabel='mGal/m')
        plotting.plot_real_grid(coh, log=log, mask=mask, title=title, save=save, clabel=None)

    def estimate_cell(self, cell=(0,0), alph=False, atype='joint'):
        """Estimate model parameters at single cell location

        Method to estimate the parameters of the flexural model at a single cell location
        of the input grids. 

        Args:
            cell (tuple): Indices of cell location within grid
            alph (bool): Whether or not to estimate parameter ``alpha``
            atype (str): Whether to use the admittance ('admit'), coherence ('coh') or both ('joint')

        Results are stored as attributes of project.

        """

        # Delete attributes to release some memory
        try:
            del self.trace
            del self.summary
            del self.map_estimate   
            del self.cell        
        except:
            print("first attempt at estimating cell parameters")

        if not isinstance(alph, bool):
            raise(Exception("'alph' should be a boolean: defaults to False"))
        self.alph = alph

        if atype not in ['admit', 'coh', 'joint']:
            raise(Exception("'atype' should be one among: 'admit', 'coh', or 'joint'"))
        self.atype = atype
        self.cell = cell

        # Extract admittance and coherence at cell indices
        adm = self.wl_admit[cell[0], cell[1], :]
        eadm = self.ewl_admit[cell[0], cell[1], :]
        coh = self.wl_coh[cell[0], cell[1], :]
        ecoh = self.ewl_coh[cell[0], cell[1], :]

        # Use model returned from function ``estimate.set_model``
        with estimate.set_model(self.k, adm, eadm, coh, ecoh, alph, atype):

            # Sample the Posterior distribution
            trace = pm.sample(cf.samples, tune=cf.tunes, cores=cf.cores)

            # Get Max a porteriori estimate
            map_estimate = pm.find_MAP(method='powell')

            # Get Summary
            summary = pm.summary(trace).round(2)

            # Store the pymc results as object attributes for later plotting
            self.cell_trace = trace
            self.cell_map_estimate = map_estimate
            self.cell_summary = summary

    def estimate_grid(self, nn=10, alph=False, atype='joint'):
        """Estimate model parameters at all (possibly decimated) locations on the grid

        Method to estimate the parameters of the flexural model at all grid point locations.
        It is also possible to decimate the number of grid cells at which to estimate parameters. 

        Args:
            nn (int): Decimator. If grid shape is ``(nx, ny)``, resulting grids will have shape of ``(int(nx/nn), int(ny/nn))``. 
            alph (bool): Whether or not to estimate parameter ``alpha``
            atype (str): Whether to use the admittance ('admit'), coherence ('coh') or both ('joint')

        Final grids of estimated parameters are stored as attributes of project.

        """

        # Import garbage collector
        import gc

        # Delete attributes to release some memory
        try:
            del self.MAP_Te_grid
            del self.std_Te_grid            
            del self.mean_F_grid
            del self.MAP_F_grid
            del self.std_F_grid
            try:        
                del self.mean_a_grid
                del self.MAP_a_grid
                del self.std_a_grid
            except:
                print("parameter 'alpha' was not previously estimated")
        except:
            print("first attempt at estimating grid parameters")

        if not isinstance(alph, bool):
            raise(Exception("'alph' should be a boolean: defaults to False"))
        self.alph = alph

        if atype not in ['admit', 'coh', 'joint']:
            raise(Exception("'atype' should be one among: 'admit', 'coh', or 'joint'"))
        self.atype = atype

        # Initialize result grids to zoroes
        mean_Te_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))
        MAP_Te_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))
        std_Te_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))
        mean_F_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))
        MAP_F_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))
        std_F_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))
        if self.alph:
            mean_a_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))
            MAP_a_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))
            std_a_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))

        # Extract topo grid to avoid computing ocean areas
        if isinstance(self.grids[0], TopoGrid):
            grid = self.grids[0]
        else:
            grid = self.grids[1]

        for i in range(0, self.nx-nn, nn):
            for j in range(0, self.ny-nn, nn):
                
                # For reference - index values
                print(i,j)

                # tuple of cell indices
                cell = (i,j)

                # Skip cell for which topo is < -200 meters
                if grid.data[i,j] < -200.:
                    continue

                # Extract admittance and coherence at cell indices
                adm = self.wl_admit[cell[0], cell[1], :]
                eadm = self.ewl_admit[cell[0], cell[1], :]
                coh = self.wl_coh[cell[0], cell[1], :]
                ecoh = self.ewl_coh[cell[0], cell[1], :]

                # Re-use model returned from function ``estimate.set_model``
                with estimate.set_model(self.k, adm, eadm, coh, ecoh, alph, atype):

                    # Sample the posterior distribution
                    trace = pm.sample(cf.samples, tune=cf.tunes, cores=cf.cores)

                    # Get Max a porteriori estimate
                    map_estimate = pm.find_MAP(method='powell')

                    # Get Summary
                    summary = pm.summary(trace).round(2)

                # Extract estimates from summary and map_estimate
                res = estimate.get_estimates(summary, map_estimate)

                # Distribute the parameters back to space
                mean_Te = res[0]
                std_Te = res[1]
                MAP_Te = res[4]
                mean_F = res[5]
                std_F = res[6]
                MAP_F = res[9]
                if self.alph:
                    mean_a = res[10]
                    std_a = res[11]
                    MAP_a = res[14]

                # Store values in smaller arrays
                mean_Te_grid[int(i/nn),int(j/nn)] = mean_Te
                MAP_Te_grid[int(i/nn),int(j/nn)] = MAP_Te
                std_Te_grid[int(i/nn),int(j/nn)] = std_Te
                mean_F_grid[int(i/nn),int(j/nn)] = mean_F
                MAP_F_grid[int(i/nn),int(j/nn)] = MAP_F
                std_F_grid[int(i/nn),int(j/nn)] = std_F
                if self.alph:
                    mean_a_grid[int(i/nn),int(j/nn)] = mean_a
                    MAP_a_grid[int(i/nn),int(j/nn)] = MAP_a
                    std_a_grid[int(i/nn),int(j/nn)] = std_a

                # Release garbage collector
                gc.collect()

        # Store grids as attributes
        self.mean_Te_grid = mean_Te_grid
        self.MAP_Te_grid = MAP_Te_grid
        self.std_Te_grid = std_Te_grid
        self.mean_F_grid = mean_F_grid
        self.MAP_F_grid = MAP_F_grid
        self.std_F_grid = std_F_grid
        if self.alph:
            self.mean_a_grid = mean_a_grid
            self.MAP_a_grid = MAP_a_grid
            self.std_a_grid = std_a_griod

    def plot_stats(self, title=None):
        """Plot statistics of estimating parameters of a single cell

        Method to plot the marginal and joint distributions of samples drawn from the 
        posterior distribution as well as the extracted statistics. Calls the function 
        ``plateflex.plotting.plot_stats`` with attributes as arguments.

        Args:
            est (bool): type of inference estimate to use for predicting admittance and coherence
            title (str): title of plot
        """

        try:
            plotting.plot_stats(self.cell_trace, self.cell_summary, \
                self.cell_map_estimate, title=title)
        except:
            raise(Exception("No 'cell' estimate available"))

    def plot_fitted(self, est='MAP', title=None):
        """Plot fitted admittance and coherence functions

        Method to plot observed and fitted admittance and coherence functions using 
        one of ``MAP`` or ``mean`` estimates. Calls the function ``plateflex.plotting.plot_fitted``
        with attributes as arguments.

        Args:
            est (bool): type of inference estimate to use for predicting admittance and coherence
            title (str): title of plot
        """

        if est not in ['mean', 'MAP']:
            raise(Exception("Choose one among: 'mean', or 'MAP'"))
            
        try:
            cell = self.cell
            k = self.k
            adm = self.wl_admit[cell[0], cell[1], :]
            eadm = self.ewl_admit[cell[0], cell[1], :]
            coh = self.wl_coh[cell[0], cell[1], :]
            ecoh = self.ewl_coh[cell[0], cell[1], :]

            # Call function from ``plotting`` module
            plotting.plot_fitted(k, adm, eadm, coh, ecoh, self.cell_summary, \
                self.cell_map_estimate, est=est, title=title)

        except:
            raise(Exception("No estimate yet available"))

    def plot_results(self, mean_Te=False, MAP_Te=False, std_Te=False, \
        mean_F=False, MAP_F=False, std_F=False, mean_a=False, MAP_a=False, \
        std_a=False, mask=None):
        """Plot grids of estimated parameters

        Method to plot grids of estimated parameters with fixed labels and titles. 
        To have more control over the plot rendering, use the function 
        ``plateflex.plotting.plot_real_grid`` with the relevant quantities and 
        plotting options.

        Args:
            mean/MAP/std_Te/F/a (bool): all variables default to False (no plot generated)
        """

        if mean_Te:
            plotting.plot_real_grid(self.mean_Te_grid, mask=mask, \
                title='Mean of posterior', clabel='Te (km)')
        if MAP_Te:
            plotting.plot_real_grid(self.MAP_Te_grid, mask=mask, \
                title='MAP estimate', clabel='Te (km)')
        if std_Te:
            plotting.plot_real_grid(self.std_Te_grid, mask=mask, \
                title='Std of posterior', clabel='Te (km)')
        if mean_F:
            plotting.plot_real_grid(self.mean_F_grid, mask=mask, \
                title='Mean of posterior', clabel='F')
        if MAP_F:
            plotting.plot_real_grid(self.MAP_F_grid, mask=mask, \
                title='MAP estimate', clabel='F')
        if std_F:
            plotting.plot_real_grid(self.std_F_grid, mask=mask, \
                title='Std of posterior', clabel='F')
        if mean_a:
            try:
                plotting.plot_real_grid(self.mean_a_grid, mask=mask, \
                    title='Mean of posterior', clabel=r'$\alpha$')
            except:
                print("parameter 'alpha' was not estimated")
        if MAP_a:
            try:
                plotting.plot_real_grid(self.MAP_a_grid, mask=mask, \
                    title='MAP estimate', clabel=r'$\alpha$')
            except:
                print("parameter 'alpha' was not estimated")
        if std_a:
            try:
                plotting.plot_real_grid(self.std_a_grid, mask=mask, \
                    title='Std of posterior', clabel=r'$\alpha$')
            except:
                print("parameter 'alpha' was not estimated")

def _npow2(x):
    return 1 if x==0 else 2**(x-1).bit_length()

def _lam2k(nx, ny, dx, dy):
    """Obtain optimal wavenumbers from grid parameters

    Calculate the optimal set of equally-spaced equivalent wavenumbers for given grid 
    parameters to be used in the wavelet analysis through the ``plateflex.cpwt`` 
    module. 

    Args:
        nx: int   
            Size of grid in x direction
        ny: int
            Size of grid in y direction
        dx: float 
            Sample distance in x direction (km)
        dy: float 
            Sample distance in y direction (km)

    Return:
        (tuple): tuple containing:
            * ns (int): Size of wavenumber array
            * k (np.ndarray): Wavenumbers (rad/m)

    Note:
        This is different from the exact Fourier wavenumbers calculated for 
        the grid using ``numpy.fft.fftfreq``, as the continuous wavelet
        transform can be defined at arbitrary wavenumbers.

    Example
    -------
    >>> from plateflex import classes
    >>> # Define fake grid
    >>> nx = ny = 300
    >>> dx = dy = 20.
    >>> classes._lam2k(nx,ny,dx,dy)
    (18, array([2.96192196e-06, 3.67056506e-06, 4.54875180e-06, 5.63704569e-06,
           6.98571509e-06, 8.65705514e-06, 1.07282651e-05, 1.32950144e-05,
           1.64758613e-05, 2.04177293e-05, 2.53026936e-05, 3.13563910e-05,
           3.88584421e-05, 4.81553673e-05, 5.96765921e-05, 7.39542826e-05,
           9.16479263e-05, 1.13574794e-04]))

    """

    # Calculate min and max wavelengths from grid
    maxlam = np.sqrt((nx*dx)**2. + (ny*dy)**2.)/4.*1.e3
    minlam = np.sqrt((2.*dx)**2. + (2.*dy)**2.)*1.e3

    # Assign first wavenumber
    lam = []; k = []; s = []
    lam.append(maxlam)
    k.append(2.*np.pi/lam[0])
    s.append(cf.k0/k[0])
    dk = np.sqrt(-2.*np.log(cf.p))/s[0]
    ss = 1

    # Loop through k's until minlam is reached
    while lam[ss-1]>minlam:
            s = (cf.k0-np.sqrt(-2.*np.log(cf.p)))/(k[ss-1]+dk)
            k.append(cf.k0/s)
            lam.append(2.*np.pi/k[ss])
            dk = np.sqrt(-2.0*np.log(cf.p))/s
            ss = ss + 1
    ns = ss

    # Compute wavenumbers
    lam = np.array(lam)
    k = np.array(2.*np.pi/lam)

    return ns, k
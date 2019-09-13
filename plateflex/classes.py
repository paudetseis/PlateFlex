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
from plateflex.cpwt import cpwt
from plateflex import conf as cf
from plateflex import plotting
import seaborn as sns
sns.set()


class FlexGrid(object):
    """
    Accepts a 2D array and Cartesian coordinates specifying the
    bounding box of the array

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
            >>> # FIGURE THIS OUT!!!

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

    def __init__(self, grid, xmin, xmax, ymin, ymax):

        FlexGrid.__init__(self, grid, xmin, xmax, ymin, ymax)


class BougGrid(GravGrid):

    def __init__(self, grid, xmin, xmax, ymin, ymax):

        GravGrid.__init__(self, grid, xmin, xmax, ymin, ymax)

        self.units = 'mGal'

    def plot(self, mask=None, title='Bouguer anomaly', save=None, clabel=None):

        plotting.plot_real_grid(self.data, title=title, mask=mask, save=save, clabel=self.units)


class FairGrid(GravGrid):

    def __init__(self, grid, xmin, xmax, ymin, ymax):

        GravGrid.__init__(self, grid, xmin, xmax, ymin, ymax)

        self.units = 'mGal'

    def plot(self, mask=None, title='Free air anomaly', save=None, clabel=None):

        plotting.plot_real_grid(self.data, title=title, mask=mask, save=save, clabel=self.units)


class TopoGrid(FlexGrid):

    def __init__(self, grid, xmin, xmax, ymin, ymax):

        FlexGrid.__init__(self, grid, xmin, xmax, ymin, ymax)

        if np.std(self.data) < 20.:
            self.data *= 1.e3

        self.units = 'meters'

    def plot(self, mask=None, title='Topography', save=None, clabel=None):

        plotting.plot_real_grid(self.data, title=title, mask=mask, save=save, clabel=self.units)


class Project(object):

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
        Return a robust iterator for FlexGrid objects

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
        elif isinstance(grid_list, FlexJointGrid):
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

        Attributes:
            wl_admit (np.ndarray): Wavelet admittance (shape (`nx,ny,ns`))
            wl_eadmit (np.ndarray): Error of wavelet admittance (shape (`nx,ny,ns`))
            wl_coh (np.ndarray): Wavelet coherence (shape (`nx,ny,ns`))
            wl_ecoh (np.ndarray): Error of wavelet coherence (shape (`nx,ny,ns`))

        Returns:
            wl_admit (np.ndarray): Wavelet admittance (shape (`nx,ny,ns`))
            wl_eadmit (np.ndarray): Error of wavelet admittance (shape (`nx,ny,ns`))
            wl_coh (np.ndarray): Wavelet coherence (shape (`nx,ny,ns`))
            wl_ecoh (np.ndarray): Error of wavelet coherence (shape (`nx,ny,ns`))
        """

        if len(self.grids)!=2:
            raise(Exception('There needs to be exactly two FlexGrid objects in Project'))

        if not any(isinstance(g, TopoGrid) for g in self.grids):
            raise(Exception('There needs to be one TopoGrid object in Project'))

        if not any(isinstance(g, GravGrid) for g in self.grids):
            raise(Exception('There needs to be one GravGrid object in Project'))

        self.k = self.grids[0].k
        self.ns = self.grids[0].ns

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

        wl_admit, ewl_admit, wl_coh, ewl_coh = \
            cpwt.wlet_admit_coh(wl_trans_topo, wl_trans_grav)

        self.wl_admit = wl_admit
        self.ewl_admit = ewl_admit
        self.wl_coh = wl_coh
        self.ewl_coh = ewl_coh

        return self.k, wl_admit, ewl_admit, wl_coh, ewl_coh

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
        plotting.plot_real_grid(adm, log=log, mask=mask, title=title, save=save, clabel=None)


def _npow2(x):
    return 1 if x==0 else 2**(x-1).bit_length()


def _lam2k(nx, ny, dx, dy):
    """
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
    >>> import plateflex.utils as ut
    >>> # Define fake grid
    >>> nx = ny = 300
    >>> dx = dy = 20.
    >>> ut.lam2k(nx,ny,dx,dy)
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

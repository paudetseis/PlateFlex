# Copyright 2019 Pascal Audet

# This file is part of Telewavesim.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from plateflex import conf as cf

def npow2(x):
    return 1 if x==0 else 2**(x-1).bit_length()

def lam2k(nx, ny, dx, dy):
    """
    Calculates the optimal set, equally-spaced equivalent wavenumbers for given grid 
    parameters to be used in the wavelet analysis through the ``plateflex.cpwt`` 
    module. 

    Args:
        nx (int):   Size of grid in x direction
        ny (int):   Size of grid in y direction
        dx (float): Sample distance in x direction (km)
        dy (float): Sample distance in y direction (km)

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

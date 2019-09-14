# Copyright 2019 Pascal Audet
#
# This file is part of Telewavesim.
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

Configuration module to set up global variables

Variables are:
    ``wavelet``:
        - k0 (float): Internal Morlet wavenumber (5.336 or higher)
        - p (float): Separation between adjacent wavenumbers (0.85)

    ``Earth parameters``:
        - E (float): Young's modulus (1.e11 Pa)
        - nu (float): Poisson's ratio (0.25)
        - g (float): Gravitational acceleration (9.81 m/s^2)
        - G (float): Gravitational constant (6.67e-11*1.e5 mGal)
        - zc (float): Crustal thickness (35.e3 m)
        - rhom (float): Uppermost mantle density (3200. kg/m^3)
        - rhoc (float): Crustal density (2700. kg/m^3)
        - rhoa (float): Air density (0. kg/m^3)
        - rhow (float): Water density (1030. kg/m^3)
        - rhof (float): Fluid density at topo/fluid interface (==rhoa or ==rhow)
        - wd (float): Water depth (0.e3 m)

    ``flex analysis``:
        - boug (bool): True: gravity anomaly is Bouguer; False: gravity anomaly is Free-air
        - water (bool): True: Include loading from water column; False: no water column
        
    ``bayes parameters``:
        - samples (int): Number of samples in single MCMC chain
        - tunes (int): Number of tuning samples
        - cores (int): Number of cores (i.e., parallel chains). For parallel runs, set conf.cores=1

"""

# wavelet parameters
global k0, p
k0 = 5.336
p = 0.85

# Earth parameters
global E, nu, g, G, zc, rhom, rhoc, rhow, rhoa, rhof, wd
E = 1.e11
nu = 0.25
g = 9.81
G = 6.67e-11*1.e5
zc = 35.*1.e3
rhom = 3200.
rhoc = 2700.
rhow = 1030.
rhoa = 0.
rhof = rhoa
wd = 0.

# flex analysis
global boug, water
boug = True
water = False

# bayes parameters
global samples, tunes
samples = 200
tunes = 200
cores = 4
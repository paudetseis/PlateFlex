# Copyright 2019 Pascal Audet

# This file is part of PlateFlex.

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

'''

Functions to calculate spectral quantities with given Te, F and alpha values.

'''
import numpy as np
import plateflex.conf as cf

def _flexfilter(psi, typ):
    if typ=='top':
        return -((cf.rhoc-cf.rhof)/(cf.rhom-cf.rhoc))*(1. + \
            psi/(cf.rhom-cf.rhoc)/cf.g)**(-1.)
    elif typ=='bot':
        return -((cf.rhoc-cf.rhof)/(cf.rhom-cf.rhoc))*(1. + \
            psi/(cf.rhoc-cf.rhof)/cf.g)


def _decon(theta, phi, k, A):
    
    mu_h = 1./(1.-theta)
    mu_w = 1./(phi-1.)
    nu_h = 2.*np.pi*cf.G*(A*(cf.rhoc-cf.rhof)*np.exp(-k*cf.wd) + \
        (cf.rhom-cf.rhoc)*theta*np.exp(-k*(cf.zc+cf.wd)))
    nu_h = nu_h/(1.-theta)
    nu_w = 2.*np.pi*cf.G*(A*(cf.rhoc-cf.rhof)*np.exp(-k*cf.wd) + \
        (cf.rhom-cf.rhoc)*phi*np.exp(-k*(cf.zc+cf.wd)))
    nu_w = nu_w/(phi-1.)

    return mu_h, mu_w, nu_h, nu_w


def _tr_func(mu_h, mu_w, nu_h, nu_w, F, alpha):
    
    r = (cf.rhoc-cf.rhof)/(cf.rhom-cf.rhoc)
    f = F/(1. - F)
    hg = nu_h*mu_h + nu_w*mu_w*(f**2)*(r**2) + (nu_h*mu_w + nu_w*mu_h)*f*r*np.cos(alpha) + \
        1j*(nu_h*mu_w - nu_w*mu_h)*f*r*np.sin(alpha)
    hh = mu_h**2 + (mu_w*f*r)**2 + 2.*mu_h*mu_w*f*r*np.cos(alpha)
    gg = nu_h**2 + (nu_w*f*r)**2 + 2.*nu_h*nu_w*f*r*np.cos(alpha)
    admittance = hg/hh
    coherency = hg/np.sqrt(hh)/np.sqrt(gg)
    coherence = np.real(coherency)**2
    
    return admittance, coherence


def real_xspec_functions(k, Te, F, alpha=np.pi/2., wd=0.):
    """
    Calculate analytical expressions for the real component of admittance, 
    coherency and coherence functions. 

    Args:
        k (np.ndarray)  : Wavenumbers (rad/m)
        Te (float)      : Effective elastic thickness (km)
        F (float)       : Subruface-to-surface load ratio [0, 1[
        alpha (float, optional)   : Phase difference between initial applied loads (rad)
        wd (float, optional)      : Water layer thickness 

    Returns:
        (tuple): tuple containing:
            * admittance (np.ndarray)    : Real admittance function (shape ``len(k)``)
            * coherence (np.ndarray)     : Coherence functions (shape ``len(k)``)

    """

    if cf.boug:
        A = 0.
    else:
        A = 1.

    if cf.water:
        cf.rhof = cf.rhow
        cf.wd = wd*1.e3
    else:
        cf.rhof = cf.rhoa
        cf.wd = 0.

    # Te in meters
    Te = Te*1.e3

    # Flexural rigidity
    D = cf.E*Te**3/12./(1.-cf.nu**2.)

    # Isostatic function
    psi = D*k**4.

    # Flexural filters
    theta = _flexfilter(psi, 'top')
    phi = _flexfilter(psi, 'bot')
    mu_h, mu_w, nu_h, nu_w = _decon(theta, phi, k, A)

    # Get spectral functions
    admittance, coherence = _tr_func(mu_h, mu_w, nu_h, nu_w, F, alpha)

    # Get real-valued admittance
    admittance = np.real(admittance)

    return admittance, coherence


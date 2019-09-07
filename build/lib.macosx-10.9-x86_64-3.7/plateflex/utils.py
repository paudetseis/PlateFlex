import numpy as np
from plateflex import conf as cf

#------------------------------------------------
#     Subroutine lam2k
#
#     Computes wavenumbers for given nx,ny,dx,dy
#------------------------------------------------
def lam2k(nx, ny, dx, dy):

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

import numpy as np
import matplotlib.pyplot as plt

def npow2(x):
    return 1 if x==0 else 2**(x-1).bit_length()
 
pars = np.loadtxt('plateflex/examples/data/params.txt')
grid = np.loadtxt('plateflex/examples/data/boug.txt')

nx = int(pars[1])
ny = int(pars[0])
dx = float(pars[3])
dy = float(pars[2])

grid = grid.reshape(nx, ny)[::-1]

from plateflex import utils as ut
ns, k = ut.lam2k(nx, ny, dx, dy)

nnx = npow2(nx)
nny = npow2(ny)

plt.imshow(grid, origin='lower')
plt.colorbar()
plt.show()


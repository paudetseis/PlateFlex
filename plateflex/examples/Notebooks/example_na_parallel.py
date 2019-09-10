import numpy as np
import matplotlib.pyplot as plt
from plateflex.cpwt import cpwt
from plateflex.cpwt import conf as cf_f
import plateflex.conf as cf
import plateflex as pf
from joblib import Parallel, delayed


def estimate_parallel(k, admit, eadmit, cohere, ecohere, x, y, nn):
    print(x,y)
    # if grid1[x,y] < 200.:
    #     continue
    adm = admit[x,y,:]
    coh = cohere[x,y,:]
    eadm = eadmit[x,y,:]
    ecoh = ecohere[x,y,:]
    trace, map_estimate, summary = \
        pf.estimate.bayes_real_estimate(\
            k, adm, eadm, coh, ecoh, alph=True, typ='joint')

    results = pf.estimate.get_estimates(summary, map_estimate)

    return results

cf_f.k0 = cf.k0
cf.samples = 1000
cf.tunes = 1000
cf.cores = 1

pars = np.loadtxt('../data/params.txt')
grid1 = np.loadtxt('../data/retopo.txt')
grid2 = np.loadtxt('../data/boug.txt')

nx = int(pars[1])
ny = int(pars[0])
dx = float(pars[3])
dy = float(pars[2])

grid1 = grid1.reshape(nx, ny)[::-1]
grid2 = grid2.reshape(nx, ny)[::-1]

pf.plotting.plot_real_grid(grid1)
pf.plotting.plot_real_grid(grid2)

ns, k = pf.utils.lam2k(nx, ny, dx, dy)

nnx = pf.utils.npow2(nx)   # Integrate this into cpwt
nny = pf.utils.npow2(ny)

wl_trans1 = cpwt.wlet_transform(grid1, nnx, nny, dx, dy, k)
wl_trans2 = cpwt.wlet_transform(grid2, nnx, nny, dx, dy, k)

wl_admit, ewl_admit, wl_coh, ewl_coh = cpwt.wlet_admit_coh(wl_trans1, wl_trans2)

# Initiate multi-processing
num_cores = 8

nn = 5

# Run nested for loop in parallel to cover the whole grid
results = Parallel(n_jobs=num_cores)(delayed(estimate_parallel) \
    (k, wl_admit, ewl_admit, wl_coh, ewl_coh, x, y, nn) for x in range(0, nx-nn, nn) \
    for y in range(0, ny-nn, nn))

# Distribute the parameters back to space
for k,res in enumerate(results):
    res_mte[k] = res[0]
    res_ste[k] = res[1]
    res_bte[k] = res[4]
    res_mF[k] = res[5]
    res_sF[k] = res[6]
    res_bF[k] = res[9]
    res_ma[k] = res[10]
    res_sa[k] = res[11]
    res_ba[k] = res[14]

mte_grid = np.reshape(res_mte,(int(nx/nn),int(ny/nn)))
ste_grid = np.reshape(res_ste,(int(nx/nn),int(ny/nn)))
bte_grid = np.reshape(res_bte,(int(nx/nn),int(ny/nn)))
mF_grid = np.reshape(res_mF,(int(nx/nn),int(ny/nn)))
sF_grid = np.reshape(res_sF,(int(nx/nn),int(ny/nn)))
bF_grid = np.reshape(res_bF,(int(nx/nn),int(ny/nn)))
ma_grid = np.reshape(res_ma,(int(nx/nn),int(ny/nn)))
sa_grid = np.reshape(res_sa,(int(nx/nn),int(ny/nn)))
ba_grid = np.reshape(res_ba,(int(nx/nn),int(ny/nn)))

mask = grid1 < -200.

pf.plotting.plot_real_grid(mte_grid, title='Mean Te (km)', mask=mask)
pf.plotting.plot_real_grid(ste_grid, title='Std Te (km)', mask=mask)
pf.plotting.plot_real_grid(bte_grid, title='MAP Te (km)', mask=mask)

pf.plotting.plot_real_grid(mF_grid, title='Mean F', mask=mask)
pf.plotting.plot_real_grid(sF_grid, title='Std F', mask=mask)
pf.plotting.plot_real_grid(bF_grid, title='MAP F', mask=mask)

pf.plotting.plot_real_grid(ma_grid, title='Mean alpha', mask=mask)
pf.plotting.plot_real_grid(sa_grid, title='Std alpha', mask=mask)
pf.plotting.plot_real_grid(ba_grid, title='MAP alpha', mask=mask)

import numpy as np
import matplotlib.pyplot as plt
from plateflex.cpwt import cpwt
from plateflex.cpwt import conf as cf_f
import plateflex.conf as cf
import plateflex as pf
from joblib import Parallel, delayed
import multiprocessing

def estimate_parallel(k, admit, corr, coh, x, y, nn):
    print(x,y)
    # if grid1[x,y] < 0.:
    #     continue
    adm1d = np.real(admit[x,y,:])
    cor1d = np.real(corr[x,y,:])
    coh1d = coh[x,y,:]
    eadm = eadmit[x,y,:]
    ecor = ecorr[x,y,:]
    ecoh = ecohe[x,y,:]
    trace, map_estimate, summary = \
        pf.estimate.bayes_real_estimate(k, adm1d, cor1d, coh1d, typ='admit_coh')
    # pf.plotting.plot_trace_stats(trace)

    mte, ste, bte, mF, sF, bF = pf.estimate.get_values(map_estimate, summary)

    return [mte, ste, bte, mF, sF, bF]


cf_f.k0 = cf.k0
cf.samples = 500
cf.tunes = 500

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
# pf.plotting.plot_real_grid(grid2)

ns, k = pf.utils.lam2k(nx, ny, dx, dy)

nnx = pf.utils.npow2(nx)   # Integrate this into cpwt
nny = pf.utils.npow2(ny)

wt_grid1 = cpwt.wlet_transform(grid1, nnx, nny, dx, dy, k)
wt_grid2 = cpwt.wlet_transform(grid2, nnx, nny, dx, dy, k)

ws_grid1, ews_grid1 = pf.wavelet.scalogram(wt_grid1)
ws_grid2, ews_grid2 = pf.wavelet.scalogram(wt_grid2)

# pf.plotting.plot_real_grid(ws_grid1[:,:,10], log=True)
# pf.plotting.plot_real_grid(ws_grid2[:,:,10], log=True)

admit, eadmit, corr, ecorr, coh, ecohe = pf.wavelet.admit_corr(wt_grid1, wt_grid2)

te_grid = np.zeros(grid1.shape)
F_grid = np.zeros(grid1.shape)

# Initiate multi-processing
num_cores = multiprocessing.cpu_count()

nn = 10

# Run nested for loop in parallel to cover the whole grid
results = Parallel(n_jobs=num_cores)(delayed(estimate_parallel) \
    (k, admit, corr, coh, x, y, nn) for x in range(0, nx-nn, nn) \
    for y in range(0, ny-nn, nn))

# Distribute the parameters back to space
for k,res in enumerate(results):
    res_mte[k] = res[0]
    res_ste[k] = res[1]
    res_bte[k] = res[2]
    res_mF[k] = res[3]
    res_sF[k] = res[4]
    res_bF[k] = res[5]

mte_grid = np.reshape(res_mte,(int(nx/nn),int(ny/nn)))
ste_grid = np.reshape(res_ste,(int(nx/nn),int(ny/nn)))
bte_grid = np.reshape(res_bte,(int(nx/nn),int(ny/nn)))
mF_grid = np.reshape(res_mF,(int(nx/nn),int(ny/nn)))
sF_grid = np.reshape(res_sF,(int(nx/nn),int(ny/nn)))
bF_grid = np.reshape(res_bF,(int(nx/nn),int(ny/nn)))

pf.plotting.plot_real_grid(mte_grid)
pf.plotting.plot_real_grid(ste_grid)
pf.plotting.plot_real_grid(bte_grid)

pf.plotting.plot_real_grid(mF_grid)
pf.plotting.plot_real_grid(sF_grid)
pf.plotting.plot_real_grid(bF_grid)

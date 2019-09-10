import numpy as np
import matplotlib.pyplot as plt
from plateflex.cpwt import cpwt
from plateflex.cpwt import conf as cf_f
import plateflex.conf as cf
import plateflex as pf

cf.k0 = 7.2
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

# pf.plotting.plot_real_grid(grid1)
# pf.plotting.plot_real_grid(grid2)

ns, k = pf.utils.lam2k(nx, ny, dx, dy)

nnx = pf.utils.npow2(nx)   # Integrate this into cpwt
nny = pf.utils.npow2(ny)

wl_trans1 = cpwt.wlet_transform(grid1, nnx, nny, dx, dy, k)
wl_trans2 = cpwt.wlet_transform(grid2, nnx, nny, dx, dy, k)

wl_sg1, ewl_sg1 = cpwt.wlet_scalogram(wl_trans1)
wl_sg2, ewl_sg2 = cpwt.wlet_scalogram(wl_trans2)

# pf.plotting.plot_real_grid(wl_sg1[:,:,10], log=True, title='Rock-equivalent topo (m^2)/m^2')
# pf.plotting.plot_real_grid(wl_sg2[:,:,10], log=True, title='Bouguer anomaly (mGal^2)/m^2')

wl_admit, ewl_admit, wl_coh, ewl_coh = cpwt.wlet_admit_coh(wl_trans1, wl_trans2)


# One sample from 2D grid
(x,y) = (200,200)
adm = wl_admit[x,y,:]
eadm = ewl_admit[x,y,:]
coh = wl_coh[x,y,:]
ecoh = ewl_coh[x,y,:]

trace, map_estimate, summary = \
    pf.estimate.bayes_real_estimate(k, adm, eadm, coh, ecoh, alph=False, typ='coh')

print(summary)
pf.plotting.plot_fitted(k, adm, eadm, coh, ecoh, summary, map_estimate, est='mean', \
    title='No alpha, coherence')
pf.plotting.plot_trace_stats(trace, summary, map_estimate, \
    title='No alpha, coherence')

trace, map_estimate, summary = \
    pf.estimate.bayes_real_estimate(k, adm, eadm, coh, ecoh, alph=True, typ='coh')

print(summary)
pf.plotting.plot_fitted(k, adm, eadm, coh, ecoh, summary, map_estimate, est='mean', \
    title='With alpha, coherence')
pf.plotting.plot_trace_stats(trace, summary, map_estimate, \
    title='With alpha, coherence')

trace, map_estimate, summary = \
    pf.estimate.bayes_real_estimate(k, adm, eadm, coh, ecoh, alph=True, typ='joint')

print(summary)
pf.plotting.plot_fitted(k, adm, eadm, coh, ecoh, summary, map_estimate, est='mean', \
    title='With alpha, joint admittance and coherence')
pf.plotting.plot_trace_stats(trace, summary, map_estimate, \
    title='With alpha, joint admittance and coherence')





# # Here we invert one in 10 points - for speed
# nn = 10
# mTe_grid = np.zeros((int(nx/nn),int(ny/nn)))
# bTe_grid = np.zeros((int(nx/nn),int(ny/nn)))
# sTe_grid = np.zeros((int(nx/nn),int(ny/nn)))
# mF_grid = np.zeros((int(nx/nn),int(ny/nn)))
# bF_grid = np.zeros((int(nx/nn),int(ny/nn)))
# sF_grid = np.zeros((int(nx/nn),int(ny/nn)))
# ma_grid = np.zeros((int(nx/nn),int(ny/nn)))
# ba_grid = np.zeros((int(nx/nn),int(ny/nn)))
# sa_grid = np.zeros((int(nx/nn),int(ny/nn)))

# for i in range(0, nx-nn, nn):
#     for j in range(0, ny-nn, nn):
        
#         print(i,j)

#         if grid1[i,j] < 0.:
#             continue

#         adm = wl_admit[i,j,:]
#         coh = wl_coh[i,j,:]
#         eadm = ewl_admit[i,j,:]
#         ecoh = ewl_coh[i,j,:]

#         trace, map_estimate, summary = \
#             pf.estimate.bayes_real_estimate(\
#                 k, adm, eadm, coh, ecoh, alph=True, typ='joint')

#         res = pf.estimate.get_estimates(summary, map_estimate)

#         # Distribute the parameters back to space
#         mte = res[0]
#         ste = res[1]
#         bte = res[4]
#         mF = res[5]
#         sF = res[6]
#         bF = res[9]
#         ma = res[10]
#         sa = res[11]
#         ba = res[14]

#         # Store values in smaller arrays
#         mTe_grid[int(i/nn),int(j/nn)] = mte
#         bTe_grid[int(i/nn),int(j/nn)] = bte
#         sTe_grid[int(i/nn),int(j/nn)] = ste
#         mF_grid[int(i/nn),int(j/nn)] = mF
#         bF_grid[int(i/nn),int(j/nn)] = bF
#         sF_grid[int(i/nn),int(j/nn)] = sF
#         ma_grid[int(i/nn),int(j/nn)] = ma
#         ba_grid[int(i/nn),int(j/nn)] = ba
#         sa_grid[int(i/nn),int(j/nn)] = sa

# mask = grid1 < -200.

# pf.plotting.plot_real_grid(mTe_grid, title='Mean Te (km)')
# pf.plotting.plot_real_grid(sTe_grid, title='Std Te (km)')
# pf.plotting.plot_real_grid(bTe_grid, title='MAP Te (km)')

# pf.plotting.plot_real_grid(mF_grid, title='Mean F')
# pf.plotting.plot_real_grid(sF_grid, title='Std F')
# pf.plotting.plot_real_grid(bF_grid, title='MAP F')

# pf.plotting.plot_real_grid(ma_grid, title='Mean alpha')
# pf.plotting.plot_real_grid(sa_grid, title='Std alpha')
# pf.plotting.plot_real_grid(ba_grid, title='MAP alpha')


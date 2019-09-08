import numpy as np
import matplotlib.pyplot as plt
from plateflex.cpwt import cpwt
from plateflex.cpwt import conf as cf_f
import plateflex.conf as cf
import plateflex as pf
import pymc3 as pm
from theano.compile.ops import as_op
import theano.tensor as tt

cf_f.k0 = cf.k0

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

wt_grid1 = cpwt.wlet_transform(grid1, nnx, nny, dx, dy, k)
wt_grid2 = cpwt.wlet_transform(grid2, nnx, nny, dx, dy, k)

ws_grid1, ews_grid1 = pf.wavelet.scalogram(wt_grid1)
ws_grid2, ews_grid2 = pf.wavelet.scalogram(wt_grid2)

# pf.plotting.plot_real_grid(ws_grid1[:,:,10], log=True)
# pf.plotting.plot_real_grid(ws_grid2[:,:,10], log=True)

admit, eadmit, corr, ecorr, coh, ecohe = pf.wavelet.admit_corr(wt_grid1, wt_grid2)

adm = np.real(admit[200,200,:])
cor = np.real(corr[200,200,:])
coh = coh[200,200,:]
eadm = eadmit[200,200,:]
ecor = ecorr[200,200,:]
ecoh = ecohe[200,200,:]

cf.samples=1000
cf.tunes=1000

trace, map_estimate, summary = pf.estimate.bayes_real_estimate(k, adm, cor, coh, typ='admit_coh')

mte, ste, bte, mF, sF, bF = pf.estimate.get_values(map_estimate, summary)

pf.plotting.plot_trace_stats(trace)

mte = np.mean(trace['Te'])
mF = np.mean(trace['F'])
padm, pcor, pcoh = pf.flexure.real_xspec_functions(k, mte, mF, 90.)

pf.plotting.plot_fitted(k, adm, eadm, coh, ecoh, padm, pcoh)
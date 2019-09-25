import numpy as np
import pandas as pd
import plateflex.conf as cf
from plateflex import TopoGrid, FairGrid, ZcGrid, Project

# # To change the wavelet parameter k0
# from plateflex.cpwt import conf_cpwt
# conf_cpwt.k0 = 5.336

# To change the model parameters:
from plateflex.flex import conf_flex
conf_flex.rhoc = 2800.
conf_flex.zc = 7.5e3

# Read header of first data set to get grid parameters
xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, nx, ny = \
pd.read_csv('data/Bathy_PAC.xyz', sep='\t', nrows=0).columns[1:].values.astype(float)
nx = int(nx); ny = int(ny)

# Then read Topo and Bouguer anomaly data 
bathydata = pd.read_csv('data/Bathy_PAC.xyz', sep='\t', \
    skiprows=1, names=['x', 'y', 'z'])['z'].values.reshape(ny,nx)[::-1]
fairdata = pd.read_csv('data/Freeair_PAC.xyz', sep='\t', \
    skiprows=1, names=['x', 'y', 'z'])['z'].values.reshape(ny,nx)[::-1]
zcdata = pd.read_csv('data/Crust_thickness_PAC.xyz', sep='\t', \
    skiprows=1, names=['x', 'y', 'z'])['z'].values.reshape(ny,nx)[::-1]

# Here we can reduce the size of the grids to make things easier to test
bathydata = bathydata[100:356, 100:356]
fairdata = fairdata[100:356, 100:356]
zcdata = zcdata[100:356, 100:356]

# Load the data as `plateflex` Grid objects
bathy = TopoGrid(bathydata, dx, dy)
fair = FairGrid(fairdata, dx, dy)
zc = ZcGrid(zcdata, dx, dy)

# Create contours
contours = bathy.make_contours(0.)

# Make mask 
mask = (bathy.data > 0.)

# Plot topo and boug with mask and contours
bathy.plot(mask=mask, contours=contours, cmap='Spectral_r', vmin=-6000., vmax=6000.)
fair.plot(mask=mask, contours=contours, cmap='seismic', vmin=-200., vmax=200.)
zc.plot(mask=mask, contours=contours, cmap='Spectral_r', vmin=0., vmax=40000.)

# Produce filtered version of water depth
bathy.filter_water_depth()
# bathy.plot_water_depth()

# Assign new Project
project = Project(grids=[bathy, fair, zc])

# Initialize it
project.init()

# Calculate wavelet admittance and coherence
project.wlet_admit_coh()

# Specify mask for cell locations to skip in analysis
project.mask = mask

cell = (200, 56)

project.inverse='L2'

project.estimate_cell(cell, alph=False, atype='admit')
project.plot_functions()
print(project.summary)

project.inverse='bayes'

project.estimate_cell(cell, alph=False, atype='admit')
project.plot_functions()
project.plot_bayes_stats()
print(project.summary)


project.estimate_grid(2, alph=False, atype='admit')
project.plot_results(mean_Te=True, contours=contours, mask=True, cmap='Spectral', vmin=0., vmax=40., sigma=1)
project.plot_results(std_Te=True, contours=contours, mask=True, cmap='Spectral', vmin=0., vmax=20., sigma=1)
project.plot_results(mean_F=True, contours=contours, mask=True, cmap='plasma', vmin=0., vmax=1., sigma=1)
project.plot_results(std_F=True, contours=contours, mask=True, cmap='plasma', vmin=0., vmax=0.5, sigma=1)
project.plot_results(chi2=True, contours=contours, mask=True, cmap='cividis', vmin=0., vmax=10., sigma=1)

# project.estimate_grid(2, alph=False, atype='joint')
# project.plot_results(mean_Te=True, contours=contours, mask=True, cmap='Spectral', vmin=0., vmax=40., sigma=1)
# project.plot_results(std_Te=True, contours=contours, mask=True, cmap='Spectral', vmin=0., vmax=20., sigma=1)
# project.plot_results(mean_F=True, contours=contours, mask=True, cmap='plasma', vmin=0., vmax=1., sigma=1)
# project.plot_results(std_F=True, contours=contours, mask=True, cmap='plasma', vmin=0., vmax=0.5, sigma=1)
# project.plot_results(chi2=True, contours=contours, mask=True, cmap='cividis', vmin=0., vmax=10., sigma=1)

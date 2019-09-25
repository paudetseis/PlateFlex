import numpy as np
import pandas as pd
import plateflex.conf as cf
from plateflex import TopoGrid, BougGrid, Project

# To change the wavelet parameter k0
from plateflex.cpwt import conf_cpwt
conf_cpwt.k0 = 5.336

# Read header of first data set to get grid parameters
xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, nx, ny = \
pd.read_csv('data/Topo_NA.xyz', sep='\t', nrows=0).columns[1:].values.astype(float)
nx = int(nx); ny = int(ny)

# Then read Topo and Bouguer anomaly data 
topodata = pd.read_csv('data/Topo_NA.xyz', sep='\t', \
    skiprows=1, names=['x', 'y', 'z'])['z'].values.reshape(ny,nx)[::-1]
bougdata = pd.read_csv('data/Bouguer_NA.xyz', sep='\t', \
    skiprows=1, names=['x', 'y', 'z'])['z'].values.reshape(ny,nx)[::-1]

# Load the data as `plateflex` Grid objects
topo = TopoGrid(topodata, dx, dy)
boug = BougGrid(bougdata, dx, dy)

# Create contours
contours = topo.make_contours(0.)

# Make mask 
mask = (topo.data < -500.)

# Plot topo and boug with mask and contours
topo.plot(mask=mask, contours=contours)

project = Project(grids=[topo, boug])
project.wlet_admit_coh()

# Specify mask for cell locations to skip in analysis
project.mask = mask

# cell = (200, 56)

project.inverse='L2'

# project.estimate_cell(cell, alph=True, atype='joint')
# project.plot_functions()
# print(project.summary)

# project.inverse='bayes'

# project.estimate_cell(cell, alph=True, atype='joint')
# project.plot_functions()
# project.plot_bayes_stats()
# print(project.summary)


project.estimate_grid(10, alph=True, atype='joint')
project.plot_results(mean_Te=True, mean_F=True, mean_a=True, chi2=True, contours=contours, mask=True)


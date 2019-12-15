
.. figure:: ../plateflex/examples/picture/plateflex_logo.png
   :align: center

.. automodule:: plateflex
   :members:

   Global Variables
   ----------------

Making gridded data
-------------------

Packaged with this software is a set of topography, gravity and crustal structure data that 
are easily imported into the ``PlateFlex`` package. Those data sets have been produced by 
processing global grids of data from the 
`WGM2012 <http://bgi.omp.obs-mip.fr/data-products/Grids-and-models/wgm2012>`_, 
`GEBCO <https://www.gebco.net/data_and_products/gridded_bathymetry_data/>`_ and 
`CRUST1.0 <https://igppweb.ucsd.edu/~gabi/crust1.html>`_ models using 
`GMT <https://www.generic-mapping-tools.org>`_.

The WGM2012 models are used to extract Topography, Bouguer and Free-air anomaly data 
over North America, and Free-air anomaly data over the NW Pacific.
The GEBCO model is used for the Bathymetry over the NW Pacific.
The CRUST1.0 models are used to extract the crustal thickness and crustal 
density (corresponding to upper crust, or layer #6) over both North America and the NW Pacific.
Since those models are all defined using a different registration and grid sampling, 
we use GMT to manipulate the grids and produce consistent data sets to use with 
``PlateFlex``. We used GMT 5.4.1 in the following examples. These commands should be 
copied to a bash script file and executed from the terminal.

WGM2012
*******

These models are defined at a resolution of 2' with a gridline registration.

.. sourcecode:: bash

    # Set region and projection for North America. Use a transverse mercator 
    # with a central meridian of -95 degrees to minimize high-latitude distortions.
    reg=-R-125/13/-28/58+r
    proj=-JT-95/3i

    # Make map of region
    ps=Bouguer_NA.ps
    gmt grdimage WGM2012_Bouguer_ponc_2min.grd $reg $proj -K -P \
                -B10WSne -CPALET_WGM_Bouguer_Global.cpt -X1.5i -Y1.25i > $ps
    gmt psscale -CPALET_WGM_Bouguer_Global.cpt -DJRM+o0.25i/0+e+mc -R -J -O >> $ps

    # Project onto Cartesian grid with sampling interval of 20 km
    gmt grdproject WGM2012_Bouguer_ponc_2min.grd -GBouguer_NA.grd $proj $reg -D20 -Fk

    # Save grid information as header for use with PlateFlex software
    gmt grdinfo -C Bouguer_NA.grd > header.txt
    sed 's/^/# /g' header.txt > Bouguer_NA.xyz 
    gmt grd2xyz Bouguer_NA.grd >> Bouguer_NA.xyz

You can repeat these commands for the corresponding Free-air anomaly and Topography data.

GEBCO
*****

This model is defined on a very fine grid (15" resolution) with a gridline registration. 
We first resample the grid onto a 2' grid.

.. sourcecode:: bash

    # Set region and projection for NW Pacific - here we use a regular 
    # Mercator since it straddles the equator.
    reg=-R145/215/-15/45
    proj=-JM3i

    # Resample on a 2' grid
    gmt grdsample GEBCO_data/GEBCO_2019.nc -GGEBCO_resampled_2m.grd -I2m -Rd

    # Make map of region
    ps=Bathy_PAC.ps
    gmt grdimage GEBCO_resampled_2m.grd $reg $proj -K -P \
                -B10WSne -CPALET_WGM_ETOPO1_Global.cpt -X1.5i -Y1.25i > $ps
    gmt psscale -CPALET_WGM_ETOPO1_Global.cpt -DJRM+o0.25i/0+e+mc -R -J -O >> $ps

    # Project onto Cartesian grid
    gmt grdproject GEBCO_resampled_2m.grd -GBathy_PAC.grd $proj $reg -D10 -Fk

    # Dump grid onto ASCII file with header
    gmt grdinfo -C Bathy_PAC.grd > header.txt
    sed 's/^/# /g' header.txt > Bathy_PAC.xyz 
    gmt grd2xyz Bathy_PAC.grd >> Bathy_PAC.xyz

CRUST1.0
********

The CRUST1.0 model is defined on a very coarse grid (1 degree). This is not a 
problem as we only use those fields in the inversion step, and not in the 
calculation of the wavelet transform. However, we still require an identical 
grid specification, so the first step is to resample them on a fine grid 
using spline interpolation.

.. note:: 

    You first need to extract the proper layer within CRUST1.0. For crustal 
    thickness, the weblink contains a global grid of crustal thickness (crsthk.xyz). 
    For the density layer, you will have to use the provided fortran code getCN1xyz.f 
    and rename the output file xyz-ro6 to something like crustal_density.xyz 
    (for example). Layer 6 corresponds to the upper crustal layer. See the weblink for details.

.. sourcecode:: bash

    # Set region and projection for North America
    reg=-R145/215/-15/45
    proj=-JM3i

    # Create 2 .grd file from the ASCII data
    gmt xyz2grd crsthk.xyz -Gcrust_thickness_1deg.grd -Rd -I1 -T

    # Resample on a 2' grid using spline interpolation
    gmt grdsample crust_thickness_1deg.grd -Gcrust_thickness_2m.grd -nb -Rg -I2m

    # Make map of region
    ps=Crust_thick_PAC.ps
    gmt makecpt -Chaxby -T5/30/1 > crust.cpt
    gmt grdimage crust_thickness_2m.grd $reg $proj -K -P \
                -B10WSne -Ccrust.cpt -X1.5i -Y1.25i > $ps
    gmt psscale -Ccrust.cpt -DJRM+o0.25i/0+e+mc -R -J -O >> $ps

    # Project onto Cartesian grid
    gmt grdproject crust_thickness_2m.grd -Gcrust_thickness.grd $proj $reg -D10 -Fk

    # Dump grid onto ASCII file with header
    gmt grdinfo -C crust_thickness.grd > header.txt
    sed 's/^/# /g' header.txt > crustal_thickness_PAC.xyz 
    gmt grd2xyz crust_thickness.grd >> crustal_thickness_PAC.xyz

# Copyright 2019 Pascal Audet
#
# This file is part of PlateFlex.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""

PlateFlex is a software for estimating the effective elastic thickness of the lithosphere
from the inversion of flexural isostatic response functions calculated from a wavelet
analysis of gravity and topography data.

Licence
-------

Copyright 2019 Pascal Audet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Installation
------------

Dependencies
++++++++++++

The current version was developed using **Python3.7** \
Also, the following packages are required:

- ``gfortran`` (https://gcc.gnu.org/wiki/GFortran) (or any Fortran compiler)
- ``numpy`` (https://numpy.org)
- ``pymc3`` (https://docs.pymc.io)
- ``seaborn`` (https://seaborn.pydata.org)

The following package is useful to draw outline of land areas, and is required
if there are NaNs in the data set (which will be interpolated over using
a ``scikit-image`` function):

- ``scikit-image`` (https://scikit-image.org)

See below for full installation details. 

Conda environment
+++++++++++++++++

We recommend creating a custom ``conda`` environment
where ``plateflex`` can be installed along with its dependencies. This will ensure
that all packages are compatible.

.. sourcecode:: bash

   conda create -n pflex python=3.8 fortran-compiler numpy pymc3 matplotlib seaborn scikit-image -c conda-forge

Activate the newly created environment:

.. sourcecode:: bash

   conda activate pflex

Installing from source
++++++++++++++++++++++

- Clone the repository:

.. sourcecode:: bash

   git clone https://github.com/paudetseis/PlateFlex.git
   cd PlateFlex

- Install using pip:

.. sourcecode:: bash

   pip install .

# .. note::

#     If you run into problems during installation using MacOS due to LLVM versions, make sure
#     you update XCode, then try creating 
#     a ``conda`` environment where only ``python`` is installed, install the
#     ``conda``-provided gfortran package, then install ``numpy`` and ``pip install .``.
#     The dependencies can be installed afterwards.

Jupyter Notebooks
+++++++++++++++++

Included in this package is a set of Jupyter Notebooks, which give examples on how to create ``Grid`` objects and estimate
the flexural parameters over whole grids. The Notebooks describe how to produce pulication quality results that closely
match those published in Audet (2014) and Kirby and Swain. (2009) for North America, as well
as those of Kalnins and Watts (2009) for the NW Pacific.

After installing ``plateflex``, these notebooks can be locally installed (i.e., in a local folder ``Examples``) 
from the package by running:

.. sourcecode:: python

    from plateflex import doc
    doc.install_doc(path='Examples')

To run the notebooks you will have to further install ``jupyter``:

.. sourcecode:: bash

    conda install jupyter

Then:

.. sourcecode:: bash

    unzip data.zip
    cd Examples
    jupyter notebook

You can then save the notebooks as ``python`` scripts and you should be good to go!

"""

__version__ = '0.1.0'

__author__ = 'Pascal Audet'


# -*- coding: utf-8 -*-
from . import conf as cf
from . import estimate
from . import plotting
from .classes import Grid, TopoGrid, GravGrid, BougGrid, FairGrid
from .classes import RhocGrid, ZcGrid, Project
from .cpwt import conf_cpwt as cf_wt
from .flex import conf_flex as cf_fl


def set_conf_cpwt():
    cf_wt.k0 = 5.336


def set_conf_flex():
    cf_fl.zc = 35.*1.e3
    cf_fl.rhom = 3200.
    cf_fl.rhoc = 2700.
    cf_fl.rhow = 1030.
    cf_fl.rhoa = 0.
    cf_fl.rhof = cf_fl.rhoa
    cf_fl.wd = 0.
    cf_fl.boug = 1


set_conf_cpwt()
set_conf_flex()


def get_conf_cpwt():
    """
    Print global variable that controls the spatio-spectral resolution of the
    wavelet transform

    .. rubric:: Example

    >>> import plateflex 
    >>> plateflex.get_conf_cpwt()
    Wavelet parameter used in plateflex.cpwt:
    -----------------------------------------
    [Internal wavenumber]      k0 (float):     5.336
    """

    print('\n'.join((
        'Wavelet parameter used in plateflex.cpwt:',
        '-----------------------------------------',
        '[Internal wavenumber]      k0 (float):     {0:.3f}'.format(
            cf_wt.k0))))


def get_conf_flex():
    """
    Print global variables that control the setup of the flexural isostatic model

    .. rubric:: Example

    >>> import plateflex 
    >>> plateflex.get_conf_flex()
    Global model parameters currently in use by plateflex:
    ------------------------------------------------------
    [Crustal thickness]        zc (float):     35000 m
    [Mantle density]           rhom (float):   3200 kg/m^3
    [Crustal density]          rhoc (float):   2700 kg/m^3
    [Water density]            rhow (float):   1030 kg/m^3
    [Air density]              rhoa (float):   0 kg/m^3
    [Fluid density]            rhof (float):   0 kg/m^3
    [Water depth]              wd (float):     0 m
    [Bouguer analysis?]        boug (int):     1 ; True
    """

    print('\n'.join((
        'Global model parameters currently in use by plateflex:',
        '------------------------------------------------------',
        '[Crustal thickness]        zc (float):     {0:.0f} m'.format(
            cf_fl.zc),
        '[Mantle density]           rhom (float):   {0:.0f} kg/m^3'.format(
            cf_fl.rhom),
        '[Crustal density]          rhoc (float):   {0:.0f} kg/m^3'.format(
            cf_fl.rhoc),
        '[Water density]            rhow (float):   {0:.0f} kg/m^3'.format(
            cf_fl.rhow),
        '[Air density]              rhoa (float):   {0:.0f} kg/m^3'.format(
            cf_fl.rhoa),
        '[Fluid density]            rhof (float):   {0:.0f} kg/m^3'.format(
            cf_fl.rhof),
        '[Water depth]              wd (float):     {0:.0f} m'.format(
            cf_fl.wd),
        '[Bouguer analysis?]        boug (int):     {0} ; {1}'.format(
            cf_fl.boug, bool(cf_fl.boug)))))

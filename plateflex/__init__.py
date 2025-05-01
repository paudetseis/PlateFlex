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

- A fortran compiler
- ``numpy<1.22`` 
- ``pymc3=3.10.0`` 
- ``seaborn``
- ``setuptools<60`` 
- ``scikit-image`` (https://scikit-image.org)

See below for full installation details. 

Conda environment
+++++++++++++++++

We recommend creating a custom ``conda`` environment
where ``plateflex`` can be installed along with its dependencies. This will ensure
that all packages are compatible.

.. Note::
    In theory, you could use your own fortran compiler. However, to ensure a proper installation,
    it is recommended to install `gfortran=9.3.0` in the `pflex` environment under MacOS,
    or simply specify `fortran-compiler` under Linux.

MacOS
=====

.. sourcecode:: bash

   conda create -n pflex -c conda-forge "python=3.8" "setuptools<60" "gfortran=9.3.0" "numpy<1.22" "pymc3=3.10.0" seaborn scikit-image

.. Note::
    In MacOS, the package `theano-pymc` used by `pymc3` throws a compilation error 
    when using `project.inverse = 'bayes'` and `project.estimate_cell(cell)`.
    We will attempt to resolve this in version 0.1.2.

Linux Ubuntu
============

.. sourcecode:: bash

   conda create -n pflex -c conda-forge "python=3.8" "setuptools<60" fortran-compiler "numpy<1.22" "pymc3=3.10.0" seaborn scikit-image

Activate the newly created environment:

.. sourcecode:: bash

   conda activate pflex

Installing development branch from GitHub
+++++++++++++++++++++++++++++++++++++++++

Install the latest version from the GitHub repository with the following command:

.. sourcecode:: bash

    pip install plateflex@git+https://github.com/paudetseis/plateflex

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

__version__ = '0.1.1'

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

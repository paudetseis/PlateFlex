# Copyright 2019 Pascal Audet
#
# This file is part of Telewavesim.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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
- ``numpy`` (https://github.com/obspy/obspy/wiki)
- ``pymc3`` (https://docs.pymc.io)
- ``seaborn`` (https://seaborn.pydata.org)

See below for full installation details. 

Download the software
+++++++++++++++++++++

- Clone the repository:

.. sourcecode:: bash

   git clone https://github.com/paudetseis/PlateFlex.git
   cd PlateFlex

Conda environment
+++++++++++++++++

We recommend creating a custom ``conda`` environment
where ``telewavesim`` can be installed along with its dependencies.

.. sourcecode:: bash

   conda create -n pflex python=3.7 numpy pymc3 matplotlib seaborn -c conda-forge

or create it from the ``pflex_env.yml`` file:

.. sourcecode:: bash

   conda env create -f pflex_env.yml

Activate the newly created environment:

.. sourcecode:: bash

   conda activate pflex

Installing using pip
++++++++++++++++++++

Once the previous steps are performed, you can install ``telewavesim`` using pip:

.. sourcecode:: bash

   pip install .

.. note::

   Please note, if you are actively working on the code, or making frequent edits, it is advisable
   to perform the pip installation with the ``-e`` flag. This enables an editable installation, where
   symbolic links are used rather than straight copies. This means that any changes made in the
   local folders will be reflected in the packages available on the system.


"""
# -*- coding: utf-8 -*-
from . import conf as cf
from . import estimate
from . import flexure
from . import plotting
from .classes import FlexGrid, TopoGrid, GravGrid, BougGrid, FairGrid, Project
from .cpwt import conf as cf_f

def set_conf():
    cf_f.k0 = cf.k0
    cf_f.p = cf.p

set_conf()
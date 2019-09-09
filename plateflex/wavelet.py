# Copyright 2019 Pascal Audet

# This file is part of Telewavesim.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Functions to calculate spectral quantities from the continuous wavelet
transform obtained from the module ``plateflex.cpwt``.

"""

import numpy as np

def scalogram(wl_trans):
    """
    Calculates the scalogram of a real-valued grid given its continuous wavelet transform.
    Averaging is done over all azimuths.

    Args:
        wl_trans (np.ndarray): Wavelet transform (shape ``(nx, ny, na, ns)``)

    Return:
        (tuple): tuple containing:
            * wl_sg (np.ndarray): Wavelet scalogram (shape ``(nx, ny, ns)``)
            * ewl_sg (np.ndarray): Uncertainty (shape ``(nx, ny, ns)``)

    """
    wl_sg = np.mean(np.abs(wl_trans*np.conj(wl_trans)),axis=(2))
    ewl_sg = np.std(np.abs(wl_trans*np.conj(wl_trans)),axis=(2),ddof=1)/np.sqrt(wl_trans.shape[2])

    return wl_sg, ewl_sg


def xscalogram(wl_trans1, wl_trans2):
    """
    Calculates the cross-scalogram of two real-valued grids given their individual 
    continuous wavelet transform. Averaging is done over all azimuths.

    Args:
        wl_trans1 (np.ndarray): Wavelet transform (shape ``(nx, ny, na, ns)``)
        wl_trans2 (np.ndarray): Wavelet transform (shape ``(nx, ny, na, ns)``)

    Return:
        (tuple): tuple containing:
            * xwl_sg (np.ndarray): Wavelet cross-scalogram (shape ``(nx, ny, ns)``)
            * exwl_sg (np.ndarray): Uncertainty (shape ``(nx, ny, ns)``)

    """

    xwl_sg = np.mean(wl_trans1*np.conj(wl_trans2),axis=(2))
    exwl_sg = np.std(np.abs(wl_trans1*np.conj(wl_trans2)),axis=(2),ddof=1)/np.sqrt(wl_trans1.shape[2])

    return xwl_sg, exwl_sg


def admit_corr(wl_trans1, wl_trans2):
    """
    Calculates the admittance, coherency and coherence between two real-valued grids 
    given their individual continuous wavelet transform. Averaging is done over all azimuths.

    Args:
        wl_trans1 (np.ndarray): Wavelet transform (shape ``(nx, ny, na, ns)``)
        wl_trans2 (np.ndarray): Wavelet transform (shape ``(nx, ny, na, ns)``)

    Return:
        (tuple): tuple containing:
            * admit (np.ndarray): Wavelet admittance (shape ``(nx, ny, ns)``)
            * eadmit (np.ndarray): Admittance uncertainty (shape ``(nx, ny, ns)``)
            * corr (np.ndarray): Wavelet coherency (shape ``(nx, ny, ns)``)
            * ecorr (np.ndarray): Coherency uncertainty (shape ``(nx, ny, ns)``)
            * coh (np.ndarray): Wavelet coherence (shape ``(nx, ny, ns)``)
            * ecoh (np.ndarray): Coherence uncertainty (shape ``(nx, ny, ns)``)

    """

    wl_sg1, ewl_sg1 = scalogram(wl_trans1)
    wl_sg2, ewl_sg2 = scalogram(wl_trans2)
    xwl_sg, exwl_sg = xscalogram(wl_trans1, wl_trans2)

    admit = xwl_sg/wl_sg1
    corr = xwl_sg/np.sqrt(wl_sg1*wl_sg2)
    coh = np.abs(corr*np.conj(corr))

    var_sg1 = (ewl_sg1/wl_sg1)**2
    var_sg2 = (ewl_sg1/wl_sg1)**2
    var_xsg = (exwl_sg/np.real(xwl_sg))**2

    eadmit = np.sqrt(var_xsg + var_sg1)*np.real(admit)
    ecorr = np.sqrt(var_xsg + var_sg1)*np.real(corr)
    ecoh = np.sqrt(var_xsg + var_sg1 + var_sg2)*coh

    return admit, eadmit, corr, ecorr, coh, ecoh
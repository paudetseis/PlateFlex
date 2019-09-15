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
This ``PlateFlex`` module contains the following functions for setting up pymc models:

- ``plateflex.estimate.set_model``
- ``plateflex.estimate.get_estimates``

Internal functions are available to define predicted admittance and coherence data
with ``theano`` decorators to be incorporated as pymc variables. These functions are
used within :class:`~plateflex.classes.Project` methods as with ``plateflex.plotting``
functions.

http://mattpitkin.github.io/samplers-demo/pages/pymc3-blackbox-likelihood/

"""

# -*- coding: utf-8 -*-
import numpy as np
import pymc3 as pm
from plateflex.flex import flex
from theano.compile.ops import as_op
import theano.tensor as tt

def set_model(k, adm, eadm, coh, ecoh, alph=False, atype='joint'):
    """
    Function to set up a ``pymc3`` model using default bounds on the prior
    distribution of parameters and observed data. Can incorporate 2 ('Te' and 'F')
    or 3 ('Te', 'F', and 'alpha') stochastic variables, and considers
    either the admittance, coherence, or joint admittance and coherence data
    to fit.

    :type k: :class:`~numpy.ndarray`
    :param k: 1D array of wavenumbers
    :type adm: :class:`~numpy.ndarray`
    :param adm: 1D array of wavelet admittance
    :type eadm: :class:`~numpy.ndarray`
    :param eadm: 1D array of error on wavelet admittance
    :type coh: :class:`~numpy.ndarray`
    :param coh: 1D array of wavelet coherence
    :type ecoh: :class:`~numpy.ndarray`
    :param ecoh: 1D array of error on wavelet coherence
    :type alph: bool, optional
    :param alph: Whether or not to estimate parameter ``alpha``
    :type atype: str, optional
    :param atype: Whether to use the admittance ('admit'), coherence ('coh') or both ('joint')

    :return: New :class:`~pymc3.model.Model` object to estimate via sampling
    """

    with pm.Model() as model:

        # k is an array - needs to be passed as distribution
        k_obs = pm.Normal('k', mu=k, sigma=1., observed=k)

        # Prior distributions
        Te = pm.Uniform('Te', lower=1., upper=250.)
        F = pm.Uniform('F', lower=0., upper=0.99999)

        if alph:

            # Prior distribution of `alpha`
            alpha = pm.Uniform('alpha', lower=0., upper=np.pi)
            admit_exp, coh_exp = real_xspec_functions_alpha(k_obs, Te, F, alpha)

        else:
            admit_exp, coh_exp = real_xspec_functions_noalpha(k_obs, Te, F)

        # Select type of analysis to perform
        if atype=='admit':

            # Uncertainty as observed distribution
            sigma = pm.Normal('sigma', mu=eadm, sigma=1., \
                observed=eadm)

            # Likelihood of observations
            admit_obs = pm.Normal('admit_obs', mu=admit_exp, \
                sigma=sigma, observed=adm)

        elif atype=='coh':

            # Uncertainty as observed distribution
            sigma = pm.Normal('sigma', mu=ecoh, sigma=1., \
                observed=ecoh)

            # Likelihood of observations
            coh_obs = pm.Normal('coh_obs', mu=coh_exp, \
                sigma=sigma, observed=coh)

        elif atype=='joint':

            # Define uncertainty as concatenated arrays
            ejoint = np.array([eadm, ecoh]).flatten()

            # Define array of observations and expected values as concatenated arrays
            joint = np.array([adm, coh]).flatten()
            joint_exp = tt.flatten(tt.concatenate([admit_exp, coh_exp]))

            # Uncertainty as observed distribution
            sigma = pm.Normal('sigma', mu=ejoint, sigma=1., \
                observed=ejoint)

            # Likelihood of observations
            joint_obs = pm.Normal('admit_coh_obs', mu=joint_exp, \
                sigma=sigma, observed=joint)

    return model


def get_estimates(summary, map_estimate):
    """
    Extract useful estimates from the Posterior distributions.

    :type summary: :class:`~pandas.core.frame.DataFrame`
    :param summary: Summary statistics from Posterior distributions
    :type map_estimate: dict
    :param map_estimate: Container for Maximum a Posteriori (MAP) estimates

    :return: 
        (tuple): tuple containing:
            * mean_te (float) : Mean value of elastic thickness from posterior (km)
            * std_te (float)  : Standard deviation of elastic thickness from posterior (km)
            * best_te (float) : Most likely elastic thickness value from posterior (km)
            * mean_F (float)  : Mean value of load ratio from posterior
            * std_F (float)   : Standard deviation of load ratio from posterior
            * best_F (float)  : Most likely load ratio value from posterior

    .. rubric:: Example

    >>> from plateflex import estimate
    >>> # MAKE THIS FUNCTION FASTER

    """

    mean_a = None

    # Go through all estimates
    for index, row in summary.iterrows():
        if index=='Te':
            mean_te = row['mean']
            std_te = row['sd']
            C2_5_te = row['hpd_2.5']
            C97_5_te = row['hpd_97.5']
            best_te = np.float(map_estimate['Te'])
        elif index=='F':
            mean_F = row['mean']
            std_F = row['sd']
            C2_5_F = row['hpd_2.5']
            C97_5_F = row['hpd_97.5']
            best_F = np.float(map_estimate['F'])
        elif index=='alpha':
            mean_a = row['mean']
            std_a = row['sd']
            C2_5_a = row['hpd_2.5']
            C97_5_a = row['hpd_97.5']
            best_a = np.float(map_estimate['alpha'])

    if mean_a is not None:
        return mean_te, std_te, C2_5_te, C97_5_te, best_te, \
            mean_F, std_F, C2_5_F, C97_5_F, best_F, \
            mean_a, std_a, C2_5_a, C97_5_a, best_a
    else:
        return mean_te, std_te, C2_5_te, C97_5_te, best_te, \
            mean_F, std_F, C2_5_F, C97_5_F, best_F


def real_xspec_functions(k, Te, F, alpha=np.pi/2., wd=0.):
    """
    Calculate analytical expressions for the real component of admittance, 
    coherency and coherence functions. 

    :type k: np.ndarray
    :param k: Wavenumbers (rad/m)
    :type Te: float
    :param Te: Effective elastic thickness (km)
    :type F: float
    :param F: Subruface-to-surface load ratio [0, 1[
    :type alpha: float, optional
    :param alpha: Phase difference between initial applied loads (rad)
    :type wd: float, optional
    :param wd: Thickness of water layer (km)

    :return:  
        (tuple): tuple containing:
            * admittance (:class:`~numpy.ndarray`): Real admittance function (shape ``len(k)``)
            * coherence (:class:`~numpy.ndarray`): Coherence functions (shape ``len(k)``)

    """

    admittance, coherence = flex.real_xspec_functions(k, Te, F, alpha, wd)

    return admittance, coherence


@as_op(itypes=[tt.dvector, tt.dscalar, tt.dscalar], 
    otypes=[tt.dvector, tt.dvector])
def real_xspec_functions_noalpha(k, Te, F):
    """
    Calculate analytical expressions for the real component of admittance, 
    coherency and coherence functions. 
    """

    adm, coh = real_xspec_functions(k, Te, F)

    return adm, coh

@as_op(itypes=[tt.dvector, tt.dscalar, tt.dscalar, tt.dscalar], 
    otypes=[tt.dvector, tt.dvector])
def real_xspec_functions_alpha(k, Te, F, alpha):
    """
    Calculate analytical expressions for the real component of admittance, 
    coherency and coherence functions. 
    """

    adm, coh = real_xspec_functions(k, Te, F, alpha)

    return adm, coh



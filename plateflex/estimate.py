# Copyright 2019 Pascal Audet

# This file is part of PlateFlex.

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

'''

Functions to estimate effective elastic thickness from spectral quantities.

'''

import numpy as np
import pymc3 as pm
from plateflex import flexure
from plateflex import conf as cf
from plateflex import plotting
from theano.compile.ops import as_op
import theano.tensor as tt
import seaborn as sns
sns.set()


class Estimate(object):

    def __init__(self, k, adm, eadm, coh, ecoh, alph=False, atype='joint'):
        self.k = k
        self.adm = adm
        self.eadm = eadm
        self.coh = coh
        self.ecoh = ecoh
        if not isinstance(alph,bool):
            raise(Exception("'alph' should be a boolean"))
        self.alph = alph
        if not atype in ['admit', 'coh', 'joint']:
            raise(Exception("'atype' should be one among: 'admit', 'coh', or 'joint'"))
        self.atype = atype

    def bayes_real_estimate(self):
        """
        Function that runs ``pymc3`` to estimate the effective elastic thickness,
        load ratio and spectal 'noise' from real-valued spectral functions.

        Args:
            k (np.ndarray)      : Wavenumbers (rad/m)
            adm (np.ndarray)    : Admittance function (mGal/m)
            coh (np.ndarray)    : Coherence functions 
            alph (bool)         : Is alpha a parameter to estimate?
            typ (bool)          : Type of analysis to perform ('admit', 'coh', 'joint')


        Returns:
            (tuple): tuple containing:
                * trace (pymc3.backends.base.MultiTrace): Posterior samples from the MCMC chains
                * map_estimate (dict): Container for Maximum a Posteriori (MAP) estimates
                * summary (pandas.core.frame.DataFrame): Summary of Posterior distributions

        Note:
            Uses ``plateflex.estimate.real_xspec_functions``.

        Example
        -------
        
        >>> import plateflex.estimate as est
        >>> import plateflex.flexure as flex
        >>> import numpy as np
        >>> # First define a fake data set or 300 points with sampling of 20 km 
        >>> x = np.linspace(0.,20.e3*300,300)
        >>> # Get wavenumbers
        >>> k = np.fft.fftfreq(300, 20.e3)
        >>> # Calculate error-free analytical spectral functions
        >>> Te = 40.
        >>> F = 0.5
        >>> alpha = 90. # For real functions this is the correct value
        >>> admit, corr, coh = flex.real_xspec_functions(k, Te, F, alpha)
        >>> # Add artificial noise - this is obviously incorrect as admittance and coherence
        ... # noise are not gaussian
        >>> admit += 0.005*np.random.randn(len(k))
        >>> coh += 0.1*np.random.randn(len(k))
        >>> # Estimate Te and F from the joint inversion of admittance and coherence
        >>> trace, map_estimate, summary = est.bayes_real_estimate(k, admit, corr, coh, typ='admit_coh')
        Auto-assigning NUTS sampler...
        Initializing NUTS using jitter+adapt_diag...
        Initializing NUTS failed. Falling back to elementwise auto-assignment.
        Multiprocess sampling (8 chains in 8 jobs)
        CompoundStep
        >CompoundStep
        >>Slice: [F]
        >>Slice: [Te]
        >NUTS: [sigma]
        Sampling 8 chains: 100%|█████████████| 16000/16000 [00:15<00:00, 1047.24draws/s]
        The estimated number of effective samples is smaller than 200 for some parameters.
        Warning: gradient not available.(E.g. vars contains discrete variables). MAP estimates may not be accurate for the default parameters. Defaulting to non-gradient minimization 'Powell'.
        logp = 468.38: 100%|████████████████████████| 274/274 [00:00<00:00, 2883.91it/s]
        >>> summary
                mean    sd  mc_error  hpd_2.5  hpd_97.5    n_eff  Rhat
        Te     39.14  1.07      0.06    37.02     41.22   171.02  1.03
        F       0.49  0.02      0.00     0.45      0.54   171.31  1.03
        sigma   0.07  0.00      0.00     0.07      0.07  3338.10  1.00

        """

        with pm.Model() as admit_model:

            # k is an array - needs to be passed as distribution
            k_obs = pm.Normal('k', mu=self.k, sigma=1., observed=self.k)

            # Prior distributions
            Te = pm.Uniform('Te', lower=2., upper=250.)
            F = pm.Uniform('F', lower=0., upper=0.99999)

            # Select whether to include alpha as a parameter to estimate
            if self.alph:

                # Prior distribution of `alpha`
                alpha = pm.Uniform('alpha', lower=0., upper=np.pi)
                admit_exp, coh_exp = real_xspec_functions_alpha(k_obs, Te, F, alpha)

            else:
                
                admit_exp, coh_exp = real_xspec_functions_noalpha(k_obs, Te, F)

            # Select type of analysis to perform
            if self.atype=='admit':

                # Uncertainty as observed distribution
                sigma = pm.Normal('sigma', mu=self.eadm, sigma=1., \
                    observed=self.eadm)

                # Likelihood of observations
                admit_obs = pm.Normal('admit_obs', mu=admit_exp, \
                    sigma=sigma, observed=self.adm)

            elif self.atype=='coh':

                # Uncertainty as observed distribution
                sigma = pm.Normal('sigma', mu=self.ecoh, sigma=1., \
                    observed=self.ecoh)

                # Likelihood of observations
                coh_obs = pm.Normal('coh_obs', mu=coh_exp, \
                    sigma=sigma, observed=self.coh)

            elif self.atype=='joint':

                # Define uncertainty as concatenated arrays
                ejoint = np.array([self.eadm, self.ecoh]).flatten()

                # Uncertainty as observed distribution
                sigma = pm.Normal('sigma', mu=ejoint, sigma=1., observed=ejoint)

                # Define array of observations and expected values as concatenated arrays
                joint = np.array([self.adm, self.coh]).flatten()
                joint_exp = tt.flatten(tt.concatenate([admit_exp, coh_exp]))

                # Likelihood of observations
                joint_obs = pm.Normal('admit_coh_obs', mu=joint_exp, sigma=sigma, 
                    observed=joint)

            # Sample the Posterior distribution
            trace = pm.sample(cf.samples, tune=cf.tunes, cores=cf.cores)

            # Get Max a porteriori estimate
            map_estimate = pm.find_MAP()

            # Get Summary
            summary = pm.summary(trace).round(2)

            self.trace = trace
            self.map_estimate = map_estimate
            self.summary = summary

        return


    def plot_stats(self, title=None):

        try:
            plotting.plot_trace_stats(self.trace, self.summary, self.map_estimate, title=title)
        except:
            print('No estimate yet available - analyzing...')
            self.bayes_real_estimate()
            plotting.plot_trace_stats(self.trace, self.summary, self.map_estimate, title=title)


    def plot_fitted(self, est='MAP', title=None):

        if not in ['mean', 'MAP']:
            raise(Exception("Choose one among: 'mean', or 'MAP'"))
            
        try:
            plotting.plot_fitted(self.k, self.adm, self.eadm, self.coh, self.ecoh, self.summary, \
                self.map_estimate, est=est, title=title)
        except:
            print('No estimate yet available - analyzing...')
            self.bayes_real_estimate()
            plotting.plot_fitted(self.k, self.adm, self.eadm, self.coh, self.ecoh, self.summary, \
                self.map_estimate, est=est, title=title)


@as_op(itypes=[tt.dvector, tt.dscalar, tt.dscalar], 
    otypes=[tt.dvector, tt.dvector])
def real_xspec_functions_noalpha(k, Te, F):
    """
    Calculate analytical expressions for the real component of admittance, 
    coherency and coherence functions. 

    Args:
        k (np.ndarray)  : Wavenumbers (rad/m)
        Te (float)      : Effective elastic thickness (km)
        F (float)       : Subruface-to-surface load ratio [0, 1[

    Returns:
        (tuple): tuple containing:
            * adm (np.ndarray)    : Real admittance function (shape ``len(k)``)
            * coh (np.ndarray)      : Coherence functions (shape ``len(k)``)

    Note:
        This function has a ``theano.compile.ops.as_op`` decorator, which
        enables its use as ``pymc3`` variable.

    """

    # Get spectral functions
    adm, coh = flexure.real_xspec_functions(k, Te, F)

    return adm, coh


@as_op(itypes=[tt.dvector, tt.dscalar, tt.dscalar, tt.dscalar], 
    otypes=[tt.dvector, tt.dvector])
def real_xspec_functions_alpha(k, Te, F, alpha):
    """
    Calculate analytical expressions for the real component of admittance, 
    coherency and coherence functions. 

    Args:
        k (np.ndarray)  : Wavenumbers (rad/m)
        Te (float)      : Effective elastic thickness (km)
        F (float)       : Subruface-to-surface load ratio [0, 1[
        alpha (float)   : Phase difference between initial applied loads (deg)

    Returns:
        (tuple): tuple containing:
            * adm (np.ndarray)    : Real admittance function (shape ``len(k)``)
            * coh (np.ndarray)      : Coherence functions (shape ``len(k)``)

    Note:
        This function has a ``theano.compile.ops.as_op`` decorator, which
        enables its use as ``pymc3`` variable.

    """

    # Get spectral functions
    adm, coh = flexure.real_xspec_functions(k, Te, F, alpha)

    return adm, coh


def get_estimates(summary, map_estimate):
    """
    Extract useful estimates from the Posterior distributions.

    Args:
        summary (pandas.core.frame.DataFrame): Summary of Posterior distributions
        map_estimate (dict): Container for Maximum a Posteriori (MAP) estimates

    Return:
        (tuple): tuple containing:
            * mean_te (float) : Mean value of elastic thickness from posterior (km)
            * std_te (float)  : Standard deviation of elastic thickness from posterior (km)
            * best_te (float) : Most likely elastic thickness value from posterior (km)
            * mean_F (float)  : Mean value of load ratio from posterior
            * std_F (float)   : Standard deviation of load ratio from posterior
            * best_F (float)  : Most likely load ratio value from posterior

    Example
    -------
    >>> # Checkout API for ``plateflex.estimate.bayes_real_estimate``
    >>> import plateflex.estimate as est
    >>> est.get_Te_F(map_estimate, summary)
    (39.14, 1.07, 39.25988186, 0.49, 0.02, 0.49426403)

    """

    mean_a = None

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

    if mean_a:
        return mean_te, std_te, C2_5_te, C97_5_te, best_te, \
            mean_F, std_F, C2_5_F, C97_5_F, best_F, \
            mean_a, std_a, C2_5_a, C97_5_a, best_a
    else:
        return mean_te, std_te, C2_5_te, C97_5_te, best_te, \
            mean_F, std_F, C2_5_F, C97_5_F, best_F


# def estimate_parallel(estimate, x, y):
#     """
#     Run pymc models in parallel from a loop through the x,y coordinates.

#     Note:
#         This implementation is faulty at the moment. Will freeze the CPU and 
#         prevent completion of the loop.

#     Args:
#         k (np.ndarray)      : Wavenumbers (rad/m)
#         admit (np.ndarray)  : Wavelet admittance (shape ``(nx, ny, ns)``)
#         eadmit (np.ndarray) : Admittance uncertainty (shape ``(nx, ny, ns)``)
#         corr (np.ndarray)   : Wavelet coherency (shape ``(nx, ny, ns)``)
#         ecorr (np.ndarray)  : Coherency uncertainty (shape ``(nx, ny, ns)``)
#         coh (np.ndarray)    : Wavelet coherence (shape ``(nx, ny, ns)``)
#         ecoh (np.ndarray)   : Coherence uncertainty (shape ``(nx, ny, ns)``)
#         x (int)             : x-index of grid
#         y (int)             : y-index of grid

#     Returns:
#         (tuple): results: tuple containing the mean, std, 95% CI and MAP estimates.

#     """

#     # Extract 1-D spectral functions and errors at single grid location
#     adm = admit[x,y,:]
#     coh = cohere[x,y,:]
#     eadm = eadmit[x,y,:]
#     ecoh = ecohere[x,y,:]

#     # Run the pymc3 model
#     trace, map_estimate, summary = \
#         pf.estimate.bayes_real_estimate(\
#             k, adm, eadm, coh, ecoh, alph=True, typ='joint')

#     # Store the estimates into a tuple
#     results = pf.estimate.get_estimates(summary, map_estimate)

#     return results



import numpy as np
import pymc3 as pm
import plateflex.flexure as flex
import plateflex.conf as cf
from theano.compile.ops import as_op
import theano.tensor as tt


@as_op(itypes=[tt.dvector, tt.dscalar, tt.dscalar, tt.dscalar], 
    otypes=[tt.dvector, tt.dvector, tt.dvector])
def real_xspec_functions(k, Te, F, alpha):
    Te = Te*1.e3
    D = cf.E*Te**3/12./(1.-cf.nu**2.)
    psi = D*k**4.
    alpha = alpha*np.pi/180.
    theta = flex.flexfilter1D(psi, 0., 0., 'top')
    phi = flex.flexfilter1D(psi, 0., 0., 'bot')
    mu_h, mu_w, nu_h, nu_w = flex.decon1D(theta, phi, k)
    admit, corr, coh = flex.tr_func(mu_h, mu_w, nu_h, nu_w, F, alpha)
    admit = np.real(admit)
    corr = np.real(corr)

    return admit, corr, coh


def bayes_real_estimate(k, adm, cor, coh, typ='real_admit'):

    with pm.Model() as admit_model:

        # Prior distributions
        Te = pm.Uniform('Te', lower=5., upper=250.)
        F = pm.Uniform('F', lower=0., upper=0.99999)
        alpha = pm.Normal('alpha', 90., observed=90.)
        sigma = pm.HalfNormal('sigma', sigma=1.)

        # k is a fixed observed array - needs to be passed as distribution
        k_obs = pm.Normal('k', mu=k, sigma=1., observed=k)

        admit_exp, corr_exp, coh_exp = real_xspec_functions(k_obs, Te, F, alpha)

        if typ=='real_admit':

            # Likelihood of observations
            admit_obs = pm.Normal('admit_obs', mu=admit_exp, sigma=sigma, 
                observed=adm)

        elif typ=='real_corr':

            # Likelihood of observations
            corr_obs = pm.Normal('corr_obs', mu=corr_exp, sigma=sigma, 
                observed=cor)

        elif typ=='coh':

            # Likelihood of observations
            coh_obs = pm.Normal('coh_obs', mu=coh_exp, sigma=sigma, 
                observed=coh)

        elif typ=='admit_coh':

            admit_coh = np.array([adm, coh]).flatten()
            admit_coh_exp = tt.flatten(tt.concatenate([admit_exp, coh_exp]))

            # Likelihood of observations
            admit_coh_obs = pm.Normal('admit_coh_obs', mu=admit_coh_exp, sigma=sigma, 
                observed=admit_coh)

        # Sample the Posterior distribution
        trace = pm.sample(cf.samples, tune=cf.tunes, cores=4)

        # Get Max a porteriori estimate
        map_estimate = pm.find_MAP()

        # Get Summary
        summary = pm.summary(trace).round(2)

    return trace, map_estimate, summary


def get_values(map_estimate, summary):

    mean_te = summary.values[0][0]
    std_te = summary.values[0][1]
    best_te = map_estimate['Te']

    mean_F = summary.values[1][0]
    std_F = summary.values[1][1]
    best_F = map_estimate['F']

    return mean_te, std_te, best_te, mean_F, std_F, best_F

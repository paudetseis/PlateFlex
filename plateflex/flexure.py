import numpy as np
import plateflex.conf as cf

def flexfilter1D(psi, zeta, sigma, typ):
    if typ=='top':
        return -(cf.rhoc/cf.drho)*(1. + psi/cf.drho/cf.g + zeta/cf.drho/cf.g + 
            sigma/cf.drho/cf.g)**(-1.)
    elif typ=='bot':
        return -(cf.rhoc/cf.drho)*(1. + psi/cf.rhoc/cf.g + zeta/cf.rhoc/cf.g + 
            sigma/cf.rhoc/cf.g)


def decon1D(theta, phi, k):
    
    mu_h = 1./(1.-theta)
    mu_w = 1./(phi-1.)
    nu_h = 2.*np.pi*cf.G*(cf.drho*theta*np.exp(-k*cf.zc))
    nu_h = nu_h/(1.-theta)
    nu_w = 2.*np.pi*cf.G*(cf.drho*phi*np.exp(-k*cf.zc))
    nu_w = nu_w/(phi-1.)

    return mu_h, mu_w, nu_h, nu_w


def tr_func(mu_h, mu_w, nu_h, nu_w, F, alpha):
    
    r = cf.rhoc/cf.drho
    f = F/(1. - F)
    hg = nu_h*mu_h + nu_w*mu_w*(f**2)*(r**2) + (nu_h*mu_w + nu_w*mu_h)*f*r*np.cos(alpha) + \
        1j*(nu_h*mu_w - nu_w*mu_h)*f*r*np.sin(alpha)
    hh = mu_h**2 + (mu_w*f*r)**2 + 2.*mu_h*mu_w*f*r*np.cos(alpha)
    gg = nu_h**2 + (nu_w*f*r)**2 + 2.*nu_h*nu_w*f*r*np.cos(alpha)
    admit = hg/hh
    corr = hg/np.sqrt(hh)/np.sqrt(gg)
    coh = np.real(corr)**2
    
    return admit, corr, coh


def real_xspec_functions(k, Te, F, alpha):
    Te = Te*1.e3
    D = cf.E*Te**3/12./(1.-cf.nu**2.)
    psi = D*k**4.
    alpha = alpha*np.pi/180.
    theta = flexfilter1D(psi, 0., 0., 'top')
    phi = flexfilter1D(psi, 0., 0., 'bot')
    mu_h, mu_w, nu_h, nu_w = decon1D(theta, phi, k)
    admit, corr, coh = tr_func(mu_h, mu_w, nu_h, nu_w, F, alpha)
    admit = np.real(admit)
    corr = np.real(corr)

    return admit, corr, coh

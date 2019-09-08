import numpy as np

def scalogram(wl_trans):

    wl_sg = np.mean(np.abs(wl_trans*np.conj(wl_trans)),axis=(2))
    ewl_sg = np.std(np.abs(wl_trans*np.conj(wl_trans)),axis=(2),ddof=1)/np.sqrt(wl_trans.shape[2])

    return wl_sg, ewl_sg


def xscalogram(wl_trans1, wl_trans2):

    xwl_sg = np.mean(wl_trans1*np.conj(wl_trans2),axis=(2))
    exwl_sg = np.std(np.abs(wl_trans1*np.conj(wl_trans2)),axis=(2),ddof=1)/np.sqrt(wl_trans1.shape[2])

    return xwl_sg, exwl_sg


def admit_corr(wl_trans1, wl_trans2):

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
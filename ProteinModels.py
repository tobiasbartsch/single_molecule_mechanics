import numpy as np

def xWLCe(F, lp, lc, K):
    '''Extension-force relation for an extensible WLC, 
    Petrosyan, R. (2016). "Improved approximations for some polymer extension models". Rehol Acta. 56: 21–26. arXiv:1606.02519. doi:10.1007/s00397-016-0977-9.

    Args:
        lp (float): persistence length
        lc (float): contour length
        K (float): enthalpic elastic modulus
        F (float): force
    '''

    kBT = 4.0591066488e-21 #thermal energy at 70 Fahrenheit

    x = lc * (4/3 - 4/(3 * np.sqrt(F*lp/kBT + 1)) - 10 * np.exp((900*kBT/(F*lp))**(1/4)) / (np.sqrt(F*lp/kBT) * ( np.exp((900*kBT/(F*lp))**(1/4)) - 1)**2 ) + (F*lp/kBT)**1.62 / (3.55 + 3.8 * (F*lp/kBT)**2.2) + F/K)
    return x

def xSeriesWLCe(F, lps, lcs, Ks):
    '''Extension-force relation for a series of WLCs,
    Petrosyan, R. (2016). "Improved approximations for some polymer extension models". Rehol Acta. 56: 21–26. arXiv:1606.02519. doi:10.1007/s00397-016-0977-9.

    Args:
        lps (list of floats): persistence lengths
        lcs (list of floats): contour lengths
        Ks (list of floats): enthalpic elastic moduli
        F (float): force
    '''

    x = 0
    for (lp, lc, K) in zip(lps, lcs, Ks):
        x += xWLCe(F, lp, lc, K)
    
    return x
    

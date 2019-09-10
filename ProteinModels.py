import numpy as np

def xFJC(F, b, N):
    '''Extension-force relation for an inextensible FJC, 
    from lecture notes LMU

    Args:
        F (float): force
        b (float): length of one stiff segment
        N (int): number of segments
        
    '''
    kBT = 4.072913134e-21
	
    x = N*b * (1/np.tanh(F*b/kBT)-kBT/(F*b))	
    return x


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
    x[F == 0] = 0
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

def x_WLCes_FJC(F, lps, lcs, Ks, b, N):
    '''Extension-force relation for a series of WLCes, itself in series with a FJC

    Args:
        F (float): force
        lps (list of floats): persistence lengths
        lcs (list of floats): contour lengths
        Ks (list of floats): enthalpic elastic moduli
        b (float): length of rigid segments in FJC
        N (int): number of stiff segments in FJC
    '''	
    x = 0
    x += xSeriesWLCe(F, lps, lcs, Ks)
    x += xFJC(F, b, N)
	
    return x



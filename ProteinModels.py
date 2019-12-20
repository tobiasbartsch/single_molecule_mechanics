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
        lc (float or np.array of floats): contour length (if an array is passed, the length of this array must be the same as dim 0 of the force matrix)
        K (float): enthalpic elastic modulus
        F (float or np.array of floats): force (if an array is passed, this function attempts to compute an extension for each force)
    '''
    
    kBT = 4.0591066488e-21 #thermal energy at 70 Fahrenheit
    x = lc * (4/3 - 4/(3 * np.sqrt(F*lp/kBT + 1)) - 10 * np.exp((900*kBT/(F*lp))**(1/4)) / (np.sqrt(F*lp/kBT) * ( np.exp((900*kBT/(F*lp))**(1/4)) - 1)**2 ) + (F*lp/kBT)**1.62 / (3.55 + 3.8 * (F*lp/kBT)**2.2) + F/K)
    #x = np.nan_to_num(x)
    return x



    # if(np.asarray([lc]).shape[0] == 1 and np.asarray([lp]).shape[0] == 1): 
    #     #we were given one lc and one lp; just compute the extension for each entry in F
    #     x = lc * (4/3 - 4/(3 * np.sqrt(F*lp/kBT + 1)) - 10 * np.exp((900*kBT/(F*lp))**(1/4)) / (np.sqrt(F*lp/kBT) * ( np.exp((900*kBT/(F*lp))**(1/4)) - 1)**2 ) + (F*lp/kBT)**1.62 / (3.55 + 3.8 * (F*lp/kBT)**2.2) + F/K)
    #     x = np.nan_to_num(x)
    #     return x
    # elif(np.asarray([lc]).shape[0] > 1 and np.asarray([F]).shape[0] > np.asarray([lc]).shape[0] and np.asarray([lp]) == 1): 
    #     #we were given several lcs, one for each row in F. Apply row wise.
    #     x = []
    #     for F_i, lc_i in zip(F,lc):
    #         x_i = lc_i * (4/3 - 4/(3 * np.sqrt(F_i*lp/kBT + 1)) - 10 * np.exp((900*kBT/(F_i*lp))**(1/4)) / (np.sqrt(F_i*lp/kBT) * ( np.exp((900*kBT/(F_i*lp))**(1/4)) - 1)**2 ) + (F_i*lp/kBT)**1.62 / (3.55 + 3.8 * (F_i*lp/kBT)**2.2) + F_i/K)
    #         x.append(x_i)
    #     x = np.nan_to_num(x)
    #     return np.array(x)
    # else:
    #     raise NotImplementedError('Unsupported combination of dimensions in F, lc, and lp.')

    #x = np.einsum('i,jk->ijk', lc, x_i) #stack different lcs along the first axis.



    # if(type(lc) is not np.ndarray and type(lp) is not np.ndarray): 
    #     #we were given one lc and one lp; just compute the extension for each entry in F
    #     x = lc * (4/3 - 4/(3 * np.sqrt(F*lp/kBT + 1)) - 10 * np.exp((900*kBT/(F*lp))**(1/4)) / (np.sqrt(F*lp/kBT) * ( np.exp((900*kBT/(F*lp))**(1/4)) - 1)**2 ) + (F*lp/kBT)**1.62 / (3.55 + 3.8 * (F*lp/kBT)**2.2) + F/K)
    #     x = np.nan_to_num(x)
    #     return x
    # # elif(len(lc) > 1 and F.shape[0] == len(lc) and type(lp) is not np.ndarray): 
    # #     #we were given several lcs, one for each row in F. Apply row wise.
    # #     x = []
    # #     for F_i, lc_i in zip(F,lc):
    # #         x_i = lc_i * (4/3 - 4/(3 * np.sqrt(F_i*lp/kBT + 1)) - 10 * np.exp((900*kBT/(F_i*lp))**(1/4)) / (np.sqrt(F_i*lp/kBT) * ( np.exp((900*kBT/(F_i*lp))**(1/4)) - 1)**2 ) + (F_i*lp/kBT)**1.62 / (3.55 + 3.8 * (F_i*lp/kBT)**2.2) + F_i/K)
    # #         np.array(x_i)[np.array(F_i) == 0] = 0
    # #         x.append(x_i)
    # #     x = np.nan_to_num(x)
    # #     return np.array(x)
    # elif(len(lc) > 1 and type(lp) is not np.ndarray): 
    #     #we were given several lcs, return a 3D array where the first dimension iterates over lc.
    #     x = []
    #     for lc_i in lc:
    #         x_i = lc_i * (4/3 - 4/(3 * np.sqrt(F*lp/kBT + 1)) - 10 * np.exp((900*kBT/(F*lp))**(1/4)) / (np.sqrt(F*lp/kBT) * ( np.exp((900*kBT/(F*lp))**(1/4)) - 1)**2 ) + (F*lp/kBT)**1.62 / (3.55 + 3.8 * (F*lp/kBT)**2.2) + F/K)
    #         np.array(x_i)[np.array(F) == 0] = 0
    #         x.append(x_i)
    #     x = np.nan_to_num(x)
    #     return np.array(x)
    # else:
    #     raise NotImplementedError('Unsupported combination of dimensions in F, lc, and lp.')

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



import pymc3 as pm
import numpy as np
from single_molecule_mechanics.ProteinModels import xWLCe



def fit_WLCe(segmented_data, num_lcs, lc_low = 50, lc_high=120, lc_sdev = 10, samples=500, tune=500, chains=4, cores=4):
    '''fit the data with extensible worm-like chains. Supports the fitting of different segments of the data with different 
    contour lengths (lcs) -- this is useful for data that features unfolding events.
    If unfolding events exist you should indicate the number of states the protein can exist in using 'num_lcs'. If you do NOT know
    how many states there are, you should run this fitter with a variety of num_lcs, and use MDL to determine the most useful model.

    Args:
      segmented_data (pd.DataFrame): dataframe containing a column 'force' (in pN), 'extension' (in nm),
                                     and 'segments' (categorical; if data point i belongs into segment j, then set this field to j).
      num_lcs (int): number of different lcs to fit to the data
      lc_low (int): lower boundary for contour lengths (nm)
      lc_high (int): upper boundary for contour lengths (nm)
      lc_sdev (int): prior for the standard deviation for the uncertainty of each contour length
      samples (int): number of samples to draw from the posterior for each chain
      tune (int): number of tuning steps for each chain
      chains (int): number of simultaneously sampling chains
      cores (int): number of CPU cores to distribute the chains over
    
    Returns:
      (model, trace): model and MCMC trace
    '''

    num_segments = len(segmented_data['segment'].unique())

    with pm.Model() as model:
        
        mus = np.linspace(lc_low, lc_high, num_lcs)
        lc_cats = pm.Normal('lc_cats', mu=mus, sd = lc_sdev, shape=num_lcs) #lc categories that all lcs must be sorted into.
        ps = np.repeat(1/num_lcs, num_lcs)
        cat = pm.Categorical('cat', p=ps, shape = num_segments) #each segment has a different lc category
        
        lcs_mean = pm.Normal('lc_mean', mu=lc_cats[cat], sd=4, shape=num_segments) #one lc per segment
        lcs_sigma = pm.HalfNormal('lc_sdev', sd=2, shape=num_segments)
        
        lcs = pm.Normal('lcs', mu=lcs_mean, sd = lcs_sigma, shape=num_segments)
        
        lp_mean = pm.Normal('lp_mean', mu=1, sd=0.1) #assume for now that lp is the same for all segments. We can live with shape=1 for now.
        lp_sigma = pm.HalfNormal('lp_sdev', sd=0.2)
        lp = pm.Normal('lp', mu=lp_mean, sd=lp_sigma)
        
        x_mean = xWLCe(np.array(segmented_data['force']), lp*1e-9, lcs[np.array(segmented_data['segment'])]*1e-9, 10e-3)*1e9

        sig = pm.Normal('sig', mu=1.5, sd=0.2)

        xi = pm.Normal('xi', mu=x_mean, sd=sig, observed=np.array(segmented_data['extension'])*1e9)
        
        trace = pm.sample(500, tune=500, chains=4, cores=4)
    
    return (model, trace) 
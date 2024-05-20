import numpy as np
import torch
from torch.distributions.normal import Normal as N

def crps_gaussian(mu, sigma, x):
    '''Function to compute the Continuous Ranked Probability Score (CRPS) of an observed value, using a Gaussian
    distribution with mean mu and variance sigma^2 as a reference. 
    
    Parameters
    ----------
    mu : float
        The mean of the reference Gaussian distribution
    sigma : float
        The standard deviation of the reference Gaussian distribution
    x : float
        The observed value to score
    
    Returns
    -------
    crps : torch.tensor
        The CRPS of the observed value
    '''
    
    # Convert float values to tensors for use in PyTorch functions
    mu, sigma, x = torch.tensor([mu]), torch.tensor([sigma]), torch.tensor([x])
    
    # Compute shorthands of the normalized observation and pi, and create a standard normal distribution
    norm = (x-mu) / sigma
    Z = N(loc=0, scale=1)
    pi = torch.tensor([torch.acos(torch.zeros(1)).item() * 2])
    
    # Compute CRPS
    crps = sigma * (norm*(2*Z.cdf(norm) - 1) + 2*torch.exp(Z.log_prob(norm)) - 1/torch.sqrt(pi))
    return crps 

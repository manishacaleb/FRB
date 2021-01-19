""" Code for the DM of FRB hosts """
from scipy.stats import lognorm

def rand_lognorm(nFRB, lognorm_s = 0.88, lognorm_floor = 0., mu = 68.):

    DM_host = lognorm.rvs(size=nFRB, s=lognorm_s, loc=lognorm_floor, scale=mu)

    return DM_host

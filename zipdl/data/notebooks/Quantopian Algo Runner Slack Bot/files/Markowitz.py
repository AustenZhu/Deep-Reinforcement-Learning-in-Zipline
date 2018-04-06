import numpy as np


def opt_weight(r, cov_mat, mu_vec):
    ones = np.ones(len(mu_vec)).reshape((len(mu_vec), 1))
    A = mu_vec.T.dot(np.linalg.inv(cov_mat)).dot(mu_vec)[0][0]
    B = ones.T.dot(np.linalg.inv(cov_mat)).dot(ones)[0][0]
    C = mu_vec.T.dot(np.linalg.inv(cov_mat)).dot(ones)[0][0]
    
    
    nu = (2*A - 2*r*C) / (A*B - C*C)
    lam = (2*r - nu * C) / A
    w = (1/2) * np.linalg.inv(cov_mat).dot(lam*mu_vec + nu*ones)
    return w

def get_vol(w, cov_mat, annualized=False):
    variance = w.T.dot(cov_mat).dot(w)[0][0]
    if annualized:
        variance = variance*252
    
    volatility = np.sqrt(variance)
    return volatility

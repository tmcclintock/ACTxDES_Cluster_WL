import sys
import numpy as np
from classy import Class
import cluster_toolkit as ct
from cluster_toolkit import concentration as conc
import pickle

def make_LSS_data(z, cosmo_dict, outfile=None):
    """
    Args:
        z (float): redshift
        cosmo_dict (dictionary): the CLASS cosmology
        outfile (string): file to save to
            
    """
    if outfile is None:
        outfile = "LSS_dict"

    LSS_dict = {}
    
    h = cosmo_dict["h"]
    Omega_b = cosmo_dict["Omega_b"]
    Omega_m = cosmo_dict["Omega_cdm"] + cosmo_dict["Omega_b"]
    n_s = cosmo_dict["n_s"]
    cosmo = Class()
    cosmo.set(cosmo_dict)
    cosmo.compute()
    sigma8 = cosmo.sigma8()
    print("sigma8 is:", sigma8)
    
    k = np.logspace(-5, 3, base=10, num=4000) #1/Mpc; comoving
    kh = k / h #h/Mpc; comoving
    r = np.logspace(-2, 3, num=1000) #Mpc/h comoving
    M = np.logspace(12, 16.3, 1000) #Msun/h
    
    z = np.asarray(z)
    for zi in z:
        P_nl = np.array([cosmo.pk(ki, zi) for ki in k])*h**3
        P_lin = np.array([cosmo.pk_lin(ki, zi) for ki in k])*h**3

        xi_nl = ct.xi.xi_mm_at_r(r, kh, P_nl)
        xi_lin = ct.xi.xi_mm_at_r(r, kh, P_lin)

        c = np.array([conc.concentration_at_M(Mi, kh, P_lin, n_s,
                                              Omega_b, Omega_m, h,
                                              Mass_type="mean") for Mi in M])
        bias = ct.bias.bias_at_M(M, kh, P_lin, Omega_m)

        args_at_z = {"k":kh, "P_lin":P_lin, "P_nl":P_nl, "r":r, "xi_lin":xi_lin,
                     "xi_nl":xi_nl, "M":M, "concentration":c, "bias":bias,
                     "h":h, "Omega_m":Omega_m, "n_s":n_s, "sigma8":sigma8,
                     "Omega_b":Omega_b}
        LSS_dict["args_at_{z:.3f}".format(z=zi)] = args_at_z

    #np.save(outfile, LSS_dict)
    pickle.dump(LSS_dict, open(outfile+".p", "wb"))
    print("Saved {0}".format(outfile))
    return

if __name__ == "__main__":
    #Fox simulation cosmology
    fox_cosmo_dict = {"h":0.6704,
                      "A_s":2.14e-9,
                      "Omega_b":0.049,
                      "Omega_cdm":0.269,
                      "n_s":0.962,
                      'P_k_max_h/Mpc':1700.,
                      'z_max_pk':2.0,
                      'output': 'mPk',
                      'non linear':'halofit'}
    fox_zs = [0.0, 0.25, 0.5, 1.0]
    fox_outfile = "LSS_fox_dict"
    make_LSS_data(fox_zs, fox_cosmo_dict, fox_outfile)
    
    #DES Y1/Y3 cosmology
    des_cosmo_dict = {"h":0.7,
                      "sigma8":0.8,
                      "Omega_b":0.05,
                      "Omega_cdm":0.25,
                      "n_s":0.96,
                      'P_k_max_h/Mpc':1700.,
                      'z_max_pk':2.0,
                      'output': 'mPk',
                      'non linear':'halofit'}
    des_zs = [0.5]
    des_outfile = "LSS_DES_dict"
    make_LSS_data(des_zs, des_cosmo_dict, des_outfile)

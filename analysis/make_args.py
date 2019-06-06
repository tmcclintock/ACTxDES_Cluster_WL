"""
Given a filename, read in the data and assemble a dictionary containing
information for the analysis.
"""

import numpy as np
import pickle
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

def get_args(filename):
    """

    Args:
        filename (string): a unique string associated a .npz file
            that contains a dsigma and cov field. All additional
            information about the stack comes from our knowledge
            of what is in the file.

    Returns:
        Dictionary containing the arguments to the modeling function.
    """
    
    if filename == "dsigma_advact_SNRgt5_z0.1-0.9.npz":
        #All clusters with SNR>5 between z\in[0.1, 0.9]
        data = np.load("../data/{0}".format(filename))
        DeltaSigma = data['dsigma']
        Cov = data['cov']

        #Constants
        h = 0.7
        Omega_m = 0.3
        mean_z  = 0.50
        median_z = 0.49 
        mean_xi = 7.77
        median_xi = 6.45

        #For this analysis, use the mean redshift
        z = mean_z

        #Radial bin edges in Mpc/h physical distances
        Redges = np.logspace(np.log10(0.02), np.log10(80.),19)
        Redges_com = (1+mean_z)*Redges #comoving Mpc/h
        Rmid = (Redges[1:] + Redges[:-1])/2

        #Check to make sure the sizes of things are correct
        assert len(Redges)-1 == len(DeltaSigma)
        assert len(Redges)-1 == len(Cov)
        assert len(Redges)-1 == len(Cov[0])

        #Our cut at 200 kpc/h physical
        cut = Rmid > 0.2 #Should be 14 bins, with 5 cut
        DeltaSigma_cut = DeltaSigma[cut]
        Cov_cut = Cov[cut]
        Cov_cut = Cov_cut[:, cut]
        Redges_com_cut = []
        for Re in Redges_com:
            if Re > 0.2:
                Redges_com_cut.append(Re)
        Redges_com_cut = np.asarray(Redges_com_cut)

        #Boost factor stuff -- note, these don't exist yet
        Rb = Rmid[cut]/(1+z) #Convert to physical distances; Mpc/h physical
        B_plus_1 = np.ones_like(DeltaSigma_cut)
        B_cov = np.diag(B_plus_1)

        #Sigma_crit_inv for the reduced shear effect
        #pc^2/hMsun comoving
        Sigma_crit_inv = 0 #for now
        
        #Path to the LSS quantities
        LSS_dict_path = "../data/LSS_files/LSS_DES_dict.p"
    else:
        raise Exception("Analysis not configured.")
    
    #Load in the dictionary for precomputed LSS quantities for
    #this cosmology
    #analysis_dict = np.load(LSS_dict_path)
    analysis_dict = pickle.load(open(LSS_dict_path, "rb"))
    LSS_dict = analysis_dict["args_at_{z:.3f}".format(z=z)]
    r = LSS_dict["r"] #Mpc/h comoving
    xi_mm = LSS_dict["xi_nl"] #Fourier transform of P_nl(k, z)
    M_array = LSS_dict["M"] #Msun/h; M200m
    concs = LSS_dict["concentration"] #same size as M_array
    biases = LSS_dict["bias"] #same size as M_array
    bias_spline = IUS(M_array, biases)
    conc_spline = IUS(M_array, concs)
    Rp = np.logspace(-2, 2.4, 1000, base=10) #Mpc/h comoving; projected distances
        
    #Make the args
    args = {"h":h, "Omega_m":Omega_m, "z":z, "mean_xi":mean_xi, "Redges":Redges,
            "Redges_com":Redges_com, "Rmid":Rmid, "DeltaSigma":DeltaSigma,
            "DeltaSigma_cut":DeltaSigma_cut, "Cov":Cov, "Cov_cut":Cov_cut,
            "r":r, "xi_mm":xi_mm, "M_array":M_array, "concentrations":concs,
            "biases":biases, "Rp":Rp, "Redges_com_cut":Redges_com_cut,
            "Boost_plus_1":B_plus_1, "B_cov":B_cov, "Rb":Rb,
            "b_spline":bias_spline, "c_spline":conc_spline,
            "Sigma_crit_inv":Sigma_crit_inv}
    return args

if __name__ == "__main__":
    fname = "dsigma_advact_SNRgt5_z0.1-0.9.npz"
    print(get_args(fname).keys())

    import model
    print(model.ACTxDES_cluster_lnpost([14, 5, 1], get_args(fname)))

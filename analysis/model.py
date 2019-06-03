import numpy as np
import cluster_toolkit as ct

def ACTxDES_cluster_lnpost(params, args):
    """Log-posterior of the cluster lensing signal.
    
    Note: this model assumes no miscentering is present.

    Masses are Msun/h
    Distances are Mpc/h comoving
    Lensing profiles are hMsun/pc^2 comoving

    """
    lM, c, Am, B0, Rs = params

    #Trivial priors
    if lM < 13 or lM > 16:
        return -1e99
    if c <= 1 or c > 15:
        return -1e99
    if 0.6 >= Am >= 1.6:
        return -1e99
    if B0 < 0 or B0 > 2:
        return -1e99
    if Rs <=0 or  Rs > 10:
        return -1e99

    #Non-trivial priors. Miscentering has been commented out
    LPfmis = 0 #(0.32 - fmis)**2/0.05**2
    LPtau  = 0 #(0.153 - tau)**2/0.03**2
    LPA    = (1.02 - Am)**2/0.0006 #Temporary values
    
    lnlike = -0.5*(LPfmis + LPtau + LPA) #Sum of the priors

    #Step 1: pull out everything that has been assmebled in the arguments dictionary.
    lam = args["richness"]
    z = args["z"]
    h = args["h"]
    Omega_m = args["Omega_m"]
    inds = args["kept_inds"] #cut on the radial bins
    Redges = args["Redges"] #Mpc/h comoving
    r = args['r'] #Mpc/h comoving
    xi_nl = args['xi_nl']
    b_spline = args["b_spline"] #Tinker 2010 bias
    Rp = args['Rp'] #Mpc/h comoving; projected on the sky
    Sigma_crit_inv = args["Sigma_crit_inv"] #pc^2/hMsun comoving

    #Including the data
    #Data is in hMsun/pc^2 comoving
    DS_data = args["DeltaSigma"]
    Cov = args["Cov"]
    
    Bp1 = args['Bp1'] #1 + B
    Bcov = args['Bcov'] #Covariance of B

    M = 10**lM #Msun/h
    xi_1h = ct.xi.xi_nfw_at_r(r, M, c, Omega_m)
    bias = b_spline(M)
    xi_2halo = ct.xi.xi_2halo(bias, xi_nl)
    xi_hm = ct.xi.xi_hm(xi_1h, xi_2halo)
    Sigma = ct.deltasigma.Sigma_at_R(Rp, r, xi_hm, M, c, Omega_m)
    DeltaSigma = ct.deltasigma.DeltaSigma_at_R(Rp, Rp, Sigma, M, c, Omega_m)

    #Rlam = args["Rlam"] #Mpc/h comoving
    #Rmis = tau*Rlam #Mpc/h
    #Sigma_mis  = ct.miscentering.Sigma_mis_at_R(Rp, Rp, Sigma, M, c, 
    #                                            Omega_m, Rmis, kernel="gamma")
    #DeltaSigma_mis = ct.miscentering.DeltaSigma_mis_at_R(Rp, Rp, Sigma_mis)
    
    #Note: here, Rs is Mpc physical but Rp is Mpc/h comoving
    boost = ct.boostfactors.boost_nfw_at_R(Rp, B0, Rs)
    full_Sigma = Sigma
    full_DeltaSigma = DeltaSigma
    #full_Sigma = (1-fmis)*Sigma + fmis*Sigma_mis
    #full_DeltaSigma = (1-fmis)*DeltaSigma + fmis*DeltaSigma_mis
    full_DeltaSigma *= Am #multiplicative bias
    full_DeltaSigma /= boost #boost factor
    full_DeltaSigma /= (1-full_Sigma*Sigma_crit_inv) #reduced shear
    ave_DeltaSigma = ct.averaging.average_profile_in_bins(Redges, Rp, full_DeltaSigma)
    
    
    X = (DS_data - ave_DeltaSigma[inds]*h*(1+z)**2)
    lnlike += -0.5*np.dot(X, np.linalg.solve(Cov, X))
    
    #Note: here, Rs is Mpc physical and Rb is the same
    boost_model = ct.boostfactors.boost_nfw_at_R(args['Rb'], B0, Rs)
    #print("boost model:", boost_model)
    Xb = Bp1 - boost_model
    lnlike += -0.5*np.dot(Xb, np.linalg.solve(Bcov, Xb))  
    return lnlike
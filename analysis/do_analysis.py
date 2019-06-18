"""
An example file for doing the ACTxDES cluster WL analysis.
"""
import numpy as np
import emcee, corner
import scipy.optimize as op

#Local files for running the analysis
import make_args
import model

fname = "dsigma_advact_SNRgt5_z0.1-0.9.npz"
args = make_args.get_args(fname)

print("Working on {0}".format(fname))
print("SNR = {0:.3f}".format(args["SNR"]))

#Model start: log10M, concentration, multiplicative bias
start = np.array([14, 3, args["Am_mean"]])


#Negative log-posterior
def nlp(params, args):
    return -model.ACTxDES_cluster_lnpost(params, args)

#Do a simple optimization
result = op.minimize(nlp, start, args=(args), method="Nelder-Mead")

print(result)

#Plot the best fit
best = result.x
DeltaSigma, ave_DeltaSigma, _ = model.ACTxDES_cluster_lnpost(best, args, True)

import matplotlib.pyplot as plt
plt.rc("text", usetex=True)
plt.rc("font", size=24, family="serif")
plt.rc("errorbar", capsize=3)

Rp = args["Rp_phys"]
Rmid = args["Rmid_cut"]
DS = args["DeltaSigma_cut"]
Cov = args["Cov_cut"]
err = np.sqrt(Cov.diagonal())
plt.errorbar(Rmid, DS, err, color='k', ls='', marker='.')
plt.loglog(Rp, DeltaSigma)

plt.ylabel(r"$\Delta\Sigma\ [h{\rm M_\odot/pc^2}\ {\rm phys}]$")

plt.xlim(0.2, 85)
plt.xlabel(r"$R\ [h^{-1}{\rm Mpc}\ {\rm phys}]$")
plt.savefig("test.png", dpi=300, bbox_inches="tight")
plt.show()

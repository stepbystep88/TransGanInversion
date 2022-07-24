import numpy as np
from scipy.signal import filtfilt

import pylops
from pylops.utils.wavelets import ricker

# plt.close("all")
np.random.seed(0)

# sphinx_gallery_thumbnail_number = 5

# model
nt0 = 301
dt0 = 0.002

t0 = np.arange(nt0) * dt0
vp = 1200 + np.arange(nt0) + filtfilt(np.ones(5) / 5.0, 1, np.random.normal(0, 80, nt0))
vs = 600 + vp / 2 + filtfilt(np.ones(5) / 5.0, 1, np.random.normal(0, 20, nt0))
rho = 1000 + vp + filtfilt(np.ones(5) / 5.0, 1, np.random.normal(0, 30, nt0))
vp[131:] += 500
vs[131:] += 200
rho[131:] += 100
vsvp = 0.5
m = np.stack((np.log(vp), np.log(vs), np.log(rho)), axis=1)

# background model
nsmooth = 50
mback = filtfilt(np.ones(nsmooth) / float(nsmooth), 1, m, axis=0)

# angles
ntheta = 21
thetamin, thetamax = 0, 35
theta = np.linspace(thetamin, thetamax, ntheta)

# wavelet
ntwav = 77
wav = ricker(t0[: ntwav // 2 + 1], 35)[0]

# lop
PPop = pylops.avo.prestack.PrestackLinearModelling(
    wav, theta, vsvp=vsvp, nt0=nt0, linearization="akirich"
)

# dense
PPop_dense = pylops.avo.prestack.PrestackLinearModelling(
    wav, theta, vsvp=vsvp, nt0=nt0, linearization="akirich", explicit=True
)

# data lop
dPP = PPop * m.ravel()
dPP = dPP.reshape(nt0, ntheta)

# data dense
dPP_dense = PPop_dense * m.T.ravel()
dPP_dense = dPP_dense.reshape(ntheta, nt0).T

# noisy data
dPPn_dense = dPP_dense + np.random.normal(0, 1e-2, dPP_dense.shape)


file_path = "D:/code_projects/matlab_projects/src/trans_gan_inversion/dPP_dense.mat"

import scipy.io as scio
scio.savemat(file_path, dict(dPP_dense=dPP_dense, vp=vp, vs=vs, rho=rho/1000))
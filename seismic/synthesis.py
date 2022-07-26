import numpy as np
import pylops
from pylops.utils.wavelets import ricker


class Synthesis:
    """
    AVO prestack synthesis tool
    """
    def __init__(self, n_sample, theta_max=35.0, n_theta=21, wavelet_len=77, avo="akirich"):
        self.theta_min = 0.0
        self.theta_max = theta_max
        self.n_theta = n_theta
        self.n_sample = n_sample
        self.angles = np.linspace(self.theta_min, self.theta_max, self.n_theta)
        self.wavelet_len = wavelet_len
        self.avo = avo

    def gen_wavelet(self, freq, dt):
        """
        generate ricker wavelet
        """
        t0 = np.arange(self.n_sample) * dt
        wav, twav, wavc = ricker(t0[: self.wavelet_len // 2 + 1], freq)
        return wav

    def gen_m(self, vp, vs, rho):
        """
        generate model m， vp's shape is [n_trace, n_sample]
        """
        # return np.stack((np.log(vp), np.log(vs), np.log(rho * 1000)), axis=0)
        return np.stack((np.log(vp), np.log(vs), np.log(rho * 1000)), axis=2)
        # return np.hstack((np.log(vp), np.log(vs), np.log(rho * 1000)))

    def gen_pre_angle_data(self, vp, vs, rho, freq, dt):
        """
        generate prestack angle-gather seismic data
        :return:
        """
        n_trace = vp.shape[0]

        m = self.gen_m(vp, vs, rho)
        vsvp = vs / vp
        vsvp = vsvp.T

        d = []
        for i in range(n_trace):
            wav = self.gen_wavelet(freq[i], dt)
            G = pylops.avo.prestack.PrestackLinearModelling(
                wav, self.angles, vsvp=vsvp[:, i], nt0=self.n_sample, linearization=self.avo, explicit=False
            )
            # m = self.gen_m(vp[i, :], vs[i, :], rho[i, :])
            di = G * m[i, :, :].ravel()
            di = di.reshape(self.n_sample, self.n_theta).astype(np.float32)
            d.append(di)

        return d


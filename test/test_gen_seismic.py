import scipy.io as scio

from seismic.synthesis import Synthesis

file_path = "D:/code_projects/matlab_projects/src/trans_gan_inversion/welldata_tests.mat"
well_data = scio.loadmat(file_path)
well_data = well_data['wellData']

n_sample = well_data.shape[0]
syn = Synthesis(n_sample)

vp = well_data[:, 0:10, 1].squeeze().transpose()
vs = well_data[:, 0:10, 2].squeeze().transpose()
rho = well_data[:, 0:10, 3].squeeze().transpose()
freq = 35
dt = 0.002

seismic_data = syn.gen_pre_angle_data(vp, vs, rho, freq, dt)

# print(seismic_data)

scio.savemat(file_path.replace(".mat", "_angle.mat"), dict(seismic_data=seismic_data))

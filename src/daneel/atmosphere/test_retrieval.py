#%%
import numpy as np
import matplotlib.pyplot as plt

obs_file = "WASP-121b_assignment3_spectrum.dat"
ret_file = "WASP-121b_spectrum.txt"

# load observed: wavelength, depth, error
obs = np.loadtxt(obs_file)
w_obs, f_obs, e_obs = obs[:,0], obs[:,1], obs[:,2]

# load retrieved: wavelength, depth
ret = np.loadtxt(ret_file)
w_ret, f_ret = ret[:,0], ret[:,1]

plt.figure(figsize=(9,6))
plt.errorbar(w_obs, f_obs, yerr=e_obs, fmt='o', label='Observed', alpha=0.6)
plt.plot(w_ret, f_ret, '-', lw=2, label='Retrieved model')

plt.xscale('log')
plt.xlabel("Wavelength [micron]")
plt.ylabel("Transit depth $(R_p/R_*)^2$")
plt.title("WASP-121b Retrieval Check")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%

#%%
import numpy as np
import batman 
import csv 
import matplotlib.pyplot as plt

# %%
limb_dark_data = np.genfromtxt("HATS-12 b_mahdis.csv", delimiter=',', skip_header=17)
u1 = np.mean(limb_dark_data[:, 5])
u2 = np.mean(limb_dark_data[:, 7])
print(u1, u2)
# %%
params = batman.TransitParams()
params.t0   = 0.                  #time of inferior conjunction
params.per  = 3.142702            #orbital period in days
params.rp   = 0.692               #planet radius (in units of stellar radii)
params.a    = 0.0441              #semi-major axis (in units of stellar radii)
params.inc  = 852.7               #orbital inclination (in degrees)
params.ecc  = 0.085               #eccentricity
params.w    = 0.                  #longitude of periastron (in degrees)
params.u    = [u1, u2]            #limb darkening coefficients [u1, u2]
params.limb_dark = "quadratic"    #limb darkening model
# %%
t = np.linspace(-0.075, 0.075, 1000)

#%%
m = batman.TransitModel(params, t)	        #initializes model
flux = m.light_curve(params)

# %%
plt.plot(t, flux)
plt.xlabel("Time from central transit (days)")
plt.ylabel("Relative flux")
# plt.ylim((0.989, 1.001))
plt.savefig("lc.png") 
plt.show()
# %%

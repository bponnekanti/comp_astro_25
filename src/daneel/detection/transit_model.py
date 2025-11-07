##added this in class 31/10

import matplotlib.pyplot as plt
import batman
import numpy as np

class Transit_Model:
    def __init__(self):

        self.params = batman.TransitParams()
        self.params.t0   = 0.                #time of inferior conjunction
        self.params.per  = 433.0             #orbital period in days
        self.params.rp   = 0.0213            #planet radius (in units of stellar radii)
        self.params.a    = 222.343           #semi-major axis (in units of stellar radii)
        self.params.inc  = 0.                #orbital inclination (in degrees)
        self.params.ecc  = 0.08              #eccentricity
        self.params.w    = -66               #longitude of periastron (in degrees)
        self.params.u    = [u1, u2]          #limb darkening coefficients [u1, u2]
        self.params.limb_dark = "quadratic"  #limb darkening model

        self.t = np.linspace(-0.075, 0.075, 1000)

        self.model = batman.TransitModel(self.params, self.t)

    def compute_light_curve(self):

        self.flux = self.model.light_curve(self.params)
        return self.flux
    
    def plot_light_curve(self, output_file):

        if not hasattr(self, 'flux'):
            self.compute_light_curve()

        plt.plot(self.t, self.flux)
        plt.xlabel("Time from central transit (days)")
        plt.ylabel("Relative flux")
        plt.savefig(output_file)
        plt.show()


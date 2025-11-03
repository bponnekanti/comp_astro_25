#%%
## task F: 

# Transit simulation for HATS-12 b
# Data sources:
#  Exoplanet parameters: https://exoplanet.eu/catalog/hats_12_b--2596/
#  dark limb coeffficients: https://exoctk.stsci.edu/limb_darkening
#%%
import numpy as np
import batman 
import csv 
import matplotlib.pyplot as plt
#%%
##mahdis' notes: to double check what is my csv data, i print the first 33 lines. 
## command with open () as f: this opens our file and close it. 
## f.readline().strip() reads each line  as string and removes extra spaces (by strip).

with open("/ca25/comp_astro_25/src/daneel/detection/batman_tutorial_notebook/HATS-12 b_mahdis.csv") as f:
    for i in range(33):
        print(f.readline().strip())
# %%
## mahdis' notes :i cleaned my table manually, is it a problem ?
## np.genfromtxt reads the data from csv file and stores it in an array. 
## delimiter = '' means that the data is separated by commas. 
## skip_header = 0 means that we do not skip any rows at the top of the file. 
## u1 and u2 are the limb darkening coefficients(c1 and c2 in table ), which are in columns 7 and 9 of the csv file.

limb_dark_data = np.genfromtxt("/ca25/comp_astro_25/src/daneel/detection/batman_tutorial_notebook/HATS-12 b_mahdis.csv", delimiter='', skip_header=1)
u1 = np.mean(limb_dark_data[:, 8])
u2 = np.mean(limb_dark_data[:, 10])
print(u1, u2)
# %%
params = batman.TransitParams()
params.t0   = 0.                  #time of inferior conjunction
params.per  = 3.142702            #orbital period in days
params.rp   = 0.06951              #planet radius (in units of stellar radii)
## BECAREFUL ABOUT THE UNITHERE, IN SITE IS BASED ON JUPYTER RADIUS
params.a    = 9.48              #semi-major axis (in units of stellar radii)
# IN THE SITE IS BASED ON AU, MEED CONVERSION.
params.inc  = 85.27               #orbital inclination (in degrees)
params.ecc  = 0.085               #eccentricity
params.w    = 0.                  #longitude of periastron (in degrees)
params.u    = [u1, u2]            #limb darkening coefficients [u1, u2]
params.limb_dark = "quadratic"    #limb darkening model
# %%
## mahdis' notes: create an array of time values from -0.075 to 0.075 with 1000 points around the transit center.
t = np.linspace(-0.075, 0.075, 1000)
#%%
##mahdis' notes: initialize the transit model with the parameters and time array, then compute the light curve.
m = batman.TransitModel(params, t)	        #initializes model
flux = m.light_curve(params)
#%%
plt.plot(t, flux ,label = 'HATS-12 b Transit Light Curve')
plt.xlabel("Time from central transit (days)")
plt.ylabel("Relative flux")
##plt.ylim((0.989, 1.001))
## this command zoom in the plot vertically to see the transit better.it did not work well here because the transit depth is large.
plt.legend()
plt.savefig("HATS-12 b_assignment1_taskF.png") 
plt.show()
##mahdis' note : save the plot as lc.png and display it.
# %%
# %%
## Task G: definition of the daneel.transit method
import numpy as np
import batman
import matplotlib.pyplot as plt
import yaml

def transit(params_path):
    print("⚠️ Warning: This function will automatically skip only one header line in the CSV. "
          "Any additional headers or non-numeric rows must be cleaned manually before use.")

    # Load parameters from YAML
    with open(params_path, 'r') as file:
        param_dict = yaml.safe_load(file)

    # Read limb-darkening coefficients from CSV
    csv_path = param_dict["csv_path"]
    limb_dark_data = np.genfromtxt(csv_path, delimiter=' ', skip_header=1)
    u1 = np.mean(limb_dark_data[:, 8])
    u2 = np.mean(limb_dark_data[:, 10])

    # Define transit parameters
    params = batman.TransitParams()
    data = param_dict["transit_params"]

    params.t0   = data.get("t0", 0.0)      # Time of central transit (days)
    params.per  = data["period"]            # Orbital period (days)
    params.rp   = data["rp"]                # Planet radius (in units of stellar radii)
    params.a    = data["a"]                 # Semi-major axis (in units of stellar radii)
    params.inc  = data["inc"]               # Orbital inclination (degrees)
    params.ecc  = data["ecc"]               # Eccentricity
    params.w    = data["w"]                 # Longitude of periastron (degrees)
    params.u    = [u1, u2]                  # Limb-darkening coefficients
    params.limb_dark = "quadratic"          # Limb-darkening model

    # Time array around transit
    t = np.linspace(-0.075, 0.075, 1000)    # Time from central transit (days)

    # Initialize transit model and compute light curve
    m = batman.TransitModel(params, t)
    flux = m.light_curve(params)

    # Plot light curve
    plt.plot(t, flux, label="Transit Light Curve")
    plt.xlabel("Time from central transit (days)")
    plt.ylabel("Relative flux")
    plt.legend()
    plt.title("Transit Simulation")
    plt.savefig("transit_plot.png")
    plt.show()

# %%

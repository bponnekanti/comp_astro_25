## Assignment 2 - Task C: Transit Light Curve Modeling for Three Planets
#%%
import yaml
from transit_model import TransitModel
import matplotlib.pyplot as plt
#%%
# Load parameters from YAML configuration file
with open('../../../HATS-12b.yaml', 'r') as file :
    data = yaml.safe_load(file)["transit"]
#%%
##planet 1 parameters:
planet1_params = data.copy()

# planet 2 parameters:
planet2_params = data.copy()
planet2_params["rp"]= planet2_params["rp"] * 0.5
planet2_params["name"]= "HATS-12c"

# planet 3 parameters:
planet3_params = data.copy()
planet3_params["rp"]= planet3_params["rp"] * 2 
planet3_params["name"]= "HATS-12d"

#%%
#apply transit model for planet 1
planet1_transit = TransitModel(planet1_params)

## compute light curve
planet1_flux = planet1_transit.compute_light_curve()

##planet 2 :
planet2_transit = TransitModel(planet2_params)
planet2_flux = planet2_transit.compute_light_curve()

## planet 3 :
planet3_transit = TransitModel(planet3_params)
planet3_flux = planet3_transit.compute_light_curve()
#%%
# Plot both light curves
plt.figure(figsize=(10, 6))
plt.plot(planet1_transit.t, planet1_flux, label=planet1_params["name"], color='pink')
plt.plot(planet2_transit.t, planet2_flux, label=planet2_params  ["name"], color='lightblue')
plt.plot(planet3_transit.t, planet3_flux, label=planet3_params  ["name"], color='lightgreen')
plt.xlabel('Time from mid-transit (days)')
plt.ylabel('Relative Flux')
plt.title(f"{planet1_params['name']}, {planet2_params['name']} and {planet3_params['name']} transit light curves")
plt.legend()
plt.grid()
plt.savefig('assignment2_taskC_mahdis.png')
plt.show()
print("Light curves of three planets saved to assignment2_taskC_mahdis.png")

# %%

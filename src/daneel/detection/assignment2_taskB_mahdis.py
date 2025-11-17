## Assignment 2 - Task B: Transit Light Curve Modeling for Two Planets
#%%
import yaml
from transit_model import TransitModel
import matplotlib.pyplot as plt
#%%
# Load parameters from YAML configuration file
with open('/ca25/comp_astro_25/HATS-12b.yaml', 'r') as file :
    data = yaml.safe_load(file)["transit"]
#%%
##planet one parameters:
planet1_params = data.copy()

# planet two parameters:
planet2_params = data.copy()
planet2_params["rp"]= planet2_params["rp"] * 0.5
planet2_params["name"]= "HATS-12c"

#%%
#apply transit model for planet one
planet1_transit = TransitModel(planet1_params)

## compute light curve
planet1_flux = planet1_transit.compute_light_curve()

##planet two :
planet2_transit = TransitModel(planet2_params)
planet2_flux = planet2_transit.compute_light_curve()

#%%
# Plot both light curves
plt.figure(figsize=(10, 6))
plt.plot(planet1_transit.t, planet1_flux, label=planet1_params["name"], color='pink')
plt.plot(planet2_transit.t, planet2_flux, label=planet2_params  ["name"], color='lightblue')
plt.xlabel('Time from mid-transit (days)')
plt.ylabel('Relative Flux')
plt.title(f"{planet1_params['name']} and {planet2_params['name']} transit light curves")
plt.legend()
plt.grid()
plt.savefig('assignment2_taskB_mahdis.png')
plt.show()
print("Light curves of two planets saved to assignment2_taskB.png")

# %%

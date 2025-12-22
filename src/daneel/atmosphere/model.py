#%%
## run the base atmosphere model for WASP-121b with random abundances

import yaml
from base import ForwardModel

def run_forward_model():
    #Load parameters
    with open('wasp121b_atmosphere.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    #Create atmosphere model
    fm = ForwardModel(params_dict=params)
    fm.setup_environment()
    fm.setup_profile(random_abundances=True)
    fm.build_models()

    # Save outputs
    planet_name = fm.planet_name
    fm.save_spectrum(f"{planet_name}_assignment3_spectrum.dat")
    fm.save_parameters(f"{planet_name}_assignment3_parameters.txt")
    fm.plot_tm_spectrum(save=True, filename=f"{planet_name}_assignment3_spectrum.png")
    fm.plot_profiles(save=True, filename=f"{planet_name}_assignment3_profile.png")
    fm.plot_all_models(save=True, filename=f"{planet_name}_assignment3_all_models.png")
    #fm.interact_update_all_models()

    print(f"\n Forward model complete for {planet_name}!")
if __name__ == "__main__":
    run_forward_model()
# %%

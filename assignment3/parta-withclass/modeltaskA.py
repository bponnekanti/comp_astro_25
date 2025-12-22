#%%
# Task A: Generate transmission spectrum for WASP-121b

import sys
import os
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# Now import your BaseAtmosphere
from daneel.atmosphere.base import BaseAtmosphere

def run_taskA():

    yaml_file = 'wasp121b_atmosphere.yaml'
    
    # Create atmosphere model directly from YAML file
    atm = BaseAtmosphere(params_file=yaml_file)
    atm.setup_environment()
    atm.setup_profile(random_abundances=True) 
    atm.build_models()
    
    # Save outputs with EXACT filenames from assignment
    planet_name = atm.planet_name

    spectrum_file = f"{planet_name}_assignment3_taskA_spectrum.dat"
    atm.save_spectrum(spectrum_file)

    params_file = f"{planet_name}_assignment3_taskA_parameters.txt"
    atm.save_parameters(params_file)

    plot_file = f"{planet_name}_assignment3_taskA_spectrum.png"
    atm.plot_tm_spectrum(save=True, filename=plot_file)
    
    # Summary
    print("\n" + "=" * 50)
    print("TASK A COMPLETE!")
    print("=" * 50)
    print(f"Generated files for {planet_name}:")
    print(f"  1. {spectrum_file}")
    print(f"  2. {params_file}")
    print(f"  3. {plot_file}")
    print("\nRandomized molecular abundances:")
    for mol, abund in atm.gas_abundances.items():
        print(f"  {mol}: {abund:.2e}")
    print("=" * 50)
    
    return atm

if __name__ == "__main__":
    atmosphere_model = run_taskA()
# %%

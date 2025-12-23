#%%

"""
Task A: Generate transmission spectrum for your planet.
Run with: python run_taskA.py
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from daneel.atmosphere.base import ForwardModel

def main():
    """Run Task A forward model."""
    
    # Your planet's YAML file
    params_file = 'wasp121b_atmosphere.yaml'  # Or your planet name
    
    print("=" * 60)
    print("TASK A: Generating Transmission Spectrum")
    print("=" * 60)
    
    # Check if YAML file exists
    if not os.path.exists(params_file):
        print(f"ERROR: File '{params_file}' not found!")
        print("Current directory:", os.getcwd())
        print("Files in directory:", os.listdir('.'))
        return
    
    try:
        # Create and run the model
        print(f"Loading parameters from: {params_file}")
        model = ForwardModel(params_file=params_file)
        
        print("\nRunning forward model...")
        model.run(
            random_abundances=True,  # Task A: random abundances
            plot_all=False,           # Don't plot all models (TM, EM, DI)
            interactive=False         # Don't show interactive sliders
        )
        
        print("\n" + "=" * 60)
        print("TASK A COMPLETE!")
        print("Output files created:")
        print(f"  • {model.planet_name}_spectrum.txt")
        print(f"  • {model.planet_name}_params.txt") 
        print(f"  • {model.planet_name}_chemistry_profile.png")
        print(f"  • {model.planet_name}_tm_spectrum.png")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
# %%

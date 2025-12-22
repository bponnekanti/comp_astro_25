## mahdis's try: 
## Base class for atmospheres

#%%

import matplotlib.pyplot as plt
#%matplotlib widget
from ipywidgets import *
import numpy as np
import yaml
import sys
import os
from pathlib import Path
from datetime import datetime
import taurex.log
taurex.log.disableLogging()
from taurex.cache import OpacityCache,CIACache
from taurex.temperature import Guillot2010
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex.model import TransmissionModel , EmissionModel, DirectImageModel
from taurex.contributions import AbsorptionContribution, CIAContribution, RayleighContribution
from taurex.binning import SimpleBinner

class ForwardModel:
    def __init__(self, params_file = None, params_dict = None):
        """
        Base class for atmospheres.
        params: dictionary or loaded YAML file with all atmosphere parameters.

        """
        if params_file:
            with open(params_file, 'r') as f:
                self.params = yaml.safe_load(f)
        elif params_dict:
            self.params = params_dict
        else:
            raise ValueError("Either params_file or params_dict must be provided")

        # Extract parameters with defaults
        self.planet_name = self.params.get('planet_name', 'Unknown_Planet')
        self.planet_radius = self.params.get('planet', {}).get('radius', 1.0)
        self.planet_mass = self.params.get('planet', {}).get('mass', 1.0)

        # Star parameters
        star = self.params.get('star', {})
        self.star_temperature = star.get('temperature', 5700.0)
        self.star_radius = star.get('radius', 1.0)      # Solar radii
        
        # Atmosphere parameters
        atmosphere = self.params.get('atmosphere', {})
        self.T_irr = float(atmosphere.get('T_irr', 1200.0))
        self.atm_min_pressure = float(atmosphere.get('min_pressure', 1e0))
        self.atm_max_pressure = float(atmosphere.get('max_pressure', 1e6))
        self.nlayers = int(atmosphere.get('nlayers', 30))

        #molecules and chemistry 
        self.molecules = atmosphere.get('molecules', ['H2O', 'CH4', 'CO2', 'CO'])
        self.fill_gases = atmosphere.get('fill_gases', ['H2', 'He'])
        self.he_h2_ratio = atmosphere.get('he_h2_ratio', 0.172)
        self.cia_pairs = atmosphere.get('cia_pairs', ['H2-H2', 'H2-He'])

        # Output files - task E
        self.spectrum_file = atmosphere.get('output_spectrum', None)
        self.tm_plot_file = atmosphere.get('output_tm_plot', None)
        self.params_file = atmosphere.get('output_params', None)

        
        # Wavelength grid
        wavelength = self.params.get('wavelength', {})
        self.wavelength_min = float(wavelength.get('min', 0.3))
        self.wavelength_max = float(wavelength.get('max', 10.0))
        self.n_wavelength_points = int(wavelength.get('n_points', 1000))

        # Initialize object attributes
        self.planet = None
        self.star = None
        self.temperature_profile = None
        self.chemistry = None
        self.opacity_cache = None
        self.cia_cache = None
        self.tm = None
        self.em = None
        self.di = None
        self.binner = None
        self.model_result = None
        self.wngrid = None
        self.gas_abundances = {}

    def setup_environment(self):

        self.opacity_cache = OpacityCache()
        self.opacity_cache.clear_cache()
        self.opacity_cache.set_opacity_path("/home/ubuntu/comp_astro_25/assignment3/Taurex/data_taurex/xsecs")

        self.cia_cache = CIACache()
        self.cia_cache.set_cia_path("/home/ubuntu/comp_astro_25/assignment3/Taurex/data_taurex/cia/hitran")


    def setup_molecule(self, molecule_name, interact_plot=False):
        """
        Load a molecule's opacity and optionally plot its cross-section with interactive temperature & pressure.

        """
        xsec = self.opacity_cache[molecule_name]

        if interact_plot:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            line, = ax.plot(10000 / xsec.wavenumberGrid,xsec.opacity(800, 1.0))
            
            ax.set_xlabel('Wavelength')
            ax.set_ylabel('Cross Section (cm$^2$/molecule)')
            ax.set_title(f'Opacity for {molecule_name}')

            def update_opacity(temperature=1500.0, pressure=6.7):
                line.set_ydata(xsec.opacity(temperature, 10**pressure))
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()

            interact(update_opacity, temperature=(800.0, 2000.0, 100),pressure=(-1.0, 10.0, 1))

        return xsec

    def setup_profile(self, random_abundances=True):
        
        # Temperature Profile
        self.temperature_profile = Guillot2010(self.T_irr)

        # Planet
        self.planet = Planet(planet_radius=self.planet_radius, planet_mass=self.planet_mass)

        # Star
        self.star = BlackbodyStar(temperature=self.star_temperature, radius=self.star_radius)

        # Chemistry with random abundances
        self.chemistry = TaurexChemistry(fill_gases=self.fill_gases,ratio=self.he_h2_ratio)

        for mol in self.molecules:
            if random_abundances:
                mix_ratio = 10 ** np.random.uniform(-8, -2)
                self.gas_abundances[mol] = mix_ratio
            else:
                mix_ratio = self.params.get(f'{mol.lower()}_abundance', 10 ** np.random.uniform(-8, -2))
                self.gas_abundances[mol] = mix_ratio
            
            self.chemistry.addGas(ConstantGas(mol, mix_ratio=mix_ratio))
            print(f"Added {mol} with abundance: {mix_ratio:.2e}")


    def build_models(self):

        # Transmission Model
        self.tm = TransmissionModel(
            planet=self.planet,
            temperature_profile=self.temperature_profile,
            chemistry=self.chemistry,
            star=self.star,
            atm_min_pressure=self.atm_min_pressure,
            atm_max_pressure=self.atm_max_pressure,
            nlayers=self.nlayers
        )
        # Emission & Direct Image
        self.em = EmissionModel(
            planet=self.planet,
            temperature_profile=self.temperature_profile,
            chemistry=self.chemistry,
            star=self.star,
            atm_min_pressure=self.atm_min_pressure,
            atm_max_pressure=self.atm_max_pressure,
            nlayers=self.nlayers
        )
        self.di = DirectImageModel(
            planet=self.planet,
            temperature_profile=self.temperature_profile,
            chemistry=self.chemistry,
            star=self.star,
            atm_min_pressure=self.atm_min_pressure,
            atm_max_pressure=self.atm_max_pressure,
            nlayers=self.nlayers
        )

        # Add contributions and build models
        for model in [self.tm, self.em, self.di]:
            model.add_contribution(AbsorptionContribution())
            model.add_contribution(CIAContribution(cia_pairs=self.cia_pairs))
            model.add_contribution(RayleighContribution())
            model.build()


        # Prepare binner & run transmission model
        # Create wavelength grid
        wavelengths = np.logspace(np.log10(self.wavelength_min),
                                  np.log10(self.wavelength_max),
                                  self.n_wavelength_points)
        self.wngrid = np.sort(10000 / wavelengths)
        # Prepare binner
        self.binner = SimpleBinner(wngrid=self.wngrid)
        self.model_result = self.tm.model()
    
    def save_spectrum(self, filename, binned=True):

        if filename is None:
            raise ValueError("Filename must be provided in save_spectrum()")
        
        # Get spectrum
        if binned:
            bin_wn, bin_rprs, _, _ = self.binner.bin_model(self.tm.model(wngrid=self.wngrid))
            wavelengths = 10000 / bin_wn
            spectrum = bin_rprs
        else:
            native_grid, rprs, _, _ = self.model_result
            wavelengths = 10000 / native_grid
            spectrum = rprs
        
        # Estimate error (10 ppm)
        spectrum_squared = spectrum ** 2
        error = spectrum_squared * 1e-5
        
        # Save to file
        data = np.column_stack([wavelengths, spectrum_squared, error])
        np.savetxt(filename, data, 
                  header='wavelength[µm] (rp/rs)^2 error',
                  fmt='%.8e %.8e %.8e')
        
        print(f"✓ Spectrum saved to {filename}")
        return wavelengths, spectrum_squared, error
    
    def save_parameters(self, filename):

        if filename is None:
            raise ValueError("Filename must be provided in save_parameters()")
        
        with open(filename, 'w') as f:
            f.write(f"=== ATMOSPHERIC MODEL PARAMETERS ===\n")
            f.write(f"Planet: {self.planet_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("--- Planetary Parameters ---\n")
            f.write(f"Planet radius: {self.planet_radius} R_jup\n")
            f.write(f"Planet mass: {self.planet_mass} M_jup\n\n")
            
            f.write("--- Stellar Parameters ---\n")
            f.write(f"Star temperature: {self.star_temperature} K\n")
            f.write(f"Star radius: {self.star_radius} R_sun\n\n")
            
            f.write("--- Atmosphere Parameters ---\n")
            f.write(f"Irradiation temperature (T_irr): {self.T_irr} K\n")
            f.write(f"Minimum pressure: {self.atm_min_pressure:.2e} Pa\n")
            f.write(f"Maximum pressure: {self.atm_max_pressure:.2e} Pa\n")
            f.write(f"Number of layers: {self.nlayers}\n\n")
            
            f.write("--- Chemistry ---\n")
            f.write(f"Molecules included: {', '.join(self.molecules)}\n")
            f.write(f"Fill gases: {', '.join(self.fill_gases)}\n")
            f.write(f"He/H2 ratio: {self.he_h2_ratio}\n\n")
            
            f.write("--- Abundances (Mixing Ratios) ---\n")
            for mol, abund in self.gas_abundances.items():
                f.write(f"{mol}: {abund:.2e}\n")
            
            f.write("\n--- CIA Pairs ---\n")
            for pair in self.cia_pairs:
                f.write(f"{pair}\n")
            
            f.write("\n--- Wavelength Grid ---\n")
            f.write(f"Minimum wavelength: {self.wavelength_min} um\n")
            f.write(f"Maximum wavelength: {self.wavelength_max} um\n")
            f.write(f"Number of points: {self.n_wavelength_points}\n")
        
        print(f"✓ Parameters saved to {filename}")

    def plot_profiles(self, save =False, filename=None):
        """Plot chemistry profile (mixing ratios vs pressure)"""

        plt.figure()
        for x,gasname in enumerate(self.chemistry.activeGases):
            plt.plot(self.chemistry.activeGasMixProfile[x],self.tm.pressureProfile/1e5,label=gasname)
        for x,gasname in enumerate(self.chemistry.inactiveGases):
            plt.plot(self.chemistry.inactiveGasMixProfile[x],self.tm.pressureProfile/1e5,label=gasname)
        plt.gca().invert_yaxis()
        plt.yscale("log")
        plt.xscale("log")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlabel("Mixing ratio")
        plt.ylabel("Pressure [bar]")
        plt.title(f"Chemistry of the Atmosphere - {self.planet_name}")
        if save:
            if filename is None:
                raise ValueError("Filename must be provided when save=True")
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"✓ Chemistry profile saved to {filename}")
        plt.show()
        

    def plot_tm_spectrum(self, binned=True, save=False, filename=None):
        """Plot Transmission Model flux (native or binned)."""

        plt.figure(figsize=(10, 6))

        if binned:
            bin_wn, bin_rprs, _, _ = self.binner.bin_model(self.tm.model(wngrid=self.wngrid))
            wavelengths = 10000 / bin_wn
            spectrum = bin_rprs**2
            label = 'Binned spectrum'
            line_style = 'b-'
        else:
            # Get native spectrum
            native_result = self.tm.model()
            native_wn = native_result[0]
            wavelengths = 10000 / native_wn
            spectrum = native_result[1]**2
            label = 'Native spectrum'
            line_style = 'r-'

        plt.plot(wavelengths, spectrum, line_style, linewidth=2, label=label)
        plt.xscale("log")
        plt.xlabel("Wavelength [µm]")
        plt.ylabel("$(R_p/R_*)^2$")
        plt.title(f"Transmission Spectrum - {self.planet_name}", fontsize=14)
        plt.ylim(min(spectrum)*0.9, max(spectrum)*1.1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        if save:
            if filename is None:
                raise ValueError("Filename must be provided when save=True")
            plt.savefig(filename)
            print(f"✓ Spectrum plot saved to {filename}")
        plt.show()

    def plot_all_models(self, save=False, filename=None):
        """Plot TM, Emission, and Direct Image models together."""
        
        fig = plt.figure(figsize=(9,4))
        tm_ax = fig.add_subplot(1,3,1)
        em_ax = fig.add_subplot(1,3,2)
        di_ax = fig.add_subplot(1,3,3)
    
        model_tm = tm_ax.plot(10000/self.wngrid, self.binner.bin_model(self.tm.model(self.wngrid))[1])
        model_em = em_ax.plot(10000/self.wngrid, self.binner.bin_model(self.em.model(self.wngrid))[1])
        model_di = di_ax.plot(10000/self.wngrid, self.binner.bin_model(self.di.model(self.wngrid))[1])
        
        tm_ax.set_xscale('log')
        em_ax.set_xscale('log')
        di_ax.set_xscale('log')
        tm_ax.set_title('Transmission')
        em_ax.set_title('Emission')
        di_ax.set_title('Direct Image')

        tm_ax.set_xlabel('Wavelength (µm)')
        em_ax.set_xlabel('Wavelength (µm)')
        di_ax.set_xlabel('Wavelength (µm)')
        tm_ax.set_ylabel('Flux')
        
        plt.tight_layout()
        if save:
            if filename is None:
                raise ValueError("Filename must be provided when save=True")
            plt.savefig(filename)
            print(f"✓ All models plot saved to {filename}")
        plt.show()
        return model_tm, model_em, model_di

    def interact_update_all_models(self):
        """Interactive sliders to update temperature & H2O abundance for TM, EM, and DI."""
        from ipywidgets import interact

        native_grid = self.wngrid
        fig = plt.figure(figsize=(9,4))
        tm_ax = fig.add_subplot(1,3,1)
        em_ax = fig.add_subplot(1,3,2)
        di_ax = fig.add_subplot(1,3,3)

        # Initial plots
        model_tm, = tm_ax.plot(10000/native_grid, self.binner.bin_model(self.tm.model(native_grid))[1])
        model_em, = em_ax.plot(10000/native_grid, self.binner.bin_model(self.em.model(native_grid))[1])
        model_di, = di_ax.plot(10000/native_grid, self.binner.bin_model(self.di.model(native_grid))[1])

        # Axes formatting
        for ax, title in zip([tm_ax, em_ax, di_ax], ['Transmission', 'Emission', 'Direct Image']):
            ax.set_xscale('log')
            ax.set_title(title)
            ax.set_xlabel('Wavelength (µm)')
            ax.grid(True, alpha=0.3)
        tm_ax.set_ylabel('Flux')

        def update_model(temperature=1500.0, h2o_mix=-4, ch4_mix=-5, co2_mix=-5, co_mix=-5):
            self.temperature_profile.equilTemperature = temperature

            self.tm['H2O'] = 10**h2o_mix
            self.em['H2O'] = 10**h2o_mix
            self.di['H2O'] = 10**h2o_mix

            self.tm['CH4'] = 10**ch4_mix
            self.em['CH4'] = 10**ch4_mix
            self.di['CH4'] = 10**ch4_mix

            self.tm['CO2'] = 10**co2_mix
            self.em['CO2'] = 10**co2_mix
            self.di['CO2'] = 10**co2_mix

            self.tm['CO'] = 10**co_mix
            self.em['CO'] = 10**co_mix
            self.di['CO'] = 10**co_mix

            # Update all models
            model_tm.set_ydata(self.binner.bin_model(self.tm.model(native_grid))[1])
            model_em.set_ydata(self.binner.bin_model(self.em.model(native_grid))[1])
            model_di.set_ydata(self.binner.bin_model(self.di.model(native_grid))[1])

            tm_ax.relim();
            tm_ax.autoscale_view()
            
            em_ax.relim();
            em_ax.autoscale_view()
            
            di_ax.relim();
            di_ax.autoscale_view()
            
            fig.canvas.draw()

        interact(update_model, 
                temperature=(800.0, 2000.0, 100), 
                h2o_mix=(-7.0, -2.0, 0.1),
                ch4_mix=(-7.0, -2.0, 0.1),
                co2_mix=(-7.0, -2.0, 0.1),
                co_mix=(-7.0, -2.0, 0.1))

        plt.tight_layout()
    
    def run(self, random_abundances=False, plot_all=False, interactive=False):
        print(f"\n{'='*50}")
        print(f"Running atmospheric analysis for {self.planet_name}")
        print(f"{'='*50}")

        # Step 1: Setup environment
        print("\n[1/5] Setting up environment...")
        self.setup_environment()

        # Step 2: Setup profiles
        print("[2/5] Setting up profiles...")
        self.setup_profile(random_abundances=random_abundances)

        # Step 3: Build models
        print("[3/5] Building models...")
        self.build_models()

        # Step 4: Save outputs
        print("[4/5] Saving outputs...")
        
        spectrum_file = self.spectrum_file
        params_file = self.params_file
        tm_plot_file = self.tm_plot_file
        
        if self.spectrum_file is None:
            self.spectrum_file = f"{self.planet_name}_spectrum.txt"
        if self.params_file is None:
            self.params_file = f"{self.planet_name}_params.txt"
        if self.tm_plot_file is None:
            self.tm_plot_file = f"{self.planet_name}_tm_spectrum.png"
    
        self.save_spectrum(filename=spectrum_file)
        self.save_parameters(filename=params_file)

        # Step 5: Create plots
        print("[5/5] Creating plots...")
        self.plot_profiles(save=True)
        self.plot_tm_spectrum(binned=True, save=True, filename=tm_plot_file)

        # Optional: interactive sliders (only in notebook)
        if interactive:
            print("[5.5/5] Launching interactive mode...")
            self.interact_update_all_models()
        
        # Optional: plot all models (TM, EM, DI)
        if plot_all:
            print("[6/5] Plotting all models (TM, EM, DI)...")
            self.plot_all_models()

        print(f"\n{'='*50}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*50}")
        print(f"✓ Spectrum saved to: {spectrum_file}")
        print(f"✓ Parameters saved to: {params_file}")
        print(f"✓ Transmission plot saved to: {tm_plot_file}")
        if plot_all:
            print("✓ All models plot displayed")
        print(f"{'='*50}")
# %%

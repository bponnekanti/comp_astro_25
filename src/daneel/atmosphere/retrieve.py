#%%
##mahdis: retrieval class 
## for task F

import numpy as np
import matplotlib.pyplot as plt
from taurex.data.spectrum.observed import ObservedSpectrum
from taurex.optimizer.nestle import NestleOptimizer
from taurex.temperature import Isothermal
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex.temperature import Guillot2010
import yaml
from daneel.atmosphere.base import ForwardModel

class RetrievalModel(ForwardModel):
    """Retrieval model class for atmospheric parameter estimation."""
    def __init__(self, params_file=None, params_dict=None):
        # Initialize parent class
        super().__init__(params_file, params_dict)
        
        # Retrieval-specific parameters
        self.observed_spectrum_path = self.params.get('retrieval', {}).get('observed_spectrum', None)
        self.fit_parameters = self.params.get('retrieval', {}).get('fit_parameters', [])
        self.priors = self.params.get('retrieval', {}).get('priors', {})
        self.num_live_points = self.params.get('retrieval', {}).get('num_live_points', 50)
        
        self.obs = None
        self.optimizer = None
        self.binner = None

        # Initialize isothermal flag
        self.use_isothermal = False

    def setup_profile(self, random_abundances=True):
        """Override parent method to switch temperature profile if fitting T"""
        super().setup_profile(random_abundances)
 
        # Determine which temperature profile to use
        if 'T' in self.fit_parameters:
            print("[Retrieval] Using Isothermal temperature profile (fitting T).")
            if 'T' in self.priors:
                bounds = self.priors['T']
                default_T = (bounds[0] + bounds[1]) / 2.0
            else:
                default_T = 1500.0
            
            # Create Isothermal profile
            self.temperature_profile = Isothermal(default_T)
            self.use_isothermal = True
        else:
            print("[Retrieval] Using Guillot2010 temperature profile.")
            # Parent already created Guillot2010, but we might need to update it
            self.temperature_profile = Guillot2010(self.T_irr)
            self.use_isothermal = False
        
        # IMPORTANT: Update the transmission model with new temperature profile
        if hasattr(self, 'tm') and self.tm is not None:
            self.tm.temperature_profile = self.temperature_profile
    
    def setup_retrieval(self):
        """Setup the observed spectrum and optimizer and configure fitting parameters."""

        if self.observed_spectrum_path is None:
            raise ValueError("Observed spectrum path must be provided in parameters.")
        
        # Load observed spectrum
        self.obs = ObservedSpectrum(self.observed_spectrum_path)
        
        # Setup optimizer
        self.optimizer = NestleOptimizer(num_live_points=self.num_live_points)
        
        # Setup binner
        self.binner = self.obs.create_binner()

        # Set model and observed data
        self.optimizer.set_model(self.tm)
        self.optimizer.set_observed(self.obs)

        print("[Retrieval] Available TM fitting parameters:")

        available_params = list(self.tm.fittingParameters.keys())
        print(f"  {available_params}")

        for param_name in self.fit_parameters:
            if param_name not in available_params:
                raise ValueError(f"[ERROR] '{param_name}' not in fittingParameters")
            actual_param = param_name

            if param_name == 'T':
            # Try different temperature parameter names
                if 'T_iso' in available_params:
                    actual_param = 'T_iso'
                    print(f"  -> Using 'T_iso' instead of 'T'")
                elif 'temperature' in available_params:
                    actual_param = 'temperature'
                    print(f"  -> Using 'temperature' instead of 'T'")
                elif 'T' in available_params:
                    actual_param = 'T'  # Keep as is
                else:
                    print(f"[WARNING] Temperature parameter '{param_name}' not found in model")
                    continue  # Skip this parameter
            
            elif param_name == 'T_irr':
                if 'T_irr' not in available_params:
                    print(f"[WARNING] T_irr not available in model")
                    continue
            
            # Check if parameter exists
            if actual_param not in available_params:
                print(f"[WARNING] Parameter '{actual_param}' not in fitting parameters. Skipping.")
                continue
            
            # ENABLE FITTING FOR THIS PARAMETER
            self.optimizer.enable_fit(actual_param)
            print(f"[Retrieval] Enabled fitting for {param_name} -> {actual_param}")
            
            # Set boundaries if provided
            if param_name in self.priors:
                bounds = [float(b) for b in self.priors[param_name]]
                
                # Set initial value (midpoint of prior)
                if param_name in ['T', 'T_irr']:
                    # Temperature parameters: linear scale
                    initial_value = (bounds[0] + bounds[1]) / 2.0
                else:
                    # Abundance parameters: log scale
                    # Check if bounds are already in log space
                    if bounds[0] > 0 and bounds[1] > 0:
                        initial_value = 10 ** ((np.log10(bounds[0]) + np.log10(bounds[1])) / 2)
                    else:
                        initial_value = (bounds[0] + bounds[1]) / 2.0
                
                # Try to set initial value in model
                try:
                    self.tm[actual_param] = initial_value
                except Exception as e:
                    print(f"[INFO] Could not set initial value for {actual_param}: {e}")
                
                # Set boundaries in optimizer
                self.optimizer.set_boundary(actual_param, bounds)
                print(f"[Retrieval] Set prior for {param_name}: {bounds} (initial: {initial_value:.2e})")  

    def run_retrieval(self):
        """Run the retrieval optimization."""
        print("[Retrieval] Starting retrieval...")
                
        solution = self.optimizer.fit()
        print("[Retrieval] Retrieval complete!")
        
        return solution

    def save_retrieved_spectrum(self, filename=None):
        """Save the best-fit spectrum."""
        if filename is None:
            filename = self.params.get('retrieval', {}).get('output_retrieved_spectrum',
                                                            f"{self.planet_name}_retrieved.txt")
        try:
            solutions = list(self.optimizer.get_solution())
            if solutions:
                # Use the first (best) solution
                _, optimized_map, _, _ = solutions[0]
                self.optimizer.update_model(optimized_map)
                print(f"[Retrieval] Applied best-fit parameters from retrieval")
            else:
                print("[Retrieval] No solutions found, using current model state")
        except Exception as e:
            print(f"[Retrieval] Could not get best solution: {e}. Using current model.")

        wavelengths = self.obs.wavelengthGrid
        best_fit_spectrum = self.binner.bin_model(self.tm.model(self.obs.wavenumberGrid))[1]
        
        # Save to file
        data = np.column_stack([wavelengths, best_fit_spectrum])
        np.savetxt(filename, data, header='wavelength[µm] (rp/rs)^2')
        
        print(f"[Retrieval] Saved retrieved spectrum to {filename}")
        return filename
    
    def plot_fit(self):
        """Plot observed vs best-fit spectrum."""
        plt.figure(figsize=(10, 6))

        solutions = list(self.optimizer.get_solution())
        if solutions:
            _, optimized_map, _, _ = solutions[0]
            self.optimizer.update_model(optimized_map)
        
        # Plot observed
        plt.errorbar(self.obs.wavelengthGrid, self.obs.spectrum, 
                    self.obs.errorBar, label='Observed', fmt='o', alpha=0.7)
        
        # Plot model
        model_spectrum = self.binner.bin_model(self.tm.model(self.obs.wavenumberGrid))[1]
        plt.plot(self.obs.wavelengthGrid, model_spectrum, 'r-', label='Best fit', linewidth=2)
        
        plt.xscale('log')
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('$(R_p/R_*)^2$')
        plt.title(f'Retrieval Fit - {self.planet_name}')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plot_file = self.params.get('retrieval', {}).get('output_fit_plot', None)
        if plot_file:
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"[Retrieval] Saved fit plot to {plot_file}")
        
        plt.show()
    
    def plot_posterior(self, filename=None):
        """Plot posterior distributions of retrieved parameters."""
        if filename is None:
            filename = self.params.get('retrieval', {}).get('output_posterior_plot',
                                                        f"{self.planet_name}_posterior.png")
        samples = self.optimizer.samples
        if samples is None:
            print("[Posterior] No samples available from optimizer.")
            return
        
        # Get parameters we tried to fit
        params = list(self.fit_parameters)
        if not params:
            print("[Posterior] No parameters were fitted.")
            return
        
        n = len(params)
        
        # Create figure with subplots
        fig, axes = plt.subplots(n, 1, figsize=(8, 4*n), squeeze=False)
        
        # Plot each parameter
        for i, p in enumerate(params):
            ax = axes[i, 0]
            
            # Find the matching sample key (handle name mapping)
            sample_key = None
            for key in samples.keys():
                # Check for direct match or partial match
                if p == key:
                    sample_key = key
                    break
                elif p in key or key in p:  # Handle mapping like 'T' -> 'T_iso'
                    sample_key = key
                    break
            
            if sample_key:
                # Get samples for this parameter
                param_samples = samples[sample_key]
                
                # Skip if no valid samples
                if len(param_samples) == 0:
                    ax.text(0.5, 0.5, f"No samples for {p}", 
                        ha='center', va='center', transform=ax.transAxes)
                    ax.set_xlabel(p)
                    continue
                
                # Plot histogram
                ax.hist(param_samples, bins=30, alpha=0.7, density=True, 
                    edgecolor='black', linewidth=0.5)
                
                # Calculate statistics
                mean_val = np.mean(param_samples)
                median_val = np.median(param_samples)
                std_val = np.std(param_samples)
                
                # Add vertical lines for statistics
                ax.axvline(mean_val, color='red', linestyle='-', 
                        linewidth=2, alpha=0.7, label=f'Mean: {mean_val:.3f}')
                ax.axvline(median_val, color='blue', linestyle='--', 
                        linewidth=2, alpha=0.7, label=f'Median: {median_val:.3f}')
                
                # Add shaded region for 1-sigma
                ax.axvspan(mean_val - std_val, mean_val + std_val, 
                        alpha=0.2, color='gray', label=f'±1σ: {std_val:.3f}')
                
                # Labels and legend
                ax.set_xlabel(f"{p} ({sample_key})")
                ax.set_ylabel("Probability Density")
                ax.legend(loc='upper right', fontsize=9)
                
                # Add text box with statistics
                stats_text = f"N = {len(param_samples)}\n" \
                            f"Mean = {mean_val:.3f}\n" \
                            f"Std = {std_val:.3f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Add grid
                ax.grid(True, alpha=0.3, linestyle='--')
                
            else:
                # No samples found for this parameter
                ax.text(0.5, 0.5, f"No samples found for '{p}'\n" \
                    f"Available: {list(samples.keys())}", 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_xlabel(p)
        
        # Adjust layout and save
        plt.suptitle(f"Posterior Distributions - {self.planet_name}", fontsize=14, y=1.02)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[Posterior] Saved to {filename}")
        
        # Optional: Show the plot
        plt.show()
        print(f"[Posterior] Saved to {filename}")
        
    def run(self):
        """Main retrieval pipeline."""
        print(f"\n{'='*50}")
        print(f"Running RETRIEVAL for {self.planet_name}")
        print(f"{'='*50}")
        
        # Step 1-3: Standard forward model setup (from parent)
        super().setup_environment()
        self.setup_profile(random_abundances=False)  # Don't randomize for retrieval
        super().build_models()
        
        # Step 4: Retrieval-specific setup
        self.setup_retrieval()
        
        # Step 5: Run retrieval
        solution = self.run_retrieval()
        
        # Step 6: Save results
        retrieved_file = self.save_retrieved_spectrum()
        
        # Step 7: Plot
        self.plot_fit()
        
        print(f"\n{'='*50}")
        print("RETRIEVAL COMPLETE!")
        print(f"Saved retrieved spectrum: {retrieved_file}")
        print(f"{'='*50}")
        
        return solution
# %%

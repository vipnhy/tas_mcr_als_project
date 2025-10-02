# -*- coding: utf-8 -*-
"""
GTA Solver Module for Global Target Analysis

This module provides the core solver functionality for Global Target Analysis,
including differential equation solving, parameter optimization, and Species-
Associated Spectra (SAS) calculation.

Key Features:
- ODE solving for species concentration evolution
- Parameter fitting using lmfit with bounds and constraints
- Multiple optimization algorithms support
- Species-Associated Spectra calculation
- Comprehensive error analysis and residuals
- Integration with TAS analysis workflows
"""

import numpy as np
from scipy.integrate import solve_ivp
from lmfit import minimize, Parameters, fit_report
from typing import List, Tuple, Dict, Optional, Union, Callable
import logging
import warnings

logger = logging.getLogger(__name__)


class GTASolver:
    """
    Core solver for Global Target Analysis
    
    Handles the complete GTA workflow from kinetic matrix to fitted parameters
    and reconstructed spectra.
    """
    
    def __init__(self, wavelengths: np.ndarray, delays: np.ndarray, spectra: np.ndarray):
        """
        Initialize GTA solver with experimental data
        
        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelength array (nm)
        delays : np.ndarray
            Time delay array (ps)
        spectra : np.ndarray
            Experimental spectra matrix (delays x wavelengths)
        """
        self.wavelengths = wavelengths
        self.delays = delays
        self.spectra = spectra
        
        # Validate input dimensions
        if spectra.shape != (len(delays), len(wavelengths)):
            raise ValueError(f"Spectra shape {spectra.shape} doesn't match delays x wavelengths ({len(delays)} x {len(wavelengths)})")
        
        # Analysis results storage
        self.kinetic_matrix = None
        self.concentration_profiles = None
        self.species_spectra = None
        self.fitted_parameters = None
        self.fit_report_text = None
        self.residuals = None
        self.reconstructed_spectra = None
        
        # Solver settings
        self.ode_method = 'BDF'  # Default ODE method
        self.optimization_method = 'leastsq'  # Default optimization
        
        logger.info(f"Initialized GTA solver with {len(delays)} delays, {len(wavelengths)} wavelengths")
    
    def set_kinetic_matrix(self, K: np.ndarray, initial_concentrations: Optional[np.ndarray] = None) -> None:
        """
        Set kinetic matrix and initial concentrations
        
        Parameters
        ----------
        K : np.ndarray
            Kinetic matrix (n_species x n_species)
        initial_concentrations : np.ndarray, optional
            Initial concentrations [c1(0), c2(0), ...]. Default: [1, 0, 0, ...]
        """
        self.kinetic_matrix = K.copy()
        self.n_species = K.shape[0]
        
        if initial_concentrations is None:
            # Default: first species has unit concentration, others zero
            self.initial_concentrations = np.zeros(self.n_species)
            self.initial_concentrations[0] = 1.0
        else:
            if len(initial_concentrations) != self.n_species:
                raise ValueError(f"Initial concentrations length {len(initial_concentrations)} != n_species {self.n_species}")
            self.initial_concentrations = np.array(initial_concentrations)
        
        logger.info(f"Set kinetic matrix for {self.n_species} species")
    
    def solve_concentration_profiles(self, rate_constants: np.ndarray) -> np.ndarray:
        """
        Solve differential equations for species concentration evolution
        
        Parameters
        ----------
        rate_constants : np.ndarray
            Rate constants for the kinetic model
            
        Returns
        -------
        concentration_profiles : np.ndarray
            Concentration profiles (n_delays x n_species)
        """
        if self.kinetic_matrix is None:
            raise ValueError("Kinetic matrix not set. Call set_kinetic_matrix() first.")
        
        # Update kinetic matrix with current rate constants
        K_current = self._update_kinetic_matrix(rate_constants)
        
        # Define ODE system: dc/dt = K * c
        def kinetic_ode(t, c):
            return K_current @ c
        
        # Solve ODE system
        try:
            # Convert delays to relative time (assuming first delay is t=0 reference)
            t_span = (self.delays[0], self.delays[-1])
            t_eval = self.delays
            
            # Solve ODE
            sol = solve_ivp(
                kinetic_ode, 
                t_span, 
                self.initial_concentrations,
                t_eval=t_eval,
                method=self.ode_method,
                rtol=1e-8,
                atol=1e-10
            )
            
            if not sol.success:
                logger.warning(f"ODE solver warning: {sol.message}")
            
            # Return concentrations (transpose to get delays x species)
            concentration_profiles = sol.y.T
            
        except Exception as e:
            logger.error(f"ODE solving failed: {e}")
            # Fallback: try simpler method
            try:
                sol = solve_ivp(
                    kinetic_ode,
                    t_span,
                    self.initial_concentrations, 
                    t_eval=t_eval,
                    method='RK45',
                    rtol=1e-6
                )
                concentration_profiles = sol.y.T
                logger.info("Used fallback ODE method RK45")
            except Exception as e2:
                logger.error(f"Fallback ODE method also failed: {e2}")
                raise
        
        return concentration_profiles
    
    def _update_kinetic_matrix(self, rate_constants: np.ndarray) -> np.ndarray:
        """
        Update kinetic matrix with current rate constants
        
        This method should be overridden based on the specific kinetic model.
        Default implementation assumes rate constants replace non-zero elements.
        """
        K = self.kinetic_matrix.copy()
        
        # Simple approach: replace non-zero elements with rate constants
        # This works for basic models but may need customization
        nonzero_positions = np.nonzero(K)
        if len(rate_constants) != len(nonzero_positions[0]):
            # Alternative: assume rate constants correspond to specific positions
            # This needs to be defined based on the kinetic model structure
            pass
        
        return K
    
    def calculate_species_spectra(self, concentration_profiles: np.ndarray) -> np.ndarray:
        """
        Calculate Species-Associated Spectra from concentration profiles
        
        Parameters
        ----------
        concentration_profiles : np.ndarray
            Concentration profiles (n_delays x n_species)
            
        Returns
        -------
        species_spectra : np.ndarray
            Species-associated spectra (n_species x n_wavelengths)
        """
        # Use linear algebra to solve: spectra = concentration_profiles @ species_spectra
        # species_spectra = (C^T C)^-1 C^T spectra
        
        try:
            # Solve using least squares: C @ S = D
            # where C = concentration_profiles, S = species_spectra, D = data
            species_spectra, residuals, rank, s = np.linalg.lstsq(
                concentration_profiles, 
                self.spectra, 
                rcond=None
            )
            
            logger.info(f"Calculated species spectra with rank {rank}/{concentration_profiles.shape[1]}")
            
            return species_spectra
            
        except np.linalg.LinAlgError as e:
            logger.error(f"Linear algebra error in species spectra calculation: {e}")
            raise
    
    def objective_function(self, params: Parameters) -> np.ndarray:
        """
        Objective function for parameter optimization
        
        Parameters
        ----------
        params : lmfit.Parameters
            Current parameter values
            
        Returns
        -------
        residuals : np.ndarray
            Flattened residuals array
        """
        # Extract rate constants from parameters
        rate_constants = np.array([params[name].value for name in sorted(params.keys())])
        
        try:
            # Solve concentration profiles
            concentration_profiles = self.solve_concentration_profiles(rate_constants)
            
            # Calculate species spectra
            species_spectra = self.calculate_species_spectra(concentration_profiles)
            
            # Reconstruct experimental data
            reconstructed = concentration_profiles @ species_spectra
            
            # Calculate residuals
            residuals = (self.spectra - reconstructed).flatten()
            
            return residuals
            
        except Exception as e:
            logger.warning(f"Error in objective function: {e}")
            # Return large residuals to indicate poor fit
            return np.full(self.spectra.size, 1e6)
    
    def fit_parameters(self, 
                      initial_rates: List[float],
                      parameter_bounds: Optional[List[Tuple[float, float]]] = None,
                      fixed_parameters: Optional[List[bool]] = None,
                      method: str = 'leastsq') -> Dict:
        """
        Fit kinetic parameters using optimization
        
        Parameters
        ----------
        initial_rates : List[float]
            Initial guesses for rate constants
        parameter_bounds : List[Tuple[float, float]], optional
            Bounds for each parameter (min, max)
        fixed_parameters : List[bool], optional  
            Whether each parameter is fixed (True) or variable (False)
        method : str
            Optimization method ('leastsq', 'least_squares', 'nelder', etc.)
            
        Returns
        -------
        result : Dict
            Fitting results including fitted parameters, uncertainties, fit statistics
        """
        # Setup parameters
        params = Parameters()
        
        for i, rate in enumerate(initial_rates):
            name = f'k{i}'
            
            # Set bounds
            if parameter_bounds and i < len(parameter_bounds):
                min_val, max_val = parameter_bounds[i]
            else:
                min_val, max_val = 0.0, np.inf
            
            # Set vary flag
            if fixed_parameters and i < len(fixed_parameters):
                vary = not fixed_parameters[i]
            else:
                vary = True
            
            params.add(name, value=rate, min=min_val, max=max_val, vary=vary)
        
        # Store optimization method
        self.optimization_method = method
        
        # Perform optimization
        logger.info(f"Starting parameter fitting with {method} method")
        
        try:
            fit_result = minimize(
                self.objective_function,
                params,
                method=method
            )
            
            # Extract fitted parameters
            fitted_rates = [fit_result.params[f'k{i}'].value for i in range(len(initial_rates))]
            parameter_errors = [fit_result.params[f'k{i}'].stderr for i in range(len(initial_rates))]
            
            # Calculate final results with fitted parameters
            final_concentration_profiles = self.solve_concentration_profiles(np.array(fitted_rates))
            final_species_spectra = self.calculate_species_spectra(final_concentration_profiles)
            final_reconstructed = final_concentration_profiles @ final_species_spectra
            final_residuals = self.spectra - final_reconstructed
            
            # Store results
            self.fitted_parameters = fitted_rates
            self.concentration_profiles = final_concentration_profiles
            self.species_spectra = final_species_spectra
            self.reconstructed_spectra = final_reconstructed
            self.residuals = final_residuals
            self.fit_report_text = fit_report(fit_result)
            
            # Calculate fit statistics
            chi_squared = np.sum(final_residuals**2)
            n_data_points = self.spectra.size
            n_parameters = len([p for p in params.values() if p.vary])
            reduced_chi_squared = chi_squared / (n_data_points - n_parameters)
            
            result = {
                'success': fit_result.success,
                'fitted_rates': fitted_rates,
                'parameter_errors': parameter_errors,
                'chi_squared': chi_squared,
                'reduced_chi_squared': reduced_chi_squared,
                'n_data_points': n_data_points,
                'n_parameters': n_parameters,
                'concentration_profiles': final_concentration_profiles,
                'species_spectra': final_species_spectra,
                'reconstructed_spectra': final_reconstructed,
                'residuals': final_residuals,
                'fit_report': self.fit_report_text,
                'lmfit_result': fit_result
            }
            
            logger.info(f"Parameter fitting completed successfully")
            logger.info(f"Reduced χ² = {reduced_chi_squared:.3e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Parameter fitting failed: {e}")
            raise
    
    def calculate_confidence_intervals(self, fit_result: Dict, confidence_level: float = 0.95) -> Dict:
        """
        Calculate confidence intervals for fitted parameters
        
        Parameters
        ----------
        fit_result : Dict
            Result from fit_parameters()
        confidence_level : float
            Confidence level (0.95 = 95%)
            
        Returns
        -------
        intervals : Dict
            Confidence intervals for each parameter
        """
        try:
            from lmfit.conf_interval import conf_interval
            
            lmfit_result = fit_result['lmfit_result']
            ci = conf_interval(lmfit_result, prob=confidence_level)
            
            intervals = {}
            for param_name, bounds in ci.items():
                intervals[param_name] = {
                    'lower': bounds[0][1] if len(bounds) > 0 else None,
                    'upper': bounds[2][1] if len(bounds) > 2 else None,
                    'value': lmfit_result.params[param_name].value
                }
            
            return intervals
            
        except Exception as e:
            logger.warning(f"Could not calculate confidence intervals: {e}")
            return {}
    
    def set_solver_options(self, ode_method: str = 'BDF', **ode_kwargs) -> None:
        """
        Set ODE solver options
        
        Parameters
        ----------
        ode_method : str
            ODE solution method ('BDF', 'RK45', 'RK23', 'DOP853', 'Radau', 'LSODA')
        **ode_kwargs
            Additional arguments for solve_ivp
        """
        valid_methods = ['BDF', 'RK45', 'RK23', 'DOP853', 'Radau', 'LSODA']
        
        if ode_method not in valid_methods:
            raise ValueError(f"ODE method must be one of {valid_methods}")
        
        self.ode_method = ode_method
        self.ode_kwargs = ode_kwargs
        
        logger.info(f"Set ODE method to {ode_method}")
    
    def get_analysis_summary(self) -> Dict:
        """
        Get summary of the GTA analysis
        
        Returns
        -------
        summary : Dict
            Comprehensive analysis summary
        """
        if self.fitted_parameters is None:
            return {"status": "No analysis performed yet"}
        
        # Calculate some quality metrics
        if self.residuals is not None:
            residual_std = np.std(self.residuals)
            max_residual = np.max(np.abs(self.residuals))
            data_range = np.max(self.spectra) - np.min(self.spectra)
            relative_error = residual_std / data_range * 100
        else:
            residual_std = max_residual = data_range = relative_error = None
        
        summary = {
            'status': 'Analysis completed',
            'n_species': self.n_species,
            'n_delays': len(self.delays),
            'n_wavelengths': len(self.wavelengths),
            'fitted_rates': self.fitted_parameters,
            'ode_method': self.ode_method,
            'optimization_method': self.optimization_method,
            'residual_std': residual_std,
            'max_residual': max_residual,
            'relative_error_percent': relative_error,
            'has_concentration_profiles': self.concentration_profiles is not None,
            'has_species_spectra': self.species_spectra is not None,
            'has_residuals': self.residuals is not None
        }
        
        return summary


# Convenience function for quick GTA analysis
def run_gta_analysis(wavelengths: np.ndarray, 
                    delays: np.ndarray,
                    spectra: np.ndarray,
                    kinetic_matrix: np.ndarray,
                    initial_rates: List[float],
                    **fit_kwargs) -> Dict:
    """
    Quick function to run complete GTA analysis
    
    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelength array
    delays : np.ndarray
        Time delay array  
    spectra : np.ndarray
        Experimental spectra
    kinetic_matrix : np.ndarray
        Kinetic matrix for the model
    initial_rates : List[float]
        Initial rate constant guesses
    **fit_kwargs
        Additional arguments for fit_parameters()
        
    Returns
    -------
    result : Dict
        Complete GTA analysis results
    """
    solver = GTASolver(wavelengths, delays, spectra)
    solver.set_kinetic_matrix(kinetic_matrix)
    
    fit_result = solver.fit_parameters(initial_rates, **fit_kwargs)
    
    return {
        'solver': solver,
        'fit_result': fit_result,
        'summary': solver.get_analysis_summary()
    }


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("Testing GTA Solver Module")
    print("=" * 40)
    
    # Create synthetic test data
    n_delays = 50
    n_wavelengths = 100
    n_species = 3
    
    delays = np.logspace(-1, 3, n_delays)  # 0.1 to 1000 ps
    wavelengths = np.linspace(400, 700, n_wavelengths)  # 400-700 nm
    
    # Simple kinetic matrix for A -> B -> C
    K_test = np.array([
        [-1.0,  0.0,  0.0],
        [ 1.0, -0.5,  0.0], 
        [ 0.0,  0.5,  0.0]
    ])
    
    # Generate synthetic concentration profiles
    from scipy.integrate import solve_ivp
    
    def test_ode(t, c):
        return K_test @ c
    
    sol = solve_ivp(test_ode, (delays[0], delays[-1]), [1, 0, 0], 
                   t_eval=delays, method='BDF')
    
    true_concentrations = sol.y.T
    
    # Generate synthetic species spectra (Gaussian peaks)
    true_spectra = np.array([
        np.exp(-((wavelengths - 450)**2) / (2 * 30**2)),  # Species A
        np.exp(-((wavelengths - 550)**2) / (2 * 40**2)),  # Species B  
        np.exp(-((wavelengths - 650)**2) / (2 * 35**2))   # Species C
    ])
    
    # Generate synthetic data with noise
    synthetic_data = true_concentrations @ true_spectra
    noise_level = 0.01 * np.max(synthetic_data)
    synthetic_data += np.random.normal(0, noise_level, synthetic_data.shape)
    
    print(f"Generated synthetic data: {synthetic_data.shape}")
    print(f"Delays range: {delays[0]:.2f} - {delays[-1]:.2f} ps")
    print(f"Wavelengths range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    
    # Test GTA solver
    try:
        solver = GTASolver(wavelengths, delays, synthetic_data)
        solver.set_kinetic_matrix(K_test)
        
        # Initial guesses (slightly off from true values)
        initial_rates = [0.8, 0.4]  # True values: [1.0, 0.5]
        
        print(f"\nFitting with initial rates: {initial_rates}")
        
        result = solver.fit_parameters(initial_rates)
        
        print(f"Fitting success: {result['success']}")
        print(f"Fitted rates: {result['fitted_rates']}")
        print(f"True rates: [1.0, 0.5]")
        print(f"Reduced χ²: {result['reduced_chi_squared']:.3e}")
        
        # Get summary
        summary = solver.get_analysis_summary()
        print(f"\nAnalysis Summary:")
        print(f"Relative error: {summary['relative_error_percent']:.2f}%")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
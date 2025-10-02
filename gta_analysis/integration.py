# -*- coding: utf-8 -*-
"""
GTA Integration Configuration

This module provides configuration and integration utilities to connect
the GTA (Global Target Analysis) module with the existing TAS MCR-ALS system.

Features:
- Configuration templates for GTA analysis
- Integration with existing batch processing
- Model variant support for kinetic models
- Naming convention compatibility
- Results export and import utilities
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class GTAConfig:
    """
    Configuration manager for GTA analysis
    
    Provides templates and validation for GTA analysis configurations,
    compatible with the existing analysis_tool configuration system.
    """
    
    def __init__(self):
        """Initialize GTA configuration manager"""
        self.model_templates = self._create_model_templates()
        self.default_settings = self._create_default_settings()
    
    def _create_model_templates(self) -> Dict[str, Dict]:
        """Create configuration templates for all kinetic models"""
        return {
            "sequential_stable": {
                "model_number": 2,
                "description": "Sequential A→B→C→...→Z (last species stable)",
                "min_species": 2,
                "reaction_equation": "A->B->C",
                "rate_constants_required": "n_species - 1",
                "pathway_type": "sequential",
                "display_name": "Sequential Model (Stable End Product)",
                "ascii_arrows": "A -> B -> C"
            },
            "sequential_to_ground": {
                "model_number": 1, 
                "description": "Sequential A→B→C→...→Z→0 (decay to ground state)",
                "min_species": 2,
                "reaction_equation": "A->B->C->v",
                "rate_constants_required": "n_species",
                "pathway_type": "sequential",
                "display_name": "Sequential Model (Ground State Decay)",
                "ascii_arrows": "A -> B -> C -> v"
            },
            "parallel_a_to_bc": {
                "model_number": 7,
                "description": "Parallel branching A→B; A→C",
                "species_count": 3,
                "reaction_equation": "A->B;A->C", 
                "rate_constants_required": 2,
                "pathway_type": "parallel",
                "display_name": "Parallel Branching from A",
                "ascii_arrows": "A -> B; A -> C"
            },
            "sequential_b_to_d_branch": {
                "model_number": 3,
                "description": "Sequential with B→D branching: A→B→C→D; B→D",
                "species_count": 4,
                "reaction_equation": "A->B->C->D;B->D",
                "rate_constants_required": 4,
                "pathway_type": "branching",
                "display_name": "Sequential with B→D Branch",
                "ascii_arrows": "A -> B -> C -> D; B -> D"
            },
            "sequential_b_to_e_branch": {
                "model_number": 4,
                "description": "Sequential with B→E branching: A→B→C→D→E; B→E",
                "species_count": 5,
                "reaction_equation": "A->B->C->D->E;B->E",
                "rate_constants_required": 5,
                "pathway_type": "branching", 
                "display_name": "Sequential with B→E Branch",
                "ascii_arrows": "A -> B -> C -> D -> E; B -> E"
            },
            "sequential_c_to_e_branch": {
                "model_number": 5,
                "description": "Sequential with C→E branching: A→B→C→D→E; C→E",
                "species_count": 5,
                "reaction_equation": "A->B->C->D->E;C->E",
                "rate_constants_required": 5,
                "pathway_type": "branching",
                "display_name": "Sequential with C→E Branch", 
                "ascii_arrows": "A -> B -> C -> D -> E; C -> E"
            },
            "sequential_c_to_f_branch": {
                "model_number": 6,
                "description": "Sequential with C→F branching: A→B→C→D→E→F; C→F",
                "species_count": 6,
                "reaction_equation": "A->B->C->D->E->F;C->F",
                "rate_constants_required": 6,
                "pathway_type": "branching",
                "display_name": "Sequential with C→F Branch",
                "ascii_arrows": "A -> B -> C -> D -> E -> F; C -> F"
            },
            "sequential_parallel_b_branch": {
                "model_number": 8,
                "description": "Sequential with parallel B branching: A→B; B→C; B→D",
                "species_count": 4,
                "reaction_equation": "A->B;B->C;B->D",
                "rate_constants_required": 3,
                "pathway_type": "branching",
                "display_name": "Sequential with Parallel B Branch",
                "ascii_arrows": "A -> B; B -> C; B -> D"
            }
        }
    
    def _create_default_settings(self) -> Dict[str, Any]:
        """Create default GTA analysis settings"""
        return {
            "ode_solver": {
                "method": "BDF",
                "rtol": 1e-8,
                "atol": 1e-10,
                "max_step": np.inf
            },
            "optimization": {
                "method": "leastsq",
                "max_nfev": 1000,
                "ftol": 1e-8,
                "xtol": 1e-8,
                "gtol": 1e-8
            },
            "parameter_bounds": {
                "rate_constant_min": 1e-6,
                "rate_constant_max": 1e6,
                "auto_bounds": True
            },
            "visualization": {
                "language": "zh",
                "style": "scientific",
                "save_plots": True,
                "plot_formats": ["png", "pdf"]
            },
            "output": {
                "save_concentration_profiles": True,
                "save_species_spectra": True,
                "save_residuals": True,
                "save_fit_report": True,
                "export_format": "csv"
            }
        }
    
    def create_gta_config(self, 
                         model_type: str,
                         rate_constants_initial: List[float],
                         species_count: Optional[int] = None,
                         custom_equation: Optional[str] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        Create GTA analysis configuration
        
        Parameters
        ----------
        model_type : str
            Model type from model_templates or 'custom'
        rate_constants_initial : List[float]
            Initial guesses for rate constants
        species_count : int, optional
            Number of species (for custom models)
        custom_equation : str, optional
            Custom reaction equation
        **kwargs
            Additional configuration options
            
        Returns
        -------
        config : Dict[str, Any]
            Complete GTA configuration dictionary
        """
        if model_type == 'custom':
            if custom_equation is None:
                raise ValueError("Custom equation required for custom model type")
            
            model_config = {
                "model_type": "custom",
                "reaction_equation": custom_equation,
                "species_count": species_count or len(rate_constants_initial) + 1,
                "display_name": f"Custom Model: {custom_equation}",
                "ascii_arrows": custom_equation.replace('->', ' -> ')
            }
        else:
            if model_type not in self.model_templates:
                raise ValueError(f"Unknown model type: {model_type}. Available: {list(self.model_templates.keys())}")
            
            model_config = self.model_templates[model_type].copy()
        
        # Build complete configuration
        config = {
            "model": model_config,
            "parameters": {
                "rate_constants_initial": rate_constants_initial,
                "parameter_bounds": kwargs.get("parameter_bounds", None),
                "fixed_parameters": kwargs.get("fixed_parameters", None)
            },
            "settings": self.default_settings.copy()
        }
        
        # Override default settings with kwargs
        for key, value in kwargs.items():
            if key in config["settings"]:
                if isinstance(config["settings"][key], dict) and isinstance(value, dict):
                    config["settings"][key].update(value)
                else:
                    config["settings"][key] = value
        
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate GTA configuration
        
        Parameters
        ----------
        config : Dict[str, Any]
            GTA configuration to validate
            
        Returns
        -------
        is_valid : bool
            True if configuration is valid
        errors : List[str]
            List of validation errors
        """
        errors = []
        
        # Check required sections
        required_sections = ["model", "parameters", "settings"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        if errors:
            return False, errors
        
        # Validate model section
        model = config["model"]
        if "model_type" in model and model["model_type"] != "custom":
            # Check if it's a known model type
            if model.get("model_number") not in range(1, 9):
                errors.append(f"Invalid model number: {model.get('model_number')}")
        
        # Validate parameters
        params = config["parameters"]
        rate_constants = params.get("rate_constants_initial", [])
        if not rate_constants:
            errors.append("No initial rate constants provided")
        elif not all(isinstance(k, (int, float)) and k > 0 for k in rate_constants):
            errors.append("Rate constants must be positive numbers")
        
        # Validate bounds if provided
        bounds = params.get("parameter_bounds")
        if bounds is not None:
            if len(bounds) != len(rate_constants):
                errors.append("Parameter bounds length must match rate constants length")
        
        return len(errors) == 0, errors
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get information about a specific model type"""
        if model_type not in self.model_templates:
            return {"error": f"Unknown model type: {model_type}"}
        
        return self.model_templates[model_type].copy()
    
    def list_available_models(self) -> Dict[str, str]:
        """Get list of all available model types with descriptions"""
        return {
            model_type: template["description"] 
            for model_type, template in self.model_templates.items()
        }
    
    def export_config(self, config: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """
        Export GTA configuration to JSON file
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to export
        file_path : str or Path
            Path to save configuration file
        """
        file_path = Path(file_path)
        
        # Convert numpy arrays to lists for JSON serialization
        config_serializable = self._make_json_serializable(config)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_serializable, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported GTA configuration to {file_path}")
    
    def import_config(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Import GTA configuration from JSON file
        
        Parameters
        ----------
        file_path : str or Path
            Path to configuration file
            
        Returns
        -------
        config : Dict[str, Any]
            Imported configuration
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"Imported GTA configuration from {file_path}")
        return config
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj


class GTAIntegration:
    """
    Integration utilities for connecting GTA with existing TAS analysis system
    
    Provides compatibility with analysis_tool configuration and naming conventions.
    """
    
    def __init__(self, config_manager: Optional[GTAConfig] = None):
        """Initialize GTA integration utilities"""
        self.config_manager = config_manager or GTAConfig()
    
    def convert_to_globalfit_config(self, gta_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert GTA configuration to globalfit-compatible format
        
        Parameters
        ----------
        gta_config : Dict[str, Any]
            GTA configuration dictionary
            
        Returns
        -------
        globalfit_config : Dict[str, Any]
            Configuration compatible with existing globalfit system
        """
        model = gta_config["model"]
        params = gta_config["parameters"]
        
        # Map GTA model to globalfit naming convention
        if model.get("model_type") == "custom":
            model_name = "custom_gta"
            display_name = model.get("display_name", "Custom GTA Model")
        else:
            # Use existing naming system from analysis_tool
            pathway_type = model.get("pathway_type", "sequential")
            model_name = f"gta_{pathway_type}"
            display_name = model.get("display_name", f"GTA {pathway_type.title()} Model")
        
        globalfit_config = {
            "model_name": model_name,
            "display_name": display_name,
            "model_type": "kinetic",
            "analysis_method": "gta",
            "parameters": {
                "rate_constants": params["rate_constants_initial"],
                "bounds": params.get("parameter_bounds"),
                "fixed": params.get("fixed_parameters")
            },
            "settings": gta_config["settings"],
            "variants": self._generate_model_variants(gta_config)
        }
        
        return globalfit_config
    
    def _generate_model_variants(self, gta_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate model variants for different parameter sets"""
        model = gta_config["model"]
        base_rates = gta_config["parameters"]["rate_constants_initial"]
        
        variants = []
        
        # Default variant
        variants.append({
            "name": "default",
            "display_name": "Default Parameters",
            "parameters": base_rates,
            "description": "Default initial parameter values"
        })
        
        # Fast kinetics variant
        fast_rates = [r * 2 for r in base_rates]
        variants.append({
            "name": "fast",
            "display_name": "Fast Kinetics",
            "parameters": fast_rates,
            "description": "2x faster rate constants"
        })
        
        # Slow kinetics variant
        slow_rates = [r * 0.5 for r in base_rates]
        variants.append({
            "name": "slow", 
            "display_name": "Slow Kinetics",
            "parameters": slow_rates,
            "description": "2x slower rate constants"
        })
        
        return variants
    
    def create_batch_config(self, 
                           gta_configs: List[Dict[str, Any]],
                           output_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create batch processing configuration for multiple GTA models
        
        Parameters
        ----------
        gta_configs : List[Dict[str, Any]]
            List of GTA configurations
        output_config : Dict[str, Any]
            Output configuration (directories, formats, etc.)
            
        Returns
        -------
        batch_config : Dict[str, Any]
            Batch processing configuration
        """
        batch_config = {
            "analysis_type": "gta_batch",
            "models": [],
            "output": output_config,
            "parallel": {
                "enabled": True,
                "max_workers": 4
            }
        }
        
        for i, gta_config in enumerate(gta_configs):
            model_entry = {
                "id": f"gta_model_{i+1}",
                "config": gta_config,
                "globalfit_config": self.convert_to_globalfit_config(gta_config)
            }
            batch_config["models"].append(model_entry)
        
        return batch_config
    
    def generate_result_naming(self, model_config: Dict[str, Any], 
                              variant: Optional[str] = None) -> str:
        """
        Generate result directory name following project conventions
        
        Parameters
        ----------
        model_config : Dict[str, Any]
            Model configuration
        variant : str, optional
            Variant name
            
        Returns
        -------
        result_name : str
            Formatted result directory name
        """
        model = model_config["model"]
        
        if model.get("model_type") == "custom":
            # For custom models, use sanitized equation
            equation = model.get("reaction_equation", "custom")
            base_name = "custom_" + equation.replace("->", "_to_").replace(";", "_").replace("v", "ground")
        else:
            # Use pathway type for predefined models  
            pathway_type = model.get("pathway_type", "sequential")
            model_num = model.get("model_number", 1)
            base_name = f"gta_{pathway_type}_model_{model_num}"
        
        # Add variant if specified
        if variant:
            base_name += f"_{variant}"
        
        # Ensure ASCII-safe naming
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in base_name)
        
        return safe_name
    
    def export_results_to_globalfit_format(self, 
                                          gta_solver,
                                          fit_result: Dict[str, Any],
                                          output_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Export GTA results in globalfit-compatible format
        
        Parameters
        ----------
        gta_solver : GTASolver
            GTA solver with results
        fit_result : Dict[str, Any]
            Fit result dictionary
        output_dir : str or Path
            Output directory
            
        Returns
        -------
        exported_files : Dict[str, str]
            Dictionary of exported file types and paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Export concentration profiles
        if gta_solver.concentration_profiles is not None:
            conc_file = output_dir / "concentration_profiles.csv"
            np.savetxt(conc_file, gta_solver.concentration_profiles, 
                      delimiter=',', header='Concentration profiles (delays x species)')
            exported_files["concentration_profiles"] = str(conc_file)
        
        # Export species spectra
        if gta_solver.species_spectra is not None:
            spectra_file = output_dir / "species_spectra.csv"
            np.savetxt(spectra_file, gta_solver.species_spectra,
                      delimiter=',', header='Species-associated spectra (species x wavelengths)')
            exported_files["species_spectra"] = str(spectra_file)
        
        # Export fitted parameters
        params_file = output_dir / "fitted_parameters.json"
        params_data = {
            "fitted_rate_constants": fit_result["fitted_rates"],
            "parameter_errors": fit_result["parameter_errors"],
            "chi_squared": fit_result["chi_squared"],
            "reduced_chi_squared": fit_result["reduced_chi_squared"],
            "n_data_points": fit_result["n_data_points"],
            "n_parameters": fit_result["n_parameters"]
        }
        
        with open(params_file, 'w') as f:
            json.dump(params_data, f, indent=2, default=str)
        exported_files["parameters"] = str(params_file)
        
        # Export fit report
        if gta_solver.fit_report_text:
            report_file = output_dir / "fit_report.txt"
            with open(report_file, 'w') as f:
                f.write(gta_solver.fit_report_text)
            exported_files["fit_report"] = str(report_file)
        
        logger.info(f"Exported GTA results to {output_dir}")
        return exported_files


# Convenience functions for integration
def create_gta_config_from_template(model_type: str, 
                                   rate_constants: List[float],
                                   **kwargs) -> Dict[str, Any]:
    """
    Quick function to create GTA configuration from template
    
    Parameters
    ----------
    model_type : str
        Model type identifier
    rate_constants : List[float]
        Initial rate constant guesses
    **kwargs
        Additional configuration options
        
    Returns
    -------
    config : Dict[str, Any]
        GTA configuration dictionary
    """
    config_manager = GTAConfig()
    return config_manager.create_gta_config(model_type, rate_constants, **kwargs)


def integrate_gta_with_globalfit(gta_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quick function to convert GTA config to globalfit format
    
    Parameters
    ----------
    gta_config : Dict[str, Any]
        GTA configuration
        
    Returns
    -------
    globalfit_config : Dict[str, Any]
        Globalfit-compatible configuration
    """
    integration = GTAIntegration()
    return integration.convert_to_globalfit_config(gta_config)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("Testing GTA Integration Module")
    print("=" * 40)
    
    # Create configuration manager
    config_manager = GTAConfig()
    
    print("Available models:")
    models = config_manager.list_available_models()
    for model_type, description in models.items():
        print(f"  {model_type}: {description}")
    
    # Test configuration creation
    print("\nTesting configuration creation...")
    
    try:
        # Test sequential model
        config = config_manager.create_gta_config(
            model_type="sequential_stable",
            rate_constants_initial=[1.0, 0.5, 0.1],
            parameter_bounds=[(0.1, 10.0), (0.01, 5.0), (0.001, 1.0)]
        )
        
        print("Created sequential model config:")
        print(f"  Model: {config['model']['display_name']}")
        print(f"  Rate constants: {config['parameters']['rate_constants_initial']}")
        
        # Validate configuration
        is_valid, errors = config_manager.validate_config(config)
        print(f"  Valid: {is_valid}")
        if errors:
            print(f"  Errors: {errors}")
        
        # Test integration
        print("\nTesting globalfit integration...")
        integration = GTAIntegration(config_manager)
        globalfit_config = integration.convert_to_globalfit_config(config)
        
        print(f"  Globalfit model name: {globalfit_config['model_name']}")
        print(f"  Display name: {globalfit_config['display_name']}")
        print(f"  Variants: {len(globalfit_config['variants'])}")
        
        # Test result naming
        result_name = integration.generate_result_naming(config)
        print(f"  Result directory name: {result_name}")
        
        print("\nGTA integration test completed successfully!")
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
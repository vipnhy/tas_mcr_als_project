# -*- coding: utf-8 -*-
"""
GTA Analysis Example Script

This script demonstrates how to use the GTA (Global Target Analysis) module
for TAS data analysis, following the design principles from EfsTA.

Examples include:
1. Using predefined kinetic models (Models 1-8)
2. Creating custom kinetic models with reaction equations
3. Parameter fitting and optimization
4. Results visualization and export
5. Integration with existing TAS analysis workflow

Usage:
    python gta_example.py [--model MODEL_TYPE] [--data DATA_FILE]
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import argparse

# Import GTA modules
from gta_analysis import (
    KineticModels, ReactionParser, GTASolver, GTAVisualization,
    GTAConfig, GTAIntegration, create_gta_config_from_template
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_delays=50, n_wavelengths=100, noise_level=0.01):
    """Generate synthetic TAS data for testing"""
    logger.info("Generating synthetic TAS data...")
    
    # Time and wavelength arrays
    delays = np.logspace(-1, 3, n_delays)  # 0.1 to 1000 ps
    wavelengths = np.linspace(400, 700, n_wavelengths)  # 400-700 nm
    
    # True kinetic parameters (A -> B -> C model)
    k1_true = 1.0  # ps^-1, A -> B
    k2_true = 0.5  # ps^-1, B -> C
    
    # Generate true concentration profiles
    conc_A = np.exp(-k1_true * delays)
    conc_B = k1_true / (k2_true - k1_true) * (np.exp(-k1_true * delays) - np.exp(-k2_true * delays))
    conc_C = 1 - conc_A - conc_B
    
    true_concentrations = np.column_stack([conc_A, conc_B, conc_C])
    
    # True species spectra (Gaussian peaks at different wavelengths)
    spec_A = np.exp(-((wavelengths - 450)**2) / (2 * 30**2))  # Blue species
    spec_B = np.exp(-((wavelengths - 550)**2) / (2 * 40**2))  # Green species  
    spec_C = np.exp(-((wavelengths - 650)**2) / (2 * 35**2))  # Red species
    
    true_spectra = np.array([spec_A, spec_B, spec_C])
    
    # Generate experimental data
    experimental_data = true_concentrations @ true_spectra
    
    # Add noise
    noise = np.random.normal(0, noise_level * np.max(experimental_data), experimental_data.shape)
    experimental_data += noise
    
    logger.info(f"Generated data: {experimental_data.shape} (delays x wavelengths)")
    logger.info(f"True rate constants: k1={k1_true:.2f}, k2={k2_true:.2f} ps^-1")
    
    return {
        'wavelengths': wavelengths,
        'delays': delays,
        'data': experimental_data,
        'true_concentrations': true_concentrations,
        'true_spectra': true_spectra,
        'true_rates': [k1_true, k2_true]
    }


def example_predefined_model():
    """Example using predefined kinetic model"""
    logger.info("=" * 50)
    logger.info("Example 1: Predefined Kinetic Model (Model 2 - Sequential)")
    logger.info("=" * 50)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data()
    
    # Create kinetic model (Model 2: A -> B -> C, stable end product)
    rate_constants = [0.8, 0.4]  # Initial guesses (slightly off from true values)
    kinetic_model = KineticModels(rate_constants)
    
    # Get kinetic matrix
    K, n_species = kinetic_model.get_kinetic_matrix(2)
    
    logger.info(f"Using Model 2: {kinetic_model.get_model_info(2)['description']}")
    logger.info(f"Kinetic matrix shape: {K.shape}")
    logger.info(f"Number of species: {n_species}")
    
    # Initialize GTA solver
    solver = GTASolver(
        synthetic_data['wavelengths'],
        synthetic_data['delays'], 
        synthetic_data['data']
    )
    
    # Set kinetic matrix
    solver.set_kinetic_matrix(K)
    
    # Fit parameters
    logger.info("Starting parameter fitting...")
    fit_result = solver.fit_parameters(
        initial_rates=rate_constants,
        parameter_bounds=[(0.1, 10.0), (0.01, 5.0)],
        method='leastsq'
    )
    
    # Display results
    logger.info(f"Fitting success: {fit_result['success']}")
    logger.info(f"Fitted rate constants: {fit_result['fitted_rates']}")
    logger.info(f"True rate constants: {synthetic_data['true_rates']}")
    logger.info(f"Reduced χ²: {fit_result['reduced_chi_squared']:.3e}")
    
    # Visualization
    viz = GTAVisualization(language='zh')
    
    # Plot Species-Associated Spectra
    fig_sas = viz.plot_species_spectra(
        synthetic_data['wavelengths'],
        solver.species_spectra,
        title="预定义模型 - 物种关联光谱"
    )
    plt.show()
    
    # Plot concentration profiles
    fig_conc = viz.plot_concentration_profiles(
        synthetic_data['delays'],
        solver.concentration_profiles,
        title="预定义模型 - 浓度时间演化"
    )
    plt.show()
    
    return solver, fit_result


def example_custom_model():
    """Example using custom reaction equation"""
    logger.info("=" * 50)
    logger.info("Example 2: Custom Kinetic Model (Reaction Equation)")
    logger.info("=" * 50)
    
    # Generate synthetic data for branching model
    synthetic_data = generate_synthetic_data()
    
    # Define custom reaction equation
    reaction_equation = "A->B->C;B->D"  # Sequential with branching
    rate_constants = [1.0, 0.5, 0.3, 0.2]  # k1, k2, k3, k4
    
    logger.info(f"Custom reaction equation: {reaction_equation}")
    logger.info(f"Initial rate constants: {rate_constants}")
    
    # Parse reaction equation
    parser = ReactionParser()
    K, n_species, parse_info = parser.parse_reaction_equation(reaction_equation, rate_constants)
    
    logger.info(f"Parsed reaction info:")
    logger.info(f"  Species: {parse_info['species']}")
    logger.info(f"  Number of species: {parse_info['n_species']}")
    logger.info(f"  Number of reactions: {parse_info['n_reactions']}")
    logger.info(f"  Pathway type: {parse_info['pathway_type']}")
    
    # Since our synthetic data is 3-species but custom model is 4-species,
    # we'll modify the data slightly or use a simpler custom model
    simple_equation = "A->B->C"
    simple_rates = [0.8, 0.4]
    
    K_simple, n_simple, info_simple = parser.parse_reaction_equation(simple_equation, simple_rates)
    
    # Initialize GTA solver with custom model
    solver = GTASolver(
        synthetic_data['wavelengths'],
        synthetic_data['delays'],
        synthetic_data['data']
    )
    
    solver.set_kinetic_matrix(K_simple)
    
    # Fit parameters
    logger.info("Fitting custom model parameters...")
    fit_result = solver.fit_parameters(
        initial_rates=simple_rates,
        method='leastsq'
    )
    
    # Display results
    logger.info(f"Custom model fitting success: {fit_result['success']}")
    logger.info(f"Fitted rate constants: {fit_result['fitted_rates']}")
    logger.info(f"Reduced χ²: {fit_result['reduced_chi_squared']:.3e}")
    
    # Visualization
    viz = GTAVisualization(language='zh')
    
    # Plot residuals
    fig_res = viz.plot_residuals_2d(
        synthetic_data['wavelengths'],
        synthetic_data['delays'],
        solver.residuals,
        title="自定义模型 - 拟合残差"
    )
    plt.show()
    
    return solver, fit_result


def example_configuration_system():
    """Example using configuration system for integration"""
    logger.info("=" * 50)
    logger.info("Example 3: Configuration System Integration")
    logger.info("=" * 50)
    
    # Create GTA configuration
    config = create_gta_config_from_template(
        model_type="sequential_stable",
        rate_constants=[1.0, 0.5, 0.1],
        parameter_bounds=[(0.1, 10.0), (0.01, 5.0), (0.001, 1.0)],
        visualization={'language': 'zh', 'style': 'scientific'}
    )
    
    logger.info("Created GTA configuration:")
    logger.info(f"  Model: {config['model']['display_name']}")
    logger.info(f"  Parameters: {config['parameters']['rate_constants_initial']}")
    
    # Validate configuration
    config_manager = GTAConfig()
    is_valid, errors = config_manager.validate_config(config)
    logger.info(f"  Configuration valid: {is_valid}")
    if errors:
        logger.warning(f"  Validation errors: {errors}")
    
    # Convert to globalfit format
    integration = GTAIntegration()
    globalfit_config = integration.convert_to_globalfit_config(config)
    
    logger.info("Converted to globalfit format:")
    logger.info(f"  Model name: {globalfit_config['model_name']}")
    logger.info(f"  Analysis method: {globalfit_config['analysis_method']}")
    logger.info(f"  Variants available: {len(globalfit_config['variants'])}")
    
    for variant in globalfit_config['variants']:
        logger.info(f"    - {variant['name']}: {variant['display_name']}")
    
    # Generate result naming
    result_name = integration.generate_result_naming(config)
    logger.info(f"  Result directory name: {result_name}")
    
    return config, globalfit_config


def example_batch_analysis():
    """Example of batch analysis with multiple models"""
    logger.info("=" * 50)
    logger.info("Example 4: Batch Analysis with Multiple Models")
    logger.info("=" * 50)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data()
    
    # Define multiple models to test
    model_configs = [
        {
            'name': 'sequential_stable',
            'type': 'predefined',
            'model_number': 2,
            'rates': [0.8, 0.4]
        },
        {
            'name': 'parallel_branching',
            'type': 'predefined', 
            'model_number': 7,
            'rates': [1.5, 0.7]  # This won't fit well, just for demonstration
        },
        {
            'name': 'custom_sequential',
            'type': 'custom',
            'equation': 'A->B->C',
            'rates': [0.9, 0.3]
        }
    ]
    
    results = []
    
    for model_config in model_configs:
        logger.info(f"Testing model: {model_config['name']}")
        
        try:
            # Create appropriate kinetic matrix
            if model_config['type'] == 'predefined':
                kinetic_model = KineticModels(model_config['rates'])
                K, n_species = kinetic_model.get_kinetic_matrix(model_config['model_number'])
            else:  # custom
                parser = ReactionParser()
                K, n_species, _ = parser.parse_reaction_equation(
                    model_config['equation'], 
                    model_config['rates']
                )
            
            # Run GTA analysis
            solver = GTASolver(
                synthetic_data['wavelengths'],
                synthetic_data['delays'],
                synthetic_data['data']
            )
            solver.set_kinetic_matrix(K)
            
            fit_result = solver.fit_parameters(
                initial_rates=model_config['rates'],
                method='leastsq'
            )
            
            # Store results
            results.append({
                'model_name': model_config['name'],
                'success': fit_result['success'],
                'fitted_rates': fit_result['fitted_rates'],
                'reduced_chi_squared': fit_result['reduced_chi_squared'],
                'solver': solver
            })
            
            logger.info(f"  Success: {fit_result['success']}")
            logger.info(f"  Reduced χ²: {fit_result['reduced_chi_squared']:.3e}")
            
        except Exception as e:
            logger.warning(f"  Failed: {e}")
            results.append({
                'model_name': model_config['name'],
                'success': False,
                'error': str(e)
            })
    
    # Compare results
    logger.info("\nBatch Analysis Results Summary:")
    logger.info("-" * 40)
    
    successful_results = [r for r in results if r.get('success', False)]
    if successful_results:
        best_model = min(successful_results, key=lambda x: x['reduced_chi_squared'])
        
        logger.info("Model comparison (by reduced χ²):")
        for result in sorted(successful_results, key=lambda x: x['reduced_chi_squared']):
            logger.info(f"  {result['model_name']}: χ² = {result['reduced_chi_squared']:.3e}")
        
        logger.info(f"\nBest model: {best_model['model_name']}")
        
        # Visualize best model results
        viz = GTAVisualization(language='zh')
        fig_quality = viz.plot_fit_quality_summary(
            {'fitted_rates': best_model['fitted_rates'],
             'parameter_errors': [None] * len(best_model['fitted_rates']),
             'chi_squared': best_model['reduced_chi_squared'] * synthetic_data['data'].size,
             'reduced_chi_squared': best_model['reduced_chi_squared'],
             'residuals': best_model['solver'].residuals}
        )
        plt.show()
    
    return results


def main():
    """Main function to run GTA examples"""
    parser = argparse.ArgumentParser(description='GTA Analysis Examples')
    parser.add_argument('--example', type=int, default=0, choices=[0, 1, 2, 3, 4],
                       help='Example to run (0=all, 1=predefined, 2=custom, 3=config, 4=batch)')
    parser.add_argument('--output-dir', type=str, default='gta_example_output',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Starting GTA Analysis Examples")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        if args.example == 0 or args.example == 1:
            solver1, result1 = example_predefined_model()
            
        if args.example == 0 or args.example == 2:
            solver2, result2 = example_custom_model()
            
        if args.example == 0 or args.example == 3:
            config, globalfit_config = example_configuration_system()
            
        if args.example == 0 or args.example == 4:
            batch_results = example_batch_analysis()
        
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
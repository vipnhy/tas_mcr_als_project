# -*- coding: utf-8 -*-
"""
Global Target Analysis (GTA) Module for TAS Data Analysis

This module provides comprehensive Global Target Analysis capabilities for 
transient absorption spectroscopy data, based on the design principles from EfsTA.

Core Components:
- Kinetic Models: Predefined and custom kinetic models for various reaction pathways
- Reaction Parser: Converts reaction equations to kinetic matrices  
- ODE Solver: Solves differential equations for species concentration evolution
- Parameter Optimization: Fits kinetic parameters using advanced algorithms
- Results Visualization: Species-associated spectra and concentration profiles

Author: TAS MCR-ALS Project
"""

# Core modules
from .kinetic_models import KineticModels, create_kinetic_model
from .reaction_parser import ReactionParser, parse_reaction, validate_reaction
from .gta_solver import GTASolver, run_gta_analysis
from .visualization import GTAVisualization
from .integration import GTAConfig, GTAIntegration, create_gta_config_from_template

__version__ = "1.0.0"
__all__ = [
    "KineticModels", "create_kinetic_model",
    "ReactionParser", "parse_reaction", "validate_reaction", 
    "GTASolver", "run_gta_analysis",
    "GTAVisualization",
    "GTAConfig", "GTAIntegration", "create_gta_config_from_template"
]
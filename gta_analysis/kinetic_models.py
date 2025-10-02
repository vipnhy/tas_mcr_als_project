# -*- coding: utf-8 -*-
"""
Kinetic Models Module for Global Target Analysis

This module implements predefined kinetic models based on EfsTA's design:
- 8 predefined kinetic models covering common reaction mechanisms
- Support for sequential, parallel, and branching pathways
- Automatic kinetic matrix generation
- Species concentration evolution calculations

Models implemented:
1. Sequential A→B→C→...→Z→0 (last species decays to ground state)
2. Sequential A→B→C→...→Z (last species stable)
3-8. Various branching models with specific connectivity patterns
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class KineticModels:
    """
    Predefined kinetic models for Global Target Analysis
    
    Based on EfsTA's 8 predefined models with additional enhancements
    for TAS analysis workflows.
    """
    
    def __init__(self, rate_constants: List[float]):
        """
        Initialize kinetic models with rate constants
        
        Parameters
        ----------
        rate_constants : List[float]
            List of reaction rate constants (k values, not lifetimes)
        """
        self.k = np.array(rate_constants)
        self.models_info = self._get_models_info()
    
    def _get_models_info(self) -> Dict[int, Dict]:
        """Get information about all available models"""
        return {
            1: {
                "name": "Sequential with ground state decay",
                "description": "A → B → C → ... → Z → 0",
                "min_species": 2,
                "pathway": "sequential_to_ground"
            },
            2: {
                "name": "Sequential without ground state decay", 
                "description": "A → B → C → ... → Z",
                "min_species": 2,
                "pathway": "sequential_stable"
            },
            3: {
                "name": "Sequential with B→D branching",
                "description": "A → B → C → D; B → D", 
                "species": 4,
                "pathway": "sequential_b_to_d_branch"
            },
            4: {
                "name": "Sequential with B→E branching",
                "description": "A → B → C → D → E; B → E",
                "species": 5, 
                "pathway": "sequential_b_to_e_branch"
            },
            5: {
                "name": "Sequential with C→E branching",
                "description": "A → B → C → D → E; C → E",
                "species": 5,
                "pathway": "sequential_c_to_e_branch"
            },
            6: {
                "name": "Sequential with C→F branching", 
                "description": "A → B → C → D → E → F; C → F",
                "species": 6,
                "pathway": "sequential_c_to_f_branch"
            },
            7: {
                "name": "Parallel branching from A",
                "description": "A → B; A → C",
                "species": 3,
                "pathway": "parallel_a_to_bc"
            },
            8: {
                "name": "Sequential with parallel B branching",
                "description": "A → B; B → C; B → D", 
                "species": 4,
                "pathway": "sequential_parallel_b_branch"
            }
        }
    
    def get_model_info(self, model_number: int) -> Dict:
        """Get information about a specific model"""
        if model_number not in self.models_info:
            raise ValueError(f"Model {model_number} not available. Choose from 1-8.")
        return self.models_info[model_number]
    
    def list_models(self) -> None:
        """Print information about all available models"""
        print("Available Kinetic Models:")
        print("=" * 50)
        for num, info in self.models_info.items():
            print(f"Model {num}: {info['name']}")
            print(f"  Description: {info['description']}")
            if 'species' in info:
                print(f"  Species count: {info['species']}")
            else:
                print(f"  Min species: {info['min_species']}")
            print(f"  Pathway type: {info['pathway']}")
            print()
    
    def get_kinetic_matrix(self, model_number: int) -> Tuple[np.ndarray, int]:
        """
        Generate kinetic matrix for specified model
        
        Parameters
        ---------- 
        model_number : int
            Model number (1-8)
            
        Returns
        -------
        K : np.ndarray
            Kinetic matrix for the model
        n_species : int
            Number of species in the model
        """
        if model_number == 1:
            return self._model_1()
        elif model_number == 2:
            return self._model_2()
        elif model_number == 3:
            return self._model_3()
        elif model_number == 4:
            return self._model_4()
        elif model_number == 5:
            return self._model_5()
        elif model_number == 6:
            return self._model_6()
        elif model_number == 7:
            return self._model_7()
        elif model_number == 8:
            return self._model_8()
        else:
            raise ValueError(f"Model {model_number} not implemented. Choose from 1-8.")
    
    def _model_1(self) -> Tuple[np.ndarray, int]:
        """
        Model 1: A → B → C → ... → Z → 0
        Sequential model where last species decays to ground state
        """
        n_species = len(self.k)
        K = np.zeros((n_species, n_species))
        
        # Main diagonal: -k values (loss terms)
        np.fill_diagonal(K, -self.k)
        
        # Lower diagonal: +k values (gain terms) 
        for i in range(n_species - 1):
            K[i + 1, i] = self.k[i]
            
        logger.info(f"Generated Model 1 matrix for {n_species} species")
        return K, n_species
    
    def _model_2(self) -> Tuple[np.ndarray, int]:
        """
        Model 2: A → B → C → ... → Z  
        Sequential model where last species is stable
        """
        n_species = len(self.k)
        K = np.zeros((n_species, n_species))
        
        # Main diagonal: -k values except last species
        for i in range(n_species - 1):
            K[i, i] = -self.k[i]
            
        # Lower diagonal: +k values
        for i in range(n_species - 1):
            K[i + 1, i] = self.k[i]
            
        logger.info(f"Generated Model 2 matrix for {n_species} species")
        return K, n_species
    
    def _model_3(self) -> Tuple[np.ndarray, int]:
        """
        Model 3: A → B → C → D; B → D
        Sequential with branching from B to D
        Requires 4 rate constants
        """
        if len(self.k) != 4:
            raise ValueError("Model 3 requires exactly 4 rate constants")
            
        K = np.zeros((4, 4))
        # A: k0 (A→B)
        K[0, 0] = -self.k[0]
        K[1, 0] = self.k[0]
        
        # B: k1 (B→C) + k3 (B→D)  
        K[1, 1] = -(self.k[1] + self.k[3])
        K[2, 1] = self.k[1]  # B→C
        K[3, 1] = self.k[3]  # B→D
        
        # C: k2 (C→D)
        K[2, 2] = -self.k[2]
        K[3, 2] = self.k[2]
        
        # D: stable
        K[3, 3] = 0
        
        logger.info("Generated Model 3 matrix (A→B→C→D; B→D)")
        return K, 4
    
    def _model_4(self) -> Tuple[np.ndarray, int]:
        """
        Model 4: A → B → C → D → E; B → E
        Sequential with branching from B to E
        Requires 5 rate constants
        """
        if len(self.k) != 5:
            raise ValueError("Model 4 requires exactly 5 rate constants")
            
        K = np.zeros((5, 5))
        # A: k0 (A→B)
        K[0, 0] = -self.k[0]
        K[1, 0] = self.k[0]
        
        # B: k1 (B→C) + k4 (B→E)
        K[1, 1] = -(self.k[1] + self.k[4])
        K[2, 1] = self.k[1]  # B→C
        K[4, 1] = self.k[4]  # B→E
        
        # C: k2 (C→D)
        K[2, 2] = -self.k[2]
        K[3, 2] = self.k[2]
        
        # D: k3 (D→E)
        K[3, 3] = -self.k[3]
        K[4, 3] = self.k[3]
        
        # E: stable
        K[4, 4] = 0
        
        logger.info("Generated Model 4 matrix (A→B→C→D→E; B→E)")
        return K, 5
    
    def _model_5(self) -> Tuple[np.ndarray, int]:
        """
        Model 5: A → B → C → D → E; C → E
        Sequential with branching from C to E
        Requires 5 rate constants
        """
        if len(self.k) != 5:
            raise ValueError("Model 5 requires exactly 5 rate constants")
            
        K = np.zeros((5, 5))
        # A: k0 (A→B)
        K[0, 0] = -self.k[0]
        K[1, 0] = self.k[0]
        
        # B: k1 (B→C)
        K[1, 1] = -self.k[1]
        K[2, 1] = self.k[1]
        
        # C: k2 (C→D) + k4 (C→E)
        K[2, 2] = -(self.k[2] + self.k[4])
        K[3, 2] = self.k[2]  # C→D
        K[4, 2] = self.k[4]  # C→E
        
        # D: k3 (D→E)
        K[3, 3] = -self.k[3]
        K[4, 3] = self.k[3]
        
        # E: stable
        K[4, 4] = 0
        
        logger.info("Generated Model 5 matrix (A→B→C→D→E; C→E)")
        return K, 5
    
    def _model_6(self) -> Tuple[np.ndarray, int]:
        """
        Model 6: A → B → C → D → E → F; C → F
        Sequential with branching from C to F
        Requires 6 rate constants
        """
        if len(self.k) != 6:
            raise ValueError("Model 6 requires exactly 6 rate constants")
            
        K = np.zeros((6, 6))
        # A: k0 (A→B)
        K[0, 0] = -self.k[0]
        K[1, 0] = self.k[0]
        
        # B: k1 (B→C)
        K[1, 1] = -self.k[1]
        K[2, 1] = self.k[1]
        
        # C: k2 (C→D) + k5 (C→F)
        K[2, 2] = -(self.k[2] + self.k[5])
        K[3, 2] = self.k[2]  # C→D
        K[5, 2] = self.k[5]  # C→F
        
        # D: k3 (D→E)
        K[3, 3] = -self.k[3]
        K[4, 3] = self.k[3]
        
        # E: k4 (E→F)
        K[4, 4] = -self.k[4]
        K[5, 4] = self.k[4]
        
        # F: stable
        K[5, 5] = 0
        
        logger.info("Generated Model 6 matrix (A→B→C→D→E→F; C→F)")
        return K, 6
    
    def _model_7(self) -> Tuple[np.ndarray, int]:
        """
        Model 7: A → B; A → C
        Parallel branching from A
        Requires 2 rate constants
        """
        if len(self.k) != 2:
            raise ValueError("Model 7 requires exactly 2 rate constants")
            
        K = np.zeros((3, 3))
        # A: k0 (A→B) + k1 (A→C)
        K[0, 0] = -(self.k[0] + self.k[1])
        K[1, 0] = self.k[0]  # A→B
        K[2, 0] = self.k[1]  # A→C
        
        # B and C: stable
        K[1, 1] = 0
        K[2, 2] = 0
        
        logger.info("Generated Model 7 matrix (A→B; A→C)")
        return K, 3
    
    def _model_8(self) -> Tuple[np.ndarray, int]:
        """
        Model 8: A → B; B → C; B → D
        Sequential with parallel branching from B
        Requires 3 rate constants
        """
        if len(self.k) != 3:
            raise ValueError("Model 8 requires exactly 3 rate constants")
            
        K = np.zeros((4, 4))
        # A: k0 (A→B)
        K[0, 0] = -self.k[0]
        K[1, 0] = self.k[0]
        
        # B: k1 (B→C) + k2 (B→D)
        K[1, 1] = -(self.k[1] + self.k[2])
        K[2, 1] = self.k[1]  # B→C
        K[3, 1] = self.k[2]  # B→D
        
        # C and D: stable
        K[2, 2] = 0
        K[3, 3] = 0
        
        logger.info("Generated Model 8 matrix (A→B; B→C; B→D)")
        return K, 4
    
    def validate_model(self, model_number: int) -> bool:
        """
        Validate if the model can be generated with current rate constants
        
        Parameters
        ----------
        model_number : int
            Model number to validate
            
        Returns
        -------
        bool
            True if model is valid with current rate constants
        """
        try:
            K, n = self.get_kinetic_matrix(model_number)
            return True
        except ValueError as e:
            logger.warning(f"Model {model_number} validation failed: {e}")
            return False
    
    def get_pathway_description(self, model_number: int) -> str:
        """Get descriptive pathway name for integration with existing naming system"""
        model_info = self.get_model_info(model_number)
        return model_info['pathway']


# Convenience function for quick model generation
def create_kinetic_model(model_number: int, rate_constants: List[float]) -> Tuple[np.ndarray, int]:
    """
    Quick function to create kinetic matrix for a specific model
    
    Parameters
    ----------
    model_number : int
        Model number (1-8)
    rate_constants : List[float] 
        List of rate constants for the model
        
    Returns
    -------
    K : np.ndarray
        Kinetic matrix
    n_species : int
        Number of species
    """
    model = KineticModels(rate_constants)
    return model.get_kinetic_matrix(model_number)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Test Model 1 (sequential to ground state)
    k_values = [1.0, 0.5, 0.1]  # Rate constants in ps^-1
    model = KineticModels(k_values)
    
    print("Testing Kinetic Models Module")
    print("=" * 40)
    
    model.list_models()
    
    # Test Model 3 (branching model)
    k_branching = [2.0, 1.0, 0.5, 0.3]  # 4 rate constants for Model 3
    model_branching = KineticModels(k_branching)
    
    try:
        K3, n3 = model_branching.get_kinetic_matrix(3)
        print(f"Model 3 Kinetic Matrix ({n3} species):")
        print(K3)
        print()
    except Exception as e:
        print(f"Error testing Model 3: {e}")
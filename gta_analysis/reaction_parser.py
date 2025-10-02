# -*- coding: utf-8 -*-
"""
Reaction Parser Module for Custom Kinetic Models

This module provides functionality to parse reaction equations and convert them
to kinetic matrices, following EfsTA's custom model design principles.

Supported syntax:
- Arrow notation: "A->B->C->v" or "A→B→C→v" 
- No-arrow notation: "ABCv"
- Branching with semicolons: "A->B->C->v;B->A;A->C"
- Void transitions: species "v" represents decay to ground state
- Species naming: A-Z (max 26 species)

Examples:
- "A->B->C->v" : Sequential A→B→C→ground
- "A->B;A->C" : Parallel branching from A
- "ABCv;BA;AC" : Same as above in compact notation
"""

import numpy as np
import re
from typing import List, Tuple, Dict, Optional, Set
import logging

logger = logging.getLogger(__name__)


class ReactionParser:
    """
    Parser for converting reaction equations to kinetic matrices
    
    Based on EfsTA's custom model parser with enhancements for
    TAS analysis workflows and better error handling.
    """
    
    def __init__(self):
        """Initialize the reaction parser"""
        self.species_map = self._create_species_map()
        self.max_species = 26
        
    def _create_species_map(self) -> Dict[str, int]:
        """Create mapping from species letters to matrix indices"""
        species_map = {}
        for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            species_map[letter] = i
        species_map['v'] = -1  # void (ground state)
        return species_map
    
    def parse_reaction_equation(self, equation: str, rate_constants: List[float]) -> Tuple[np.ndarray, int, Dict]:
        """
        Parse reaction equation and generate kinetic matrix
        
        Parameters
        ----------
        equation : str
            Reaction equation string
        rate_constants : List[float]
            Rate constants for each reaction step
            
        Returns
        -------
        K : np.ndarray
            Kinetic matrix for the reaction system
        n_species : int
            Number of species in the system
        info : Dict
            Information about the parsed reaction system
        """
        # Clean and validate input
        equation = equation.strip().replace('→', '->')
        self._validate_equation(equation)
        
        # Parse equation structure
        pathways = self._split_pathways(equation)
        all_species = self._extract_all_species(pathways)
        n_species = len(all_species)
        
        # Validate rate constants
        n_reactions = self._count_reactions(pathways)
        if len(rate_constants) != n_reactions:
            raise ValueError(f"Expected {n_reactions} rate constants, got {len(rate_constants)}")
        
        # Generate kinetic matrix
        K = self._build_kinetic_matrix(pathways, all_species, rate_constants)
        
        # Create information dictionary
        info = {
            'equation': equation,
            'pathways': pathways,
            'species': sorted(all_species),
            'n_species': n_species,
            'n_reactions': n_reactions,
            'has_void_decay': 'v' in equation,
            'pathway_type': self._classify_pathway_type(pathways)
        }
        
        logger.info(f"Parsed reaction equation: {equation}")
        logger.info(f"Generated {n_species}-species kinetic matrix")
        
        return K, n_species, info
    
    def _validate_equation(self, equation: str) -> None:
        """Validate reaction equation syntax"""
        if not equation:
            raise ValueError("Empty reaction equation")
        
        # Check for valid characters
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ->v;")
        if not set(equation.replace(' ', '')).issubset(valid_chars):
            invalid = set(equation.replace(' ', '')) - valid_chars
            raise ValueError(f"Invalid characters in equation: {invalid}")
        
        # Check for mixing arrow and no-arrow notation within pathways
        pathways = equation.split(';')
        for pathway in pathways:
            pathway = pathway.strip()
            has_arrows = '->' in pathway
            # If arrows present, should not have adjacent letters without arrows
            if has_arrows:
                # Check if there are any sequences of uppercase letters that are not separated by arrows
                # Split by arrows and check if any segment has multiple consecutive letters
                segments = pathway.split('->')
                for segment in segments:
                    segment = segment.strip()
                    if len(segment) > 1 and segment.isalpha():
                        # This segment has multiple letters without arrows between them
                        raise ValueError(f"Multiple species without arrows in segment '{segment}' of pathway: {pathway}")
            # If no arrows, should not have arrow characters
            elif '->' in pathway:
                raise ValueError(f"Inconsistent arrow usage in pathway: {pathway}")
    
    def _split_pathways(self, equation: str) -> List[List[str]]:
        """Split equation into individual reaction pathways"""
        pathways = []
        
        for pathway_str in equation.split(';'):
            pathway_str = pathway_str.strip()
            if not pathway_str:
                continue
                
            if '->' in pathway_str:
                # Arrow notation
                species_list = [s.strip() for s in pathway_str.split('->')]
            else:
                # No-arrow notation - each character is a species
                species_list = list(pathway_str.replace(' ', ''))
            
            # Remove empty strings
            species_list = [s for s in species_list if s]
            
            if len(species_list) < 2:
                raise ValueError(f"Pathway must have at least 2 species: {pathway_str}")
                
            pathways.append(species_list)
        
        return pathways
    
    def _extract_all_species(self, pathways: List[List[str]]) -> Set[str]:
        """Extract all unique species from pathways (excluding void)"""
        all_species = set()
        
        for pathway in pathways:
            for species in pathway:
                if species != 'v':  # Exclude void
                    all_species.add(species)
        
        # Validate species are in alphabetical order
        species_list = sorted(all_species)
        expected = [chr(ord('A') + i) for i in range(len(species_list))]
        
        if species_list != expected:
            raise ValueError(f"Species must be consecutive letters starting from A. Got: {species_list}, Expected: {expected}")
        
        return all_species
    
    def _count_reactions(self, pathways: List[List[str]]) -> int:
        """Count total number of reaction steps"""
        total_reactions = 0
        
        for pathway in pathways:
            # Each pathway contributes (n_species - 1) reactions
            total_reactions += len(pathway) - 1
        
        return total_reactions
    
    def _build_kinetic_matrix(self, pathways: List[List[str]], all_species: Set[str], 
                             rate_constants: List[float]) -> np.ndarray:
        """Build kinetic matrix from parsed pathways"""
        n_species = len(all_species)
        K = np.zeros((n_species, n_species))
        
        # Species index mapping (excluding void)
        species_to_idx = {}
        for i, species in enumerate(sorted(all_species)):
            species_to_idx[species] = i
        
        rate_idx = 0
        
        # Process each pathway
        for pathway in pathways:
            # Process each reaction in the pathway
            for i in range(len(pathway) - 1):
                reactant = pathway[i]
                product = pathway[i + 1]
                
                if rate_idx >= len(rate_constants):
                    raise ValueError("Not enough rate constants for all reactions")
                
                k = rate_constants[rate_idx]
                rate_idx += 1
                
                if reactant != 'v':  # Valid reactant
                    reactant_idx = species_to_idx[reactant]
                    
                    # Loss term (diagonal)
                    K[reactant_idx, reactant_idx] -= k
                    
                    # Gain term (off-diagonal) - only if product is not void
                    if product != 'v':
                        product_idx = species_to_idx[product]
                        K[product_idx, reactant_idx] += k
        
        return K
    
    def _classify_pathway_type(self, pathways: List[List[str]]) -> str:
        """Classify the type of reaction pathway"""
        if len(pathways) == 1:
            return "sequential"
        elif len(pathways) == 2:
            return "branching" 
        else:
            return "complex_network"
    
    def validate_equation_syntax(self, equation: str) -> Tuple[bool, str]:
        """
        Validate equation syntax without parsing
        
        Parameters
        ----------
        equation : str
            Reaction equation to validate
            
        Returns
        -------
        is_valid : bool
            True if syntax is valid
        message : str
            Validation message or error description
        """
        try:
            self._validate_equation(equation)
            return True, "Valid equation syntax"
        except ValueError as e:
            return False, str(e)
    
    def get_equation_info(self, equation: str) -> Dict:
        """
        Get information about equation without generating matrix
        
        Parameters
        ----------
        equation : str
            Reaction equation string
            
        Returns
        -------
        info : Dict
            Information about the equation structure
        """
        try:
            equation = equation.strip().replace('→', '->')
            self._validate_equation(equation)
            
            pathways = self._split_pathways(equation)
            all_species = self._extract_all_species(pathways)
            n_reactions = self._count_reactions(pathways)
            
            return {
                'valid': True,
                'species': sorted(all_species),
                'n_species': len(all_species),
                'n_reactions': n_reactions,
                'pathways': pathways,
                'has_void_decay': 'v' in equation,
                'pathway_type': self._classify_pathway_type(pathways),
                'required_rate_constants': n_reactions
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'species': [],
                'n_species': 0,
                'n_reactions': 0,
                'pathways': [],
                'has_void_decay': False,
                'pathway_type': 'invalid',
                'required_rate_constants': 0
            }
    
    def suggest_rate_constants(self, equation: str, timescale_ps: Optional[float] = None) -> List[float]:
        """
        Suggest reasonable rate constants for an equation
        
        Parameters
        ----------
        equation : str
            Reaction equation
        timescale_ps : float, optional
            Expected timescale in picoseconds
            
        Returns
        -------
        rate_constants : List[float]
            Suggested rate constants in ps^-1
        """
        info = self.get_equation_info(equation)
        
        if not info['valid']:
            raise ValueError(f"Invalid equation: {info['error']}")
        
        n_reactions = info['n_reactions']
        
        if timescale_ps is None:
            # Default suggestions based on typical TAS timescales
            suggestions = []
            for i in range(n_reactions):
                # Faster reactions first, then slower
                k = 1.0 / (10**(i * 0.5))  # Decreasing by ~3x each step
                suggestions.append(k)
        else:
            # Base suggestions on provided timescale
            base_rate = 1.0 / timescale_ps
            suggestions = []
            for i in range(n_reactions):
                k = base_rate * (0.5 ** i)  # Each step 2x slower
                suggestions.append(k)
        
        return suggestions


# Convenience functions
def parse_reaction(equation: str, rate_constants: List[float]) -> Tuple[np.ndarray, int, Dict]:
    """
    Quick function to parse reaction equation
    
    Parameters
    ----------
    equation : str
        Reaction equation string
    rate_constants : List[float]
        Rate constants for reactions
        
    Returns
    -------
    K : np.ndarray
        Kinetic matrix
    n_species : int
        Number of species
    info : Dict
        Parsing information
    """
    parser = ReactionParser()
    return parser.parse_reaction_equation(equation, rate_constants)


def validate_reaction(equation: str) -> Tuple[bool, str]:
    """
    Quick function to validate reaction equation
    
    Parameters
    ----------
    equation : str
        Reaction equation to validate
        
    Returns
    -------
    is_valid : bool
        True if valid
    message : str
        Validation message
    """
    parser = ReactionParser()
    return parser.validate_equation_syntax(equation)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    parser = ReactionParser()
    
    print("Testing Reaction Parser Module")
    print("=" * 40)
    
    # Test cases
    test_equations = [
        ("A->B->C->v", [1.0, 0.5, 0.1]),
        ("A->B;A->C", [2.0, 1.0]),
        ("ABC;BA", [1.0, 0.5, 0.3]),
        ("A->B->C->D;B->D", [2.0, 1.0, 0.5, 0.3]),
    ]
    
    for equation, rate_constants in test_equations:
        print(f"\nTesting equation: {equation}")
        print(f"Rate constants: {rate_constants}")
        
        try:
            # Get equation info
            info = parser.get_equation_info(equation)
            print(f"Species: {info['species']}")
            print(f"Reactions: {info['n_reactions']}")
            print(f"Pathway type: {info['pathway_type']}")
            
            # Parse to matrix
            K, n, parse_info = parser.parse_reaction_equation(equation, rate_constants)
            print(f"Kinetic Matrix ({n} species):")
            print(K)
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Test validation
    print("\n" + "=" * 40)
    print("Testing equation validation:")
    
    invalid_equations = [
        "A->B->X",  # Non-consecutive species
        "A->B->C;Z->Y",  # Non-consecutive species
        "A->B->C;A->B->D->C",  # Mixing notations incorrectly
        "",  # Empty equation
    ]
    
    for eq in invalid_equations:
        is_valid, msg = parser.validate_equation_syntax(eq)
        print(f"'{eq}': {is_valid} - {msg}")
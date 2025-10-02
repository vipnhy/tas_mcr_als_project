# -*- coding: utf-8 -*-
"""
GTA Visualization Module

Provides comprehensive visualization capabilities for Global Target Analysis results,
including Species-Associated Spectra (SAS), concentration profiles, residuals,
and other diagnostic plots.

Features:
- Species-Associated Spectra plots with proper labeling
- Concentration time evolution profiles  
- 2D and 3D residual analysis
- Data reconstruction quality assessment
- Publication-ready figure generation
- Bilingual support (Chinese/English) following project standards
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple, Union
import logging

# Import project-specific visualization utilities
try:
    from ..utils.visualization import setup_matplotlib_chinese, get_color_palette
except ImportError:
    # Fallback if project utilities not available
    def setup_matplotlib_chinese():
        pass
    def get_color_palette():
        return plt.cm.Set1.colors

logger = logging.getLogger(__name__)


class GTAVisualization:
    """
    Comprehensive visualization for GTA analysis results
    
    Provides methods to create publication-ready plots for all aspects
    of Global Target Analysis including SAS, concentrations, and residuals.
    """
    
    def __init__(self, language: str = 'zh', style: str = 'scientific'):
        """
        Initialize GTA visualization
        
        Parameters
        ----------
        language : str
            Language for labels ('zh' for Chinese, 'en' for English)  
        style : str
            Plot style ('scientific', 'presentation', 'paper')
        """
        self.language = language
        self.style = style
        
        # Setup matplotlib for Chinese if needed
        if language == 'zh':
            setup_matplotlib_chinese()
        
        # Define labels in both languages
        self._setup_labels()
        
        # Setup plot style
        self._setup_plot_style()
        
        logger.info(f"Initialized GTA visualization with language={language}, style={style}")
    
    def _setup_labels(self) -> None:
        """Setup labels for both languages"""
        self.labels = {
            'zh': {
                'wavelength': '波长 (nm)',
                'time': '时间 (ps)', 
                'delay': '延迟时间 (ps)',
                'absorption': '吸收变化 (ΔA)',
                'absorption_milli': '吸收变化 (mΔA)',
                'concentration': '浓度',
                'species': '物种',
                'residuals': '残差',
                'sas_title': '物种关联光谱 (SAS)',
                'concentration_title': '浓度时间演化',
                'residuals_title': '拟合残差分析',
                'species_a': '物种 A',
                'species_b': '物种 B', 
                'species_c': '物种 C',
                'species_d': '物种 D',
                'species_e': '物种 E',
                'species_f': '物种 F',
                'fit_quality': '拟合质量',
                'data_vs_fit': '实验数据 vs 拟合结果',
                'relative_error': '相对误差 (%)',
                'chi_squared': 'χ² 值',
                'reduced_chi_squared': '约化 χ²'
            },
            'en': {
                'wavelength': 'Wavelength (nm)',
                'time': 'Time (ps)',
                'delay': 'Delay time (ps)', 
                'absorption': 'Absorption change (ΔA)',
                'absorption_milli': 'Absorption change (mΔA)',
                'concentration': 'Concentration',
                'species': 'Species',
                'residuals': 'Residuals',
                'sas_title': 'Species-Associated Spectra (SAS)',
                'concentration_title': 'Concentration Time Evolution',
                'residuals_title': 'Fitting Residuals Analysis',
                'species_a': 'Species A',
                'species_b': 'Species B',
                'species_c': 'Species C', 
                'species_d': 'Species D',
                'species_e': 'Species E',
                'species_f': 'Species F',
                'fit_quality': 'Fit Quality',
                'data_vs_fit': 'Experimental Data vs Fit',
                'relative_error': 'Relative Error (%)',
                'chi_squared': 'χ² Value',
                'reduced_chi_squared': 'Reduced χ²'
            }
        }
    
    def _setup_plot_style(self) -> None:
        """Setup plot style parameters"""
        styles = {
            'scientific': {
                'figure_size': (8, 6),
                'dpi': 150,
                'font_size': 10,
                'line_width': 1.5,
                'marker_size': 4
            },
            'presentation': {
                'figure_size': (10, 7),
                'dpi': 120,
                'font_size': 12,
                'line_width': 2.0,
                'marker_size': 6
            },
            'paper': {
                'figure_size': (6, 4.5),
                'dpi': 300,
                'font_size': 8,
                'line_width': 1.0,
                'marker_size': 3
            }
        }
        
        self.style_params = styles.get(self.style, styles['scientific'])
        
        # Apply matplotlib style
        plt.rcParams.update({
            'figure.figsize': self.style_params['figure_size'],
            'figure.dpi': self.style_params['dpi'],
            'font.size': self.style_params['font_size'],
            'lines.linewidth': self.style_params['line_width'],
            'lines.markersize': self.style_params['marker_size'],
            'axes.linewidth': 0.8,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'legend.frameon': False
        })
    
    def plot_species_spectra(self, 
                           wavelengths: np.ndarray,
                           species_spectra: np.ndarray,
                           species_names: Optional[List[str]] = None,
                           title: Optional[str] = None,
                           save_path: Optional[str] = None) -> Figure:
        """
        Plot Species-Associated Spectra (SAS)
        
        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelength array (nm)
        species_spectra : np.ndarray
            Species spectra (n_species x n_wavelengths)
        species_names : List[str], optional
            Custom species names
        title : str, optional
            Custom plot title
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure
        """
        n_species = species_spectra.shape[0]
        
        # Generate species names if not provided
        if species_names is None:
            species_names = [self.labels[self.language][f'species_{chr(97+i)}'] for i in range(min(n_species, 6))]
            if n_species > 6:
                species_names.extend([f'Species {chr(71+i)}' for i in range(n_species - 6)])
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.style_params['figure_size'])
        
        # Get color palette
        colors = get_color_palette()
        
        # Plot each species spectrum
        for i, spectrum in enumerate(species_spectra):
            color = colors[i % len(colors)]
            label = species_names[i] if i < len(species_names) else f'Species {i+1}'
            
            ax.plot(wavelengths, spectrum, 
                   color=color, 
                   linewidth=self.style_params['line_width'],
                   label=label)
        
        # Formatting
        ax.set_xlabel(self.labels[self.language]['wavelength'])
        ax.set_ylabel(self.labels[self.language]['absorption'])
        ax.set_title(title or self.labels[self.language]['sas_title'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.style_params['dpi'], bbox_inches='tight')
            logger.info(f"Saved SAS plot to {save_path}")
        
        return fig
    
    def plot_concentration_profiles(self,
                                  delays: np.ndarray,
                                  concentration_profiles: np.ndarray,
                                  species_names: Optional[List[str]] = None,
                                  log_scale: bool = True,
                                  title: Optional[str] = None,
                                  save_path: Optional[str] = None) -> Figure:
        """
        Plot concentration time evolution profiles
        
        Parameters
        ----------
        delays : np.ndarray
            Time delay array (ps)
        concentration_profiles : np.ndarray
            Concentration profiles (n_delays x n_species)
        species_names : List[str], optional
            Custom species names
        log_scale : bool
            Use logarithmic time scale
        title : str, optional
            Custom plot title
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure
        """
        n_species = concentration_profiles.shape[1]
        
        # Generate species names if not provided
        if species_names is None:
            species_names = [self.labels[self.language][f'species_{chr(97+i)}'] for i in range(min(n_species, 6))]
            if n_species > 6:
                species_names.extend([f'Species {chr(71+i)}' for i in range(n_species - 6)])
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.style_params['figure_size'])
        
        # Get color palette
        colors = get_color_palette()
        
        # Plot each species concentration
        for i in range(n_species):
            color = colors[i % len(colors)]
            label = species_names[i] if i < len(species_names) else f'Species {i+1}'
            
            ax.plot(delays, concentration_profiles[:, i],
                   color=color,
                   linewidth=self.style_params['line_width'],
                   label=label)
        
        # Formatting
        ax.set_xlabel(self.labels[self.language]['delay'])
        ax.set_ylabel(self.labels[self.language]['concentration'])
        ax.set_title(title or self.labels[self.language]['concentration_title'])
        
        if log_scale and np.min(delays) > 0:
            ax.set_xscale('log')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.style_params['dpi'], bbox_inches='tight')
            logger.info(f"Saved concentration profiles to {save_path}")
        
        return fig
    
    def plot_residuals_2d(self,
                         wavelengths: np.ndarray,
                         delays: np.ndarray, 
                         residuals: np.ndarray,
                         title: Optional[str] = None,
                         save_path: Optional[str] = None) -> Figure:
        """
        Plot 2D residuals heatmap
        
        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelength array (nm)
        delays : np.ndarray
            Time delay array (ps)
        residuals : np.ndarray
            Residuals matrix (n_delays x n_wavelengths)
        title : str, optional
            Custom plot title
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.style_params['figure_size'])
        
        # Calculate color scale limits
        vmax = np.percentile(np.abs(residuals), 95)
        vmin = -vmax
        
        # Create heatmap
        im = ax.contourf(wavelengths, delays, residuals, 
                        levels=50, cmap='RdBu_r', 
                        vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(self.labels[self.language]['residuals'])
        
        # Formatting
        ax.set_xlabel(self.labels[self.language]['wavelength'])
        ax.set_ylabel(self.labels[self.language]['delay'])
        ax.set_title(title or self.labels[self.language]['residuals_title'])
        
        # Use log scale for time if appropriate
        if np.min(delays) > 0:
            ax.set_yscale('log')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.style_params['dpi'], bbox_inches='tight')
            logger.info(f"Saved 2D residuals plot to {save_path}")
        
        return fig
    
    def plot_data_vs_fit_comparison(self,
                                   wavelengths: np.ndarray,
                                   delays: np.ndarray,
                                   experimental_data: np.ndarray,
                                   fitted_data: np.ndarray,
                                   selected_delays: Optional[List[float]] = None,
                                   title: Optional[str] = None,
                                   save_path: Optional[str] = None) -> Figure:
        """
        Plot comparison of experimental vs fitted data at selected delays
        
        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelength array (nm)
        delays : np.ndarray
            Time delay array (ps)
        experimental_data : np.ndarray
            Experimental data (n_delays x n_wavelengths)
        fitted_data : np.ndarray
            Fitted data (n_delays x n_wavelengths)
        selected_delays : List[float], optional
            Specific delays to show. If None, automatically select representative delays
        title : str, optional
            Custom plot title
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure
        """
        # Select representative delays if not provided
        if selected_delays is None:
            # Select delays logarithmically spaced
            if len(delays) > 6:
                indices = np.logspace(0, np.log10(len(delays)-1), 6, dtype=int)
                selected_delays = delays[indices]
            else:
                selected_delays = delays[::max(1, len(delays)//6)]
        
        # Find closest delay indices
        delay_indices = []
        for target_delay in selected_delays:
            idx = np.argmin(np.abs(delays - target_delay))
            delay_indices.append(idx)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        colors = ['blue', 'red']
        
        for i, idx in enumerate(delay_indices[:6]):
            ax = axes[i]
            delay_val = delays[idx]
            
            # Plot experimental and fitted data
            ax.plot(wavelengths, experimental_data[idx, :], 
                   color=colors[0], linewidth=1.5, 
                   label='实验数据' if self.language == 'zh' else 'Experimental')
            ax.plot(wavelengths, fitted_data[idx, :], 
                   color=colors[1], linewidth=1.5, linestyle='--',
                   label='拟合数据' if self.language == 'zh' else 'Fitted')
            
            ax.set_title(f't = {delay_val:.1f} ps')
            ax.set_xlabel(self.labels[self.language]['wavelength'])
            ax.set_ylabel(self.labels[self.language]['absorption'])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle(title or self.labels[self.language]['data_vs_fit'])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.style_params['dpi'], bbox_inches='tight')
            logger.info(f"Saved data vs fit comparison to {save_path}")
        
        return fig
    
    def plot_fit_quality_summary(self,
                                fit_result: Dict,
                                save_path: Optional[str] = None) -> Figure:
        """
        Plot comprehensive fit quality summary
        
        Parameters
        ----------
        fit_result : Dict
            Fit result dictionary from GTASolver
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(12, 10))
        
        # Extract data from fit result
        fitted_rates = fit_result['fitted_rates']
        parameter_errors = fit_result['parameter_errors']
        chi_squared = fit_result['chi_squared']
        reduced_chi_squared = fit_result['reduced_chi_squared']
        residuals = fit_result['residuals']
        
        # Plot 1: Parameter values with error bars
        ax1 = plt.subplot(2, 3, 1)
        x_pos = range(len(fitted_rates))
        bars = ax1.bar(x_pos, fitted_rates, 
                      yerr=parameter_errors if parameter_errors[0] is not None else None,
                      capsize=5, alpha=0.7)
        ax1.set_xlabel('参数索引' if self.language == 'zh' else 'Parameter Index')
        ax1.set_ylabel('速率常数 (ps⁻¹)' if self.language == 'zh' else 'Rate Constant (ps⁻¹)')
        ax1.set_title('拟合参数' if self.language == 'zh' else 'Fitted Parameters')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'k{i}' for i in range(len(fitted_rates))])
        
        # Plot 2: Residuals histogram
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(residuals.flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax2.set_xlabel(self.labels[self.language]['residuals'])
        ax2.set_ylabel('频次' if self.language == 'zh' else 'Frequency')
        ax2.set_title('残差分布' if self.language == 'zh' else 'Residuals Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Fit statistics
        ax3 = plt.subplot(2, 3, 3)
        stats_text = f"""
        {self.labels[self.language]['chi_squared']}: {chi_squared:.2e}
        {self.labels[self.language]['reduced_chi_squared']}: {reduced_chi_squared:.3f}
        
        残差标准差: {np.std(residuals):.2e}
        最大残差: {np.max(np.abs(residuals)):.2e}
        """ if self.language == 'zh' else f"""
        {self.labels[self.language]['chi_squared']}: {chi_squared:.2e}
        {self.labels[self.language]['reduced_chi_squared']}: {reduced_chi_squared:.3f}
        
        Residual Std: {np.std(residuals):.2e}
        Max Residual: {np.max(np.abs(residuals)):.2e}
        """
        
        ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='center')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title(self.labels[self.language]['fit_quality'])
        
        # Additional plots can be added here...
        
        # Overall title
        fig.suptitle('GTA拟合质量总结' if self.language == 'zh' else 'GTA Fit Quality Summary', 
                    fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.style_params['dpi'], bbox_inches='tight')
            logger.info(f"Saved fit quality summary to {save_path}")
        
        return fig
    
    def create_comprehensive_report(self,
                                  gta_solver,
                                  fit_result: Dict,
                                  output_dir: str) -> List[str]:
        """
        Create comprehensive GTA analysis report with all plots
        
        Parameters
        ----------
        gta_solver : GTASolver
            The GTA solver instance with results
        fit_result : Dict
            Fit result dictionary
        output_dir : str
            Directory to save all plots
            
        Returns
        -------
        saved_files : List[str]
            List of saved file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        
        # Plot 1: Species-Associated Spectra
        if gta_solver.species_spectra is not None:
            fig_sas = self.plot_species_spectra(
                gta_solver.wavelengths,
                gta_solver.species_spectra,
                save_path=os.path.join(output_dir, 'sas_spectra.png')
            )
            saved_files.append(os.path.join(output_dir, 'sas_spectra.png'))
            plt.close(fig_sas)
        
        # Plot 2: Concentration Profiles
        if gta_solver.concentration_profiles is not None:
            fig_conc = self.plot_concentration_profiles(
                gta_solver.delays,
                gta_solver.concentration_profiles,
                save_path=os.path.join(output_dir, 'concentration_profiles.png')
            )
            saved_files.append(os.path.join(output_dir, 'concentration_profiles.png'))
            plt.close(fig_conc)
        
        # Plot 3: 2D Residuals
        if gta_solver.residuals is not None:
            fig_res = self.plot_residuals_2d(
                gta_solver.wavelengths,
                gta_solver.delays,
                gta_solver.residuals,
                save_path=os.path.join(output_dir, 'residuals_2d.png')
            )
            saved_files.append(os.path.join(output_dir, 'residuals_2d.png'))
            plt.close(fig_res)
        
        # Plot 4: Data vs Fit Comparison
        if gta_solver.reconstructed_spectra is not None:
            fig_comp = self.plot_data_vs_fit_comparison(
                gta_solver.wavelengths,
                gta_solver.delays,
                gta_solver.spectra,
                gta_solver.reconstructed_spectra,
                save_path=os.path.join(output_dir, 'data_vs_fit.png')
            )
            saved_files.append(os.path.join(output_dir, 'data_vs_fit.png'))
            plt.close(fig_comp)
        
        # Plot 5: Fit Quality Summary
        fig_quality = self.plot_fit_quality_summary(
            fit_result,
            save_path=os.path.join(output_dir, 'fit_quality_summary.png')
        )
        saved_files.append(os.path.join(output_dir, 'fit_quality_summary.png'))
        plt.close(fig_quality)
        
        logger.info(f"Created comprehensive GTA report in {output_dir}")
        logger.info(f"Saved {len(saved_files)} plots")
        
        return saved_files


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("Testing GTA Visualization Module")
    print("=" * 40)
    
    # Create synthetic test data
    n_delays = 50
    n_wavelengths = 100
    n_species = 3
    
    wavelengths = np.linspace(400, 700, n_wavelengths)
    delays = np.logspace(-1, 3, n_delays)
    
    # Generate synthetic species spectra
    species_spectra = np.array([
        np.exp(-((wavelengths - 450)**2) / (2 * 30**2)),
        np.exp(-((wavelengths - 550)**2) / (2 * 40**2)),  
        np.exp(-((wavelengths - 650)**2) / (2 * 35**2))
    ])
    
    # Generate synthetic concentration profiles
    concentration_profiles = np.zeros((n_delays, n_species))
    for i, t in enumerate(delays):
        concentration_profiles[i, 0] = np.exp(-t / 100)  # A decay
        concentration_profiles[i, 1] = np.exp(-t / 100) - np.exp(-t / 50)  # B rise/decay
        concentration_profiles[i, 2] = 1 - np.exp(-t / 50)  # C rise
    
    # Generate synthetic residuals
    residuals = np.random.normal(0, 0.001, (n_delays, n_wavelengths))
    
    # Test visualization
    viz = GTAVisualization(language='zh')
    
    try:
        # Test SAS plot
        print("Testing SAS plot...")
        fig_sas = viz.plot_species_spectra(wavelengths, species_spectra)
        plt.show()
        
        # Test concentration profiles
        print("Testing concentration profiles...")
        fig_conc = viz.plot_concentration_profiles(delays, concentration_profiles)
        plt.show()
        
        # Test residuals
        print("Testing residuals plot...")
        fig_res = viz.plot_residuals_2d(wavelengths, delays, residuals)
        plt.show()
        
        print("All visualization tests completed successfully!")
        
    except Exception as e:
        print(f"Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
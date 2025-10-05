import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm
import corner
from matplotlib import rc

# Configure matplotlib
rc('text', usetex=True)
rc('font', family='serif')
rc('font', serif='times')
rc('mathtext', default='sf')
rc("lines", markeredgewidth=1)
rc("lines", linewidth=1)
rc('axes', labelsize=20)
rc("axes", linewidth=0.5)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)
rc('legend', fontsize=10)
rc('ytick', right=True, direction='in')
rc('xtick', top=True, direction='in')
rc('xtick.major', pad=15)
rc('ytick.major', pad=15)
rc('xtick.major', size=12)
rc('ytick.major', size=12)
rc('xtick.minor', size=7)
rc('ytick.minor', size=7)

def set_tick_sizes(ax, major=12, minor=7):
    for line in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
        line.set_markersize(major)
    for tick in ax.xaxis.get_minor_ticks() + ax.yaxis.get_minor_ticks():
        tick.tick1line.set_markersize(minor)
        tick.tick2line.set_markersize(minor)

def calculate_pp_values(flow, test_loader, sample, device, num_posterior_samples=5000):
    """
    Calculate P-P values for the trained normalizing flow model.
    
    Parameters:
    -----------
    flow : Flow
        Trained normalizing flow model
    test_loader : DataLoader
        Test data loader
    device : torch.device
        Device to run computation on
    num_posterior_samples : int
        Number of posterior samples to draw
    
    Returns:
    --------
    pp_values : np.ndarray
        P-P values for each parameter and each test sample
    parameter_names : list
        Names of parameters
    """
    
    flow.eval()
    pp_values = []
    parameter_names = list(sample.keys())
    
    with torch.no_grad():
        for idx, (theta_true, data) in enumerate(tqdm(test_loader, desc="Computing P-P values")):
            theta_true = theta_true.to(device)
            data = data.to(device)
            
            # Generate posterior samples
            posterior_samples = flow.sample(num_posterior_samples, context=data)
            posterior_samples = posterior_samples.squeeze(0).cpu().numpy()
            theta_true_np = theta_true.squeeze(0).cpu().numpy()
            
            # Calculate P-values for each parameter
            p_values = []
            for param_idx in range(len(parameter_names)):
                # Get posterior samples for this parameter
                param_samples = posterior_samples[:, param_idx]
                true_value = theta_true_np[param_idx]
                
                # Calculate what fraction of posterior samples are less than true value
                p_value = np.mean(param_samples < true_value)
                p_values.append(p_value)
            
            pp_values.append(p_values)
    
    return np.array(pp_values), parameter_names

def plot_pp_plot(pp_values, parameter_names, confidence_level=0.95):
    """
    Generate P-P plots for model calibration assessment.
    
    Parameters:
    -----------
    pp_values : np.ndarray
        P-P values from calculate_pp_values
    parameter_names : list
        Names of parameters
    confidence_level : float
        Confidence level for the credible interval
    """
    
    n_params = len(parameter_names)
    n_tests = len(pp_values)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_bound = stats.beta.ppf(alpha/2, np.arange(1, n_tests+1), n_tests - np.arange(1, n_tests+1) + 1)
    upper_bound = stats.beta.ppf(1 - alpha/2, np.arange(1, n_tests+1), n_tests - np.arange(1, n_tests+1) + 1)
    
    # Dynamically determine subplot layout based on number of parameters
    if n_params <= 4:
        ncols = 2
        nrows = 2
    elif n_params <= 6:
        ncols = 3
        nrows = 2
    elif n_params <= 9:
        ncols = 3
        nrows = 3
    elif n_params <= 12:
        ncols = 4
        nrows = 3
    else:
        ncols = 5
        nrows = int(np.ceil(n_params / 5))
    
    # Create figure with appropriate size
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    
    # Handle case where there's only one subplot
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Expected quantiles (uniform distribution)
    expected_quantiles = np.linspace(0, 1, n_tests)
    
    for i, param_name in enumerate(parameter_names):
        ax = axes[i]
        
        # Get p-values for this parameter
        param_pp_values = pp_values[:, i]
        
        # Sort p-values
        sorted_pp = np.sort(param_pp_values)
        
        # Plot P-P plot
        ax.plot(expected_quantiles, sorted_pp, 'b-', linewidth=2, label='Observed')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfect calibration')
        
        # Plot confidence interval
        ax.fill_between(expected_quantiles, lower_bound, upper_bound, 
                       alpha=0.3, color='gray', label=f'{confidence_level*100}% CI')
        
        ax.set_xlabel('Expected quantile', fontsize=10)
        ax.set_ylabel('Observed quantile', fontsize=10)
        ax.set_title(f'{param_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add legend only to first subplot to avoid clutter
        if i == 0:
            ax.legend(fontsize=8, loc='upper left')
    
    # Hide unused subplots
    for i in range(len(parameter_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Q-Q Plots for All Parameters', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle
    plt.show()
    
    return fig

def compute_pp_statistics(pp_values, parameter_names):
    """
    Compute summary statistics for P-P values.
    
    Parameters:
    -----------
    pp_values : np.ndarray
        P-P values from calculate_pp_values
    parameter_names : list
        Names of parameters
    """
    
    print("P-P Plot Statistics:")
    print("=" * 50)
    
    for i, param_name in enumerate(parameter_names):
        param_pp_values = pp_values[:, i]
        
        # Kolmogorov-Smirnov test against uniform distribution
        ks_stat, ks_p_value = stats.kstest(param_pp_values, 'uniform')
        
        # Anderson-Darling test
        try:
            ad_stat, ad_crit_vals, ad_significance = stats.anderson(param_pp_values, dist='uniform')
        except:
            ad_stat = np.nan
            ad_significance = np.nan
        
        # Mean and standard deviation
        mean_pp = np.mean(param_pp_values)
        std_pp = np.std(param_pp_values)
        
        print(f"\n{param_name}:")
        print(f"  Mean P-value: {mean_pp:.4f} (should be ~0.5)")
        print(f"  Std P-value: {std_pp:.4f} (should be ~{1/np.sqrt(12):.4f})")
        print(f"  KS test p-value: {ks_p_value:.4f} (>0.05 indicates good calibration)")
        if not np.isnan(ad_stat):
            print(f"  Anderson-Darling statistic: {ad_stat:.4f}")

def plot_overlay_pp_plot(pp_values, parameter_names, confidence_level=0.95):
    """
    Plot all P-P plots overlaid on a single figure with different colors, 
    showing both 95% and 99% confidence intervals.

    Parameters
    ----------
    pp_values : ndarray
        Shape (n_tests, n_parameters), P-P values for each parameter across tests.
    parameter_names : list of str
        Names of the parameters.
    confidence_level : float
        Confidence interval for the shaded region (default 0.95).
    """
    n_tests = pp_values.shape[0]

    # Calculate 95% CI
    alpha = 1 - confidence_level
    alpha_lower_bound = stats.beta.ppf(alpha/2, np.arange(1, n_tests+1), n_tests - np.arange(1, n_tests+1) + 1)
    alpha_upper_bound = stats.beta.ppf(1 - alpha/2, np.arange(1, n_tests+1), n_tests - np.arange(1, n_tests+1) + 1)

    # Calculate 99% CI
    beta = 1 - 0.99
    beta_lower_bound = stats.beta.ppf(beta/2, np.arange(1, n_tests+1), n_tests - np.arange(1, n_tests+1) + 1)
    beta_upper_bound = stats.beta.ppf(1 - beta/2, np.arange(1, n_tests+1), n_tests - np.arange(1, n_tests+1) + 1)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    set_tick_sizes(ax)

    # Expected quantiles
    expected_quantiles = np.linspace(0, 1, n_tests)

    # Colors for different parameters
    colors = plt.cm.tab20(np.linspace(0, 1, len(parameter_names)))

    # Plot each parameter
    for i, param_name in enumerate(parameter_names):
        sorted_pp = np.sort(pp_values[:, i])
        ax.plot(expected_quantiles, sorted_pp, color=colors[i], linewidth=2, 
                label=param_name, alpha=0.8)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')

    # Fill 99% CI first (lighter)
    ax.fill_between(expected_quantiles, beta_lower_bound, beta_upper_bound, 
                    alpha=0.1, color='gray', label='99\% CI')

    # Fill 95% CI on top (darker)
    ax.fill_between(expected_quantiles, alpha_lower_bound, alpha_upper_bound, 
                    alpha=0.2, color='gray', label=f'{confidence_level*100:.0f}\% CI')

    # Labels and styling
    ax.set_xlabel('Expected quantile', fontsize=12)
    ax.set_ylabel('Observed quantile', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('QQ_Plot.pdf')
    plt.show()

# # Alternative: Compact version with all parameters in one plot overlaid
# def plot_overlay_pp_plot(pp_values, parameter_names, confidence_level=0.95):
#     """
#     Plot all P-P plots overlaid on a single figure with different colors.
#     """
#     n_tests = len(pp_values)
    
#     # Calculate confidence intervals
#     alpha = 1 - confidence_level
#     lower_bound = stats.beta.ppf(alpha/2, np.arange(1, n_tests+1), n_tests - np.arange(1, n_tests+1) + 1)
#     upper_bound = stats.beta.ppf(1 - alpha/2, np.arange(1, n_tests+1), n_tests - np.arange(1, n_tests+1) + 1)
    
#     plt.figure(figsize=(10, 8))
    
#     # Expected quantiles (uniform distribution)
#     expected_quantiles = np.linspace(0, 1, n_tests)
    
#     # Define colors for different parameters
#     colors = plt.cm.tab20(np.linspace(0, 1, len(parameter_names)))
    
#     for i, param_name in enumerate(parameter_names):
#         # Get p-values for this parameter
#         param_pp_values = pp_values[:, i]
        
#         # Sort p-values
#         sorted_pp = np.sort(param_pp_values)
        
#         # Plot P-P plot
#         plt.plot(expected_quantiles, sorted_pp, color=colors[i], linewidth=2, 
#                 label=param_name, alpha=0.8)
    
#     # Plot perfect calibration line
#     plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    
#     # Plot confidence interval
#     plt.fill_between(expected_quantiles, lower_bound, upper_bound, 
#                     alpha=0.2, color='gray', label=f'{confidence_level*100}% CI')
    
#     plt.xlabel('Expected quantile', fontsize=12)
#     plt.ylabel('Observed quantile', fontsize=12)
#     # plt.title('Q-Q Plots for All Parameters', fontsize=14, fontweight='bold')
#     plt.grid(True, alpha=0.3)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
    
#     plt.tight_layout()
#     plt.savefig('QQ_Plot.pdf')
#     plt.show()

# # One-at-a-time P-P plot for detailed analysis
# def plot_individual_pp_plots(pp_values, parameter_names, confidence_level=0.95):
#     """
#     Plot individual P-P plots for each parameter separately.
#     """
#     n_tests = len(pp_values)
    
#     # Calculate confidence intervals
#     alpha = 1 - confidence_level
#     lower_bound = stats.beta.ppf(alpha/2, np.arange(1, n_tests+1), n_tests - np.arange(1, n_tests+1) + 1)
#     upper_bound = stats.beta.ppf(1 - alpha/2, np.arange(1, n_tests+1), n_tests - np.arange(1, n_tests+1) + 1)
    
#     for i, param_name in enumerate(parameter_names):
#         plt.figure(figsize=(8, 6))
        
#         # Get p-values for this parameter
#         param_pp_values = pp_values[:, i]
        
#         # Sort p-values
#         sorted_pp = np.sort(param_pp_values)
        
#         # Expected quantiles (uniform distribution)
#         expected_quantiles = np.linspace(0, 1, n_tests)
        
#         # Plot P-P plot
#         plt.plot(expected_quantiles, sorted_pp, 'b-', linewidth=2, label='Observed')
#         plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfect calibration')
        
#         # Plot confidence interval
#         plt.fill_between(expected_quantiles, lower_bound, upper_bound, 
#                         alpha=0.3, color='gray', label=f'{confidence_level*100}% CI')
        
#         plt.xlabel('Expected quantile')
#         plt.ylabel('Observed quantile')
#         plt.title(f'P-P Plot: {param_name}')
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#         plt.xlim(0, 1)
#         plt.ylim(0, 1)
        
#         plt.tight_layout()
#         plt.show()

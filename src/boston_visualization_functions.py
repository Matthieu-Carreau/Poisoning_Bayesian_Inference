# Functions to make plots for the experiments in the Boston housing dataset

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch

CLR_ORIGINAL = "Green"
CLR_TARGET = "Red"
CLR_TAINTED = "Blue"

colors = [CLR_ORIGINAL, CLR_TARGET, CLR_TAINTED]

n_features = 13

feature_names = [
    'CRIM',
    'ZN',
    'INDUS',
    'CHAS',
    'NOX',
    'RM',
    'AGE',
    'DIS',
    'RAD',
    'TAX',
    'PTRATIO',
    'B',
    'LSTAT']

AX_LBL_SIZE = 22
TICKS_LBL_SIZE = 15
LEGEND_LBL_SIZE = 15

def plot_samples_kde(beta_samples, beta_samples_target, beta_samples_tainted, ylim=None, title=""):
    """Plot a kde for all marginal distributions of all Horseshoe parameters"""

    samples_list = (beta_samples, beta_samples_target, beta_samples_tainted)
    labels = ["Original posterior", "Target posterior", "Tainted posterior"]

    fig, axes = plt.subplots(2, 15, figsize=(20, 6))
    alpha = 0.1

    for i in range(3):
        samples = samples_list[i]
        
        # Intercept
        sns.kdeplot(y=samples["intercept"], ax=axes[0, 0], color=colors[i], fill=True, alpha=alpha, label=labels[i])

        # Global shrinkage
        sns.kdeplot(y=np.log(samples["global_shrinkage"]), ax=axes[1, 0], color=colors[i], fill=True, alpha=alpha, label=labels[i])

        # Sigma
        sns.kdeplot(y=np.log(samples["sigma"]), ax=axes[1, -1], color=colors[i], fill=True, alpha=alpha, label=labels[i])

        # Betas
        for beta_idx in range(n_features):
            sns.kdeplot(y=samples["beta"][:, beta_idx], ax=axes[0, beta_idx + 1], color=colors[i], fill=True, alpha=alpha)

        # Local shrinkage
        for beta_idx in range(n_features):
            sns.kdeplot(y=np.log(samples["local_shrinkage"][:, beta_idx]), ax=axes[1, beta_idx + 1], color=colors[i], fill=True, alpha=alpha)

    # Intercept
    axes[0, 0].set_xticks([])
    axes[0, 0].set_xticklabels([])
    axes[0, 0].set_xlabel("Intercept")
    axes[0, 0].set_ylabel("")
    axes[0, 0].grid(True)

    # Global shrinkage
    axes[1, 0].set_xticks([])
    axes[1, 0].set_xticklabels([])
    axes[1, 0].set_xlabel("log-global\nshrinkage")
    axes[1, 0].set_ylabel("")
    axes[1, 0].grid(True)
    
    # Noise parameter
    axes[1, -1].set_xticks([])
    axes[1, -1].set_xticklabels([])
    axes[1, -1].set_xlabel("log-sigma")
    axes[1, -1].set_ylabel("")
    axes[1, -1].grid(True)

    # Remove the unused top right frame
    axes[0, 14].axis("off")

    for beta_idx, name in enumerate(feature_names):
        axes[0, beta_idx + 1].set_xticks([])
        axes[0, beta_idx + 1].set_xticklabels([])
        axes[0, beta_idx + 1].set_xlabel(name)
        axes[0, beta_idx + 1].set_ylabel("")
        axes[0, beta_idx + 1].grid(True)

        axes[1, beta_idx + 1].set_xticks([])
        axes[1, beta_idx + 1].set_xticklabels([])
        axes[1, beta_idx + 1].set_xlabel(name + " (log-ls)")
        axes[1, beta_idx + 1].set_ylabel("")
        axes[1, beta_idx + 1].grid(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_samples_kde_single_param(samples_orig, samples_target, samples_tainted, xlabel="", xlim=None,  ylim=None):
    """Plot a kde for the marginal distributions of one parameter"""

    samples_list = (samples_orig, samples_target, samples_tainted)
    labels = ["Original posterior", "Target posterior", "Tainted posterior"]

    plt.figure(figsize=(7, 7))
    alpha = 0.3

    for i in range(3):
        samples = samples_list[i]
        # Betas
        sns.kdeplot(x=samples, color=colors[i], fill=True, alpha=alpha, label=labels[i])

    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(xlabel, fontsize=AX_LBL_SIZE)
    plt.ylabel("Density", fontsize=AX_LBL_SIZE)
    plt.xticks(fontsize=TICKS_LBL_SIZE)
    plt.yticks(fontsize=TICKS_LBL_SIZE)
    plt.grid(True)

    plt.legend(fontsize=LEGEND_LBL_SIZE) 
    plt.tight_layout()
    plt.show()
    

def plot_intervals_NIG(beta_samples, beta_samples_target, beta_samples_tainted, ylim=None, title=""):
    """Plot credible intervals for parameters of the NIG model from samples"""
    
    samples_tuple = (beta_samples, beta_samples_target, beta_samples_tainted)
    x_shifts = np.linspace(-1, 1, 3) * 0.15
    labels = ["Original posterior", "Target posterior", "Tainted posterior"]
    x_axis = np.arange(1, n_features + 1) 
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, n_features]})

    for i in range(3):
        samples = samples_tuple[i]

        # Mean and credible intervals for betas
        beta_means = np.mean(samples, axis=0)
        beta_lower = np.percentile(samples, 2.5, axis=0)
        beta_upper = np.percentile(samples, 97.5, axis=0)
        
        axes[0].errorbar(x_shifts[i], beta_means[0], 
                        yerr=[[beta_means[0] - beta_lower[0]], [beta_upper[0] - beta_means[0]]], 
                        fmt='o', capsize=5, color=colors[i])
        axes[1].errorbar(x_axis + x_shifts[i], beta_means[1:], 
                        yerr=[beta_means[1:] - beta_lower[1:], beta_upper[1:] - beta_means[1:]], 
                        fmt='o', capsize=5, color=colors[i], label=labels[i])
        
    # Intercept
    axes[0].set_xticks([0])
    axes[0].set_xlim(*x_shifts[::2]*2)
    axes[0].set_xticklabels(["Intercept"])
    axes[0].set_ylabel('95% confidence interval')
    axes[0].grid(True)

    # Covariates
    axes[1].axhline(0, color='gray', linestyle='--')
    axes[1].set_xlabel('Covariate')
    axes[1].set_xticks(x_axis)
    axes[1].set_xticklabels(feature_names)
    axes[1].grid(True)

    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()

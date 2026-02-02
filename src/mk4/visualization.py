"""
MK4 Visualization - Plotting functions for results.

This module provides visualization functions for MK4 analysis
results, optimized for both exploratory analysis and publication.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, List
from pathlib import Path

# Set consistent style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_chaos_distribution(
    control_values: np.ndarray,
    disease_values: np.ndarray,
    title: str = "Chaos Distribution",
    xlabel: str = "Chaos Metric",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6)
) -> plt.Figure:
    """
    Plot chaos value distributions for control and disease groups.

    Creates overlapping histograms with KDE curves.

    Parameters
    ----------
    control_values : np.ndarray
        Chaos values for control group
    disease_values : np.ndarray
        Chaos values for disease group
    title : str
        Plot title
    xlabel : str
        X-axis label
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(control_values, bins=20, alpha=0.5,
            color='blue', label='Control', density=True)
    ax.hist(disease_values, bins=20, alpha=0.5,
            color='red', label='Disease', density=True)

    from scipy.stats import gaussian_kde

    x_range = np.linspace(
        min(control_values.min(), disease_values.min()),
        max(control_values.max(), disease_values.max()),
        200
    )

    kde_control = gaussian_kde(control_values)
    kde_disease = gaussian_kde(disease_values)

    ax.plot(x_range, kde_control(x_range), 'b-', linewidth=2)
    ax.plot(x_range, kde_disease(x_range), 'r-', linewidth=2)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_comparison_boxplot(
    control_values: np.ndarray,
    disease_values: np.ndarray,
    title: str = "Chaos Comparison",
    ylabel: str = "Chaos Metric",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 8)
) -> plt.Figure:
    """
    Plot box plot comparing control and disease groups.

    Parameters
    ----------
    control_values : np.ndarray
        Control group values
    disease_values : np.ndarray
        Disease group values
    title : str
        Plot title
    ylabel : str
        Y-axis label
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    data = [control_values, disease_values]
    labels = ['Control', 'Disease']

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    widths=0.6)

    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, values in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(values))
        ax.scatter(x, values, alpha=0.4, s=30, color='black')

    means = [np.mean(vals) for vals in data]
    ax.scatter([1, 2], means, color='red', s=100,
               marker='D', zorder=3, label='Mean')

    from scipy import stats
    t_stat, p_val = stats.ttest_ind(control_values, disease_values)

    y_max = max(control_values.max(), disease_values.max())
    y_range = y_max - min(control_values.min(), disease_values.min())

    bar_height = y_max + 0.05 * y_range
    ax.plot([1, 2], [bar_height, bar_height], 'k-', linewidth=1.5)

    if p_val < 0.001:
        sig_text = '***'
    elif p_val < 0.01:
        sig_text = '**'
    elif p_val < 0.05:
        sig_text = '*'
    else:
        sig_text = 'ns'

    ax.text(1.5, bar_height + 0.02 * y_range, sig_text,
            ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_roc_curve(
    roc_data: Dict[str, np.ndarray],
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8)
) -> plt.Figure:
    """
    Plot ROC curve.

    Parameters
    ----------
    roc_data : dict
        Dictionary with 'fpr', 'tpr', 'auc' keys
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(roc_data['fpr'], roc_data['tpr'],
            'b-', linewidth=2,
            label=f"MK4 (AUC = {roc_data['auc']:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1,
            label='Random (AUC = 0.500)')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_confusion_matrix(
    metrics: Dict[str, int],
    labels: List[str] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 6)
) -> plt.Figure:
    """
    Plot confusion matrix.

    Parameters
    ----------
    metrics : dict
        Dictionary with 'tn', 'fp', 'fn', 'tp' keys
    labels : list
        Class labels
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if labels is None:
        labels = ['Control', 'Disease']

    fig, ax = plt.subplots(figsize=figsize)

    cm = np.array([
        [metrics['tn'], metrics['fp']],
        [metrics['fn'], metrics['tp']]
    ])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar=True, square=True, ax=ax,
                annot_kws={'size': 16, 'weight': 'bold'})

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_results_summary(
    results: Dict,
    disease_name: str = "Disease",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 10)
) -> plt.Figure:
    """
    Create comprehensive results summary figure.

    Combines multiple plots into one figure.

    Parameters
    ----------
    results : dict
        Complete analysis results from analyze_disease_dataset()
    disease_name : str
        Name of disease for titles
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    stat = results['statistics']
    classif = results['classification']
    roc = results['roc']

    # Plot 1: ROC Curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(roc['fpr'], roc['tpr'], 'b-', linewidth=2,
             label=f"AUC = {roc['auc']:.3f}")
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    cm = np.array([
        [classif['tn'], classif['fp']],
        [classif['fn'], classif['tp']]
    ])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Control', disease_name],
                yticklabels=['Control', disease_name],
                cbar=False, square=True, ax=ax2,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax2.set_title('Confusion Matrix', fontweight='bold')

    # Plot 3: Performance Metrics Bar Chart
    ax3 = fig.add_subplot(gs[1, 0])
    metrics_names = ['Accuracy', 'Sensitivity', 'Specificity']
    metrics_values = [
        classif['accuracy'],
        classif['sensitivity'],
        classif['specificity']
    ]
    colors = ['steelblue', 'coral', 'lightgreen']
    bars = ax3.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    ax3.set_ylabel('Score')
    ax3.set_ylim([0, 1.1])
    ax3.set_title('Classification Performance', fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)

    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{value:.1%}', ha='center', va='bottom', fontweight='bold')

    # Plot 4: Statistics Summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    p_sig = '***' if stat['p_value'] < 0.001 else '**' if stat['p_value'] < 0.01 else '*' if stat['p_value'] < 0.05 else 'ns'

    summary_text = (
        f"STATISTICAL ANALYSIS SUMMARY\n"
        f"{'=' * 40}\n\n"
        f"Sample Sizes:\n"
        f"  Control: {results['n_control']}\n"
        f"  {disease_name}: {results['n_disease']}\n"
        f"  Balance: {results['balance'] * 100:.1f}% control\n\n"
        f"Group Comparison:\n"
        f"  Mean Control: {stat['mean_group1']:.4f}\n"
        f"  Mean {disease_name}: {stat['mean_group2']:.4f}\n"
        f"  Difference: {stat['mean_diff']:.4f}\n"
        f"  Effect Size (Cohen's d): {stat['effect_size']:.2f}\n"
        f"  p-value: {stat['p_value']:.4f} {p_sig}\n\n"
        f"Classification:\n"
        f"  Threshold: {results['threshold']:.4f}\n"
        f"  Accuracy: {classif['accuracy']:.1%}\n"
        f"  Sensitivity: {classif['sensitivity']:.1%}\n"
        f"  Specificity: {classif['specificity']:.1%}\n"
        f"  AUC: {roc['auc']:.3f}"
    )

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    fig.suptitle(f'MK4 Analysis: {disease_name} Detection',
                 fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def save_all_plots(
    control_values: np.ndarray,
    disease_values: np.ndarray,
    results: Dict,
    disease_name: str = "Disease",
    output_dir: str = "results/figures"
) -> None:
    """
    Generate and save all standard plots.

    Parameters
    ----------
    control_values : np.ndarray
        Control chaos values
    disease_values : np.ndarray
        Disease chaos values
    results : dict
        Analysis results
    disease_name : str
        Disease name for filenames
    output_dir : str
        Output directory path
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Generating plots for {disease_name}...")

    plot_chaos_distribution(
        control_values, disease_values,
        title=f"{disease_name} Chaos Distribution",
        save_path=f"{output_dir}/{disease_name.lower()}_distribution.png"
    )
    plt.close()

    plot_comparison_boxplot(
        control_values, disease_values,
        title=f"{disease_name} Chaos Comparison",
        save_path=f"{output_dir}/{disease_name.lower()}_boxplot.png"
    )
    plt.close()

    plot_roc_curve(
        results['roc'],
        title=f"{disease_name} ROC Curve",
        save_path=f"{output_dir}/{disease_name.lower()}_roc.png"
    )
    plt.close()

    plot_confusion_matrix(
        results['classification'],
        labels=['Control', disease_name],
        title=f"{disease_name} Confusion Matrix",
        save_path=f"{output_dir}/{disease_name.lower()}_confusion.png"
    )
    plt.close()

    plot_results_summary(
        results,
        disease_name=disease_name,
        save_path=f"{output_dir}/{disease_name.lower()}_summary.png"
    )
    plt.close()

    print(f"All plots saved to {output_dir}/")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    print("MK4 Visualization - Self Test")
    print("=" * 60)

    np.random.seed(42)
    control = np.random.normal(7.8, 0.2, 50)
    disease = np.random.normal(8.2, 0.2, 50)

    from src.mk4.analysis import analyze_disease_dataset
    results = analyze_disease_dataset(
        control, disease, verbose=False
    )

    print("\n1. Testing chaos distribution plot...")
    fig1 = plot_chaos_distribution(control, disease)
    print("   Distribution plot created")
    plt.close()

    print("\n2. Testing boxplot...")
    fig2 = plot_comparison_boxplot(control, disease)
    print("   Boxplot created")
    plt.close()

    print("\n3. Testing ROC curve...")
    fig3 = plot_roc_curve(results['roc'])
    print("   ROC curve created")
    plt.close()

    print("\n4. Testing confusion matrix...")
    fig4 = plot_confusion_matrix(results['classification'])
    print("   Confusion matrix created")
    plt.close()

    print("\n5. Testing results summary...")
    fig5 = plot_results_summary(results)
    print("   Summary plot created")
    plt.close()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

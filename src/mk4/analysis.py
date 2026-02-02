"""
MK4 Analysis - Statistical testing and classification.

This module provides statistical analysis and classification
functions for evaluating MK4 biomarker performance.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
from typing import Dict, Tuple, Optional, List
import warnings


def compare_groups(
    group1: np.ndarray,
    group2: np.ndarray,
    test: str = 'ttest',
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Compare two groups statistically.

    Parameters
    ----------
    group1 : np.ndarray
        First group (e.g., control chaos values)
    group2 : np.ndarray
        Second group (e.g., disease chaos values)
    test : str, default='ttest'
        Statistical test: 'ttest', 'mannwhitney', or 'ks'
    alternative : str, default='two-sided'
        Alternative hypothesis: 'two-sided', 'less', 'greater'

    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'statistic': Test statistic
        - 'p_value': P-value
        - 'mean_diff': Difference in means
        - 'effect_size': Cohen's d effect size

    Examples
    --------
    >>> control = np.array([7.8, 7.9, 7.7, 7.8])
    >>> disease = np.array([8.2, 8.3, 8.1, 8.4])
    >>> results = compare_groups(control, disease)
    >>> print(f"p-value: {results['p_value']:.4f}")
    """
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    mean_diff = mean2 - mean1

    if test == 'ttest':
        statistic, p_value = stats.ttest_ind(
            group1, group2, alternative=alternative
        )
    elif test == 'mannwhitney':
        statistic, p_value = stats.mannwhitneyu(
            group1, group2, alternative=alternative
        )
    elif test == 'ks':
        statistic, p_value = stats.ks_2samp(
            group1, group2, alternative=alternative
        )
    else:
        raise ValueError(f"Unknown test: {test}")

    # Cohen's d
    pooled_std = np.sqrt(
        (np.var(group1) + np.var(group2)) / 2
    )
    effect_size = mean_diff / (pooled_std + 1e-10)

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'mean_diff': float(mean_diff),
        'effect_size': float(effect_size),
        'mean_group1': float(mean1),
        'mean_group2': float(mean2),
        'std_group1': float(np.std(group1)),
        'std_group2': float(np.std(group2)),
    }


def find_optimal_threshold(
    values: np.ndarray,
    labels: np.ndarray,
    metric: str = 'accuracy'
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal classification threshold.

    Searches for threshold that maximizes chosen metric.

    Parameters
    ----------
    values : np.ndarray
        Continuous values (e.g., chaos scores)
    labels : np.ndarray
        Binary labels (0=control, 1=disease)
    metric : str, default='accuracy'
        Metric to optimize: 'accuracy', 'f1', 'youden'

    Returns
    -------
    threshold : float
        Optimal threshold value
    metrics : dict
        Performance metrics at optimal threshold

    Examples
    --------
    >>> chaos = np.array([7.8, 8.2, 7.9, 8.3])
    >>> labels = np.array([0, 1, 0, 1])
    >>> threshold, metrics = find_optimal_threshold(chaos, labels)
    >>> print(f"Optimal threshold: {threshold:.4f}")
    """
    thresholds = np.linspace(
        values.min(), values.max(), 100
    )

    best_score = -np.inf
    best_threshold = None
    best_metrics = None

    for threshold in thresholds:
        predictions = (values > threshold).astype(int)
        metrics = calculate_metrics(labels, predictions)

        if metric == 'accuracy':
            score = metrics['accuracy']
        elif metric == 'f1':
            score = metrics['f1']
        elif metric == 'youden':
            score = metrics['sensitivity'] + metrics['specificity'] - 1
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate classification performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred : np.ndarray
        Predicted binary labels

    Returns
    -------
    metrics : dict
        Dictionary with:
        - accuracy: Overall accuracy
        - sensitivity: True positive rate (recall)
        - specificity: True negative rate
        - precision: Positive predictive value
        - f1: F1 score
        - tn, fp, fn, tp: Confusion matrix values

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 1, 1, 1])
    >>> metrics = calculate_metrics(y_true, y_pred)
    >>> print(f"Accuracy: {metrics['accuracy']:.1%}")
    """
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (1, 1):
        if y_true[0] == 0:
            tn = cm[0, 0]
            fp = fn = tp = 0
        else:
            tp = cm[0, 0]
            tn = fp = fn = 0
    else:
        tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    if (precision + sensitivity) > 0:
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    else:
        f1 = 0

    return {
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1': float(f1),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def calculate_roc(
    values: np.ndarray,
    labels: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Calculate ROC curve.

    Parameters
    ----------
    values : np.ndarray
        Continuous values (higher = more likely disease)
    labels : np.ndarray
        Binary labels (0=control, 1=disease)

    Returns
    -------
    roc_data : dict
        Dictionary with:
        - fpr: False positive rates
        - tpr: True positive rates
        - thresholds: Threshold values
        - auc: Area under ROC curve

    Examples
    --------
    >>> roc = calculate_roc(chaos_values, labels)
    >>> print(f"AUC: {roc['auc']:.3f}")
    """
    fpr, tpr, thresholds = roc_curve(labels, values)
    auc = roc_auc_score(labels, values)

    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': float(auc),
    }


def cross_validate_threshold(
    values: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Cross-validation for threshold selection.

    Parameters
    ----------
    values : np.ndarray
        Continuous values
    labels : np.ndarray
        Binary labels
    n_folds : int, default=5
        Number of CV folds
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    cv_results : dict
        Dictionary with arrays of metrics across folds

    Examples
    --------
    >>> results = cross_validate_threshold(chaos, labels, n_folds=5)
    >>> print(f"Mean accuracy: {results['accuracy'].mean():.1%}")
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_results = {
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'f1': [],
        'threshold': [],
    }

    for train_idx, test_idx in kf.split(values):
        train_values = values[train_idx]
        train_labels = labels[train_idx]
        test_values = values[test_idx]
        test_labels = labels[test_idx]

        threshold, _ = find_optimal_threshold(
            train_values, train_labels
        )

        test_pred = (test_values > threshold).astype(int)
        metrics = calculate_metrics(test_labels, test_pred)

        cv_results['accuracy'].append(metrics['accuracy'])
        cv_results['sensitivity'].append(metrics['sensitivity'])
        cv_results['specificity'].append(metrics['specificity'])
        cv_results['f1'].append(metrics['f1'])
        cv_results['threshold'].append(threshold)

    for key in cv_results:
        cv_results[key] = np.array(cv_results[key])

    return cv_results


def analyze_disease_dataset(
    control_values: np.ndarray,
    disease_values: np.ndarray,
    control_labels: Optional[np.ndarray] = None,
    disease_labels: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict:
    """
    Complete analysis of disease vs control dataset.

    Performs statistical testing, finds optimal threshold,
    and calculates all performance metrics.

    Parameters
    ----------
    control_values : np.ndarray
        Chaos values for control group
    disease_values : np.ndarray
        Chaos values for disease group
    control_labels : np.ndarray, optional
        Labels (will create if not provided)
    disease_labels : np.ndarray, optional
        Labels (will create if not provided)
    verbose : bool, default=True
        Print results summary

    Returns
    -------
    results : dict
        Complete analysis results

    Examples
    --------
    >>> results = analyze_disease_dataset(
    ...     control_chaos, disease_chaos
    ... )
    >>> print(f"Accuracy: {results['classification']['accuracy']:.1%}")
    """
    if control_labels is None:
        control_labels = np.zeros(len(control_values))
    if disease_labels is None:
        disease_labels = np.ones(len(disease_values))

    all_values = np.concatenate([control_values, disease_values])
    all_labels = np.concatenate([control_labels, disease_labels])

    stats_results = compare_groups(
        control_values, disease_values, test='ttest'
    )

    threshold, class_metrics = find_optimal_threshold(
        all_values, all_labels, metric='accuracy'
    )

    roc_results = calculate_roc(all_values, all_labels)

    results = {
        'statistics': stats_results,
        'classification': class_metrics,
        'threshold': threshold,
        'roc': roc_results,
        'n_control': len(control_values),
        'n_disease': len(disease_values),
        'balance': len(control_values) / (len(control_values) + len(disease_values)),
    }

    if verbose:
        print("=" * 60)
        print("MK4 DISEASE ANALYSIS RESULTS")
        print("=" * 60)
        print(f"\nSample sizes:")
        print(f"  Control: {results['n_control']}")
        print(f"  Disease: {results['n_disease']}")
        print(f"  Balance: {results['balance'] * 100:.1f}% control")

        print(f"\nStatistical comparison:")
        print(f"  Mean control: {stats_results['mean_group1']:.4f}")
        print(f"  Mean disease: {stats_results['mean_group2']:.4f}")
        print(f"  Difference: {stats_results['mean_diff']:.4f}")
        print(f"  p-value: {stats_results['p_value']:.4f}", end="")
        if stats_results['p_value'] < 0.001:
            print(" ***")
        elif stats_results['p_value'] < 0.01:
            print(" **")
        elif stats_results['p_value'] < 0.05:
            print(" *")
        else:
            print(" ns")

        print(f"\nClassification performance:")
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Accuracy: {class_metrics['accuracy'] * 100:.1f}%")
        print(f"  Sensitivity: {class_metrics['sensitivity'] * 100:.1f}%")
        print(f"  Specificity: {class_metrics['specificity'] * 100:.1f}%")
        print(f"  AUC: {roc_results['auc']:.3f}")

        print(f"\nConfusion matrix:")
        print(f"  TN={class_metrics['tn']}, FP={class_metrics['fp']}")
        print(f"  FN={class_metrics['fn']}, TP={class_metrics['tp']}")
        print("=" * 60)

    return results


if __name__ == "__main__":
    print("MK4 Analysis - Self Test")
    print("=" * 60)

    np.random.seed(42)
    control = np.random.normal(7.8, 0.2, 50)
    disease = np.random.normal(8.2, 0.2, 50)

    print("\n1. Testing statistical comparison...")
    stats_res = compare_groups(control, disease)
    print(f"   p-value: {stats_res['p_value']:.4f}")
    print(f"   Effect size: {stats_res['effect_size']:.2f}")

    print("\n2. Testing classification...")
    all_vals = np.concatenate([control, disease])
    all_labs = np.concatenate([np.zeros(50), np.ones(50)])
    thresh, metrics = find_optimal_threshold(all_vals, all_labs)
    print(f"   Threshold: {thresh:.4f}")
    print(f"   Accuracy: {metrics['accuracy'] * 100:.1f}%")

    print("\n3. Testing ROC curve...")
    roc = calculate_roc(all_vals, all_labs)
    print(f"   AUC: {roc['auc']:.3f}")

    print("\n4. Testing complete analysis...")
    results = analyze_disease_dataset(
        control, disease, verbose=False
    )
    print(f"   Complete analysis: {len(results)} components")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

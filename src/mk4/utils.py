"""
MK4 Utilities - Data loading, validation, and preprocessing.

This module provides utility functions for working with
RNA-seq expression data in MK4 analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, Optional
import warnings


def load_expression_data(
    filepath: Union[str, Path],
    sample_axis: int = 0
) -> np.ndarray:
    """
    Load gene expression data from file.

    Supports CSV and numpy formats. Auto-detects format from
    file extension.

    Parameters
    ----------
    filepath : str or Path
        Path to data file
    sample_axis : int, default=0
        Which axis represents samples (0=rows, 1=columns)

    Returns
    -------
    expression_matrix : np.ndarray
        Expression data, shape (n_samples, n_genes)

    Notes
    -----
    For CSV files, assumes:
    - First row: gene names (skipped)
    - First column: sample IDs (used as index)
    - Remaining: expression values
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.suffix == '.npy':
        data = np.load(filepath)
    elif filepath.suffix == '.npz':
        loaded = np.load(filepath)
        if 'expression' in loaded:
            data = loaded['expression']
        elif 'data' in loaded:
            data = loaded['data']
        else:
            data = loaded[loaded.files[0]]
    elif filepath.suffix in ['.csv', '.txt', '.tsv']:
        df = pd.read_csv(filepath, index_col=0)
        data = df.values
    else:
        raise ValueError(
            f"Unsupported file format: {filepath.suffix}. "
            f"Supported: .npy, .npz, .csv, .txt, .tsv"
        )

    if len(data.shape) == 1:
        data = data.reshape(1, -1)

    if sample_axis == 1:
        data = data.T

    print(f"Loaded {filepath.name}: {data.shape[0]} samples, "
          f"{data.shape[1]} genes")

    return data


def load_labels(
    filepath: Union[str, Path]
) -> np.ndarray:
    """
    Load sample labels from file.

    Parameters
    ----------
    filepath : str or Path
        Path to labels file

    Returns
    -------
    labels : np.ndarray
        Binary labels (0=control, 1=disease)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)

    label_cols = ['label', 'class', 'group', 'disease', 'Label']
    label_col = None

    for col in label_cols:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    labels = df[label_col].values

    if labels.dtype == object or labels.dtype.name.startswith('str'):
        unique_vals = np.unique(labels)
        if len(unique_vals) != 2:
            warnings.warn(
                f"Expected 2 classes, found {len(unique_vals)}: {unique_vals}"
            )
        label_map = {unique_vals[0]: 0, unique_vals[1]: 1}
        labels = np.array([label_map[val] for val in labels])

    labels = labels.astype(int)

    print(f"Loaded labels: {len(labels)} samples "
          f"({labels.sum()} disease, {len(labels) - labels.sum()} control)")

    return labels


def validate_expression_data(
    expression_matrix: np.ndarray,
    min_genes: int = 1000,
    min_samples: int = 5,
    check_nan: bool = True,
    check_inf: bool = True
) -> None:
    """
    Validate expression data quality.

    Parameters
    ----------
    expression_matrix : np.ndarray
        Expression data to validate
    min_genes : int, default=1000
        Minimum number of genes required
    min_samples : int, default=5
        Minimum number of samples required
    check_nan : bool, default=True
        Check for NaN values
    check_inf : bool, default=True
        Check for infinite values

    Raises
    ------
    ValueError
        If validation fails
    """
    if len(expression_matrix.shape) != 2:
        raise ValueError(
            f"Expected 2D array, got shape {expression_matrix.shape}"
        )

    n_samples, n_genes = expression_matrix.shape

    if n_samples < min_samples:
        raise ValueError(
            f"Too few samples: {n_samples} < {min_samples}"
        )

    if n_genes < min_genes:
        warnings.warn(
            f"Few genes: {n_genes} < {min_genes}. Results may be unreliable."
        )

    if check_nan and np.any(np.isnan(expression_matrix)):
        n_nan = np.isnan(expression_matrix).sum()
        raise ValueError(
            f"Data contains {n_nan} NaN values. "
            f"Please handle missing data before analysis."
        )

    if check_inf and np.any(np.isinf(expression_matrix)):
        n_inf = np.isinf(expression_matrix).sum()
        raise ValueError(
            f"Data contains {n_inf} infinite values. "
            f"Please check data quality."
        )

    print(f"Validation passed: {n_samples} samples, {n_genes} genes")


def preprocess_expression_data(
    expression_matrix: np.ndarray,
    remove_low_variance: bool = True,
    variance_percentile: float = 10.0,
    fill_nan: bool = True,
    normalize: bool = False
) -> np.ndarray:
    """
    Preprocess expression data.

    Parameters
    ----------
    expression_matrix : np.ndarray
        Raw expression data
    remove_low_variance : bool, default=True
        Remove low-variance genes
    variance_percentile : float, default=10.0
        Percentile threshold for variance filtering
    fill_nan : bool, default=True
        Fill NaN with median
    normalize : bool, default=False
        Z-score normalize per sample

    Returns
    -------
    processed_matrix : np.ndarray
        Preprocessed expression data
    """
    data = expression_matrix.copy()
    original_genes = data.shape[1]

    if fill_nan and np.any(np.isnan(data)):
        print("  Filling NaN values with column medians...")
        for col in range(data.shape[1]):
            col_data = data[:, col]
            if np.any(np.isnan(col_data)):
                median_val = np.nanmedian(col_data)
                col_data[np.isnan(col_data)] = median_val
                data[:, col] = col_data

    if remove_low_variance:
        variances = np.var(data, axis=0)
        threshold = np.percentile(variances, variance_percentile)
        high_var_mask = variances > threshold
        data = data[:, high_var_mask]
        print(f"  Removed low-variance genes: {original_genes} -> {data.shape[1]}")

    if normalize:
        print("  Z-score normalizing samples...")
        data = (data - data.mean(axis=1, keepdims=True)) / \
               (data.std(axis=1, keepdims=True) + 1e-10)

    print(f"Preprocessing complete: {data.shape[0]} samples, "
          f"{data.shape[1]} genes")

    return data


def save_results(
    results: dict,
    output_dir: Union[str, Path],
    prefix: str = "mk4"
) -> None:
    """
    Save analysis results to files.

    Parameters
    ----------
    results : dict
        Results dictionary from MK4Analyzer
    output_dir : str or Path
        Output directory
    prefix : str
        Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to only serializable arrays/scalars for CSV
    csv_data = {k: v for k, v in results.items()
                if isinstance(v, (np.ndarray, list, int, float))}
    df = pd.DataFrame(csv_data)
    csv_path = output_dir / f"{prefix}_results.csv"
    df.to_csv(csv_path, index=False)

    npz_path = output_dir / f"{prefix}_results.npz"
    np.savez(npz_path, **{k: v for k, v in results.items()
                          if isinstance(v, np.ndarray)})

    print(f"Saved results to {output_dir}/")
    print(f"  - {csv_path.name}")
    print(f"  - {npz_path.name}")


def create_balanced_dataset(
    expression_matrix: np.ndarray,
    labels: np.ndarray,
    balance_ratio: float = 0.5,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create balanced dataset by undersampling majority class.

    Parameters
    ----------
    expression_matrix : np.ndarray
        Expression data
    labels : np.ndarray
        Binary labels
    balance_ratio : float, default=0.5
        Target ratio of minority class
    random_state : int, optional
        Random seed

    Returns
    -------
    balanced_data : np.ndarray
        Balanced expression data
    balanced_labels : np.ndarray
        Balanced labels
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_class1 = int(labels.sum())
    n_class0 = len(labels) - n_class1

    if n_class1 < n_class0:
        minority_idx = np.where(labels == 1)[0]
        majority_idx = np.where(labels == 0)[0]
    else:
        minority_idx = np.where(labels == 0)[0]
        majority_idx = np.where(labels == 1)[0]

    n_target_majority = int(len(minority_idx) / balance_ratio - len(minority_idx))

    if n_target_majority < len(majority_idx):
        majority_sample = np.random.choice(
            majority_idx, size=n_target_majority, replace=False
        )
    else:
        majority_sample = majority_idx

    selected_idx = np.concatenate([minority_idx, majority_sample])
    np.random.shuffle(selected_idx)

    balanced_data = expression_matrix[selected_idx]
    balanced_labels = labels[selected_idx]

    n_c0 = (balanced_labels == 0).sum()
    n_c1 = (balanced_labels == 1).sum()

    print(f"Balanced dataset: {n_c0} control, {n_c1} disease "
          f"({n_c1 / (n_c0 + n_c1) * 100:.1f}% disease)")

    return balanced_data, balanced_labels


def get_version() -> str:
    """Get MK4 version."""
    return "0.1.0"


def print_summary(
    expression_matrix: np.ndarray,
    labels: Optional[np.ndarray] = None
) -> None:
    """
    Print dataset summary.

    Parameters
    ----------
    expression_matrix : np.ndarray
        Expression data
    labels : np.ndarray, optional
        Sample labels
    """
    n_samples, n_genes = expression_matrix.shape

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Samples: {n_samples}")
    print(f"Genes: {n_genes}")

    if labels is not None:
        n_disease = int(labels.sum())
        n_control = len(labels) - n_disease
        print(f"Control: {n_control}")
        print(f"Disease: {n_disease}")
        print(f"Balance: {n_control / (n_control + n_disease) * 100:.1f}% control")

    print(f"\nExpression range: [{expression_matrix.min():.2f}, "
          f"{expression_matrix.max():.2f}]")
    print(f"Mean: {expression_matrix.mean():.2f}")
    print(f"Std: {expression_matrix.std():.2f}")
    print("=" * 60)


if __name__ == "__main__":
    print("MK4 Utils - Self Test")
    print("=" * 60)

    np.random.seed(42)
    test_expression = np.random.randn(50, 10000) * 2 + 5
    test_labels = np.array([0] * 25 + [1] * 25)

    print("\n1. Testing data validation...")
    validate_expression_data(test_expression)

    print("\n2. Testing preprocessing...")
    processed = preprocess_expression_data(
        test_expression,
        remove_low_variance=True
    )

    print("\n3. Testing balanced dataset creation...")
    balanced_X, balanced_y = create_balanced_dataset(
        test_expression, test_labels,
        balance_ratio=0.5,
        random_state=42
    )

    print("\n4. Testing summary...")
    print_summary(test_expression, test_labels)

    print("\n5. Testing version...")
    version = get_version()
    print(f"   Version: {version}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

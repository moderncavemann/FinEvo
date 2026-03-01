"""
Utility Functions - ECCV 2026

Common utilities for EconAgent simulation
"""

import re
import numpy as np
from datetime import datetime
from typing import List


# Tax brackets (monthly, scaled from annual)
BRACKETS = list(np.array([0, 97, 394.75, 842, 1607.25, 2041, 5103]) * 100 / 12)

# Simulation start time
WORLD_START_TIME = datetime.strptime('2001.01', '%Y.%m')


def prettify_document(document: str) -> str:
    """
    Clean and normalize text by removing excess whitespace

    Args:
        document: Input text

    Returns:
        Cleaned text with single spaces
    """
    cleaned = re.sub(r'\s+', ' ', document).strip()
    return cleaned


def format_numbers(numbers: List[float]) -> str:
    """
    Format list of numbers for display

    Args:
        numbers: List of numbers

    Returns:
        Formatted string like '[1.00, 2.00, 3.00]'
    """
    return '[' + ', '.join('{:.2f}'.format(num) for num in numbers) + ']'


def format_percentages(numbers: List[float]) -> str:
    """
    Format list of numbers as percentages

    Args:
        numbers: List of numbers (0-1 range)

    Returns:
        Formatted string like '[10.00%, 20.00%, 30.00%]'
    """
    return '[' + ', '.join('{:.2%}'.format(num) for num in numbers) + ']'


def compute_gini(wealths: np.ndarray) -> float:
    """
    Compute Gini coefficient for wealth distribution

    Args:
        wealths: Array of wealth values

    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    wealths = np.sort(wealths)
    n = len(wealths)
    if n == 0 or np.sum(wealths) == 0:
        return 0.0
    cumx = np.cumsum(wealths)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def compute_summary_stats(values: np.ndarray) -> dict:
    """
    Compute summary statistics for an array

    Args:
        values: Input array

    Returns:
        Dict with mean, median, std, min, max
    """
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }

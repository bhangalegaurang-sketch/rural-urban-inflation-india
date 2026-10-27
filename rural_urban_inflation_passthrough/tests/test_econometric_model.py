"""
Tests for econometric model and data processing.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.econometric_model import (
    load_and_preprocess_data,
    compute_descriptive_statistics,
    fit_econometric_model,
    get_cpi_basket_weights,
)

RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "cpi_inflation_cleaned.csv"


@pytest.fixture
def clean_data():
    return load_and_preprocess_data(raw_csv_path=RAW_CSV_PATH)


def test_data_loading(clean_data):
    df = clean_data
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 142
    assert "cpi_rural" in df.columns
    assert "cpi_urban" in df.columns
    assert "wedge" in df.columns
    assert not df.isnull().values.any()


def test_descriptive_statistics(clean_data):
    df = clean_data
    stats = compute_descriptive_statistics(df)
    assert isinstance(stats, pd.DataFrame)
    assert len(stats) == 3
    
    rural_stats = stats.loc["Rural Inflation (π^Rural)"]
    urban_stats = stats.loc["Urban Inflation (π^Urban)"]
    
    # Assert statistical values
    assert np.isclose(rural_stats["Mean (%)"], 4.94, atol=0.05)
    assert np.isclose(urban_stats["Mean (%)"], 4.82, atol=0.05)
    assert np.isclose(rural_stats["Std Dev (%)"], 1.94, atol=0.05)
    assert np.isclose(urban_stats["Std Dev (%)"], 1.54, atol=0.05)


def test_econometric_model(clean_data):
    df = clean_data
    results = fit_econometric_model(df)
    
    # Check estimated coefficients and fit
    assert np.isclose(results.beta_hat, 1.0813, atol=0.01)
    assert np.isclose(results.alpha_hat, -0.2759, atol=0.01)
    assert np.isclose(results.r_squared, 0.7349, atol=0.01)
    assert np.isclose(results.durbin_watson_stat, 0.1852, atol=0.01)
    
    # Check parity test p-value (> 0.05 fails to reject H0)
    assert results.wald_pval > 0.05
    assert not results.parity_rejected_5pct


def test_basket_weights():
    weights_df = get_cpi_basket_weights()
    assert isinstance(weights_df, pd.DataFrame)
    assert len(weights_df) == 7  # 6 commodity categories + 1 Total row
    
    # Check food and housing weights
    food_row = weights_df[weights_df["Group / Category"].str.contains("Food")].iloc[0]
    assert np.isclose(food_row["Rural Weight (%)"], 54.18)
    assert np.isclose(food_row["Urban Weight (%)"], 36.29)
    assert np.isclose(food_row["Rural - Urban Wedge (pp)"], 17.89)

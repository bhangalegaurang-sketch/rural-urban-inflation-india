"""
Econometric Model Module: Rural–Urban Inflation Pass-Through Analysis
Author: Economics Researcher, Gokhale Institute of Politics and Economics (GIPE)
Specification: \\pi_t^{Rural} = \\alpha + \\beta \\pi_t^{Urban} + \\varepsilon_t
Hypothesis: H_0: \\alpha = 0, \\beta = 1 (45-degree Parity)
Inference: OLS with Newey-West (1987) Heteroskedasticity & Autocorrelation Consistent (HAC) Standard Errors
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson


@dataclass
class EconometricResults:
    """Container for econometric regression and hypothesis test results."""
    n_obs: int
    alpha_hat: float
    beta_hat: float
    alpha_se_ols: float
    beta_se_ols: float
    alpha_t_ols: float
    beta_t_ols: float
    alpha_pval_ols: float
    beta_pval_ols: float
    
    # Newey-West HAC Robust Inference
    alpha_se_hac: float
    beta_se_hac: float
    alpha_z_hac: float
    beta_z_hac: float
    alpha_pval_hac: float
    beta_pval_hac: float
    hac_maxlags: int
    
    # Goodness of Fit & Diagnostics
    r_squared: float
    adj_r_squared: float
    f_stat: float
    f_pval: float
    residual_std_error: float
    durbin_watson_stat: float
    jarque_bera_stat: float
    jarque_bera_pval: float
    
    # 45-Degree Parity Hypothesis Test (H0: alpha = 0, beta = 1)
    wald_f_stat: float
    wald_pval: float
    wald_df_num: int
    wald_df_denom: int
    parity_rejected_5pct: bool
    
    # Raw statsmodels objects
    ols_model: sm.regression.linear_model.RegressionResultsWrapper
    hac_model: sm.regression.linear_model.RegressionResultsWrapper


def load_and_preprocess_data(
    raw_csv_path: str | Path,
    processed_csv_path: Optional[str | Path] = None
) -> pd.DataFrame:
    """
    Ingests official MoSPI CPI inflation data, filters rural and urban series,
    merges the contemporaneous time series, and optionally saves processed data.
    
    Parameters:
    -----------
    raw_csv_path : str or Path
        Path to raw cpi_inflation_cleaned.csv
    processed_csv_path : str or Path, optional
        Path where processed CSV will be written
        
    Returns:
    --------
    pd.DataFrame
        Clean merged DataFrame with columns ['date', 'cpi_rural', 'cpi_urban', 'wedge']
    """
    raw_path = Path(raw_csv_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")
        
    df = pd.read_csv(raw_path)
    
    # Filter Sector General-Overall Headline YoY Inflation
    rural = (
        df[(df["sector"] == "Rural") & (df["subgroup"] == "General-Overall")][["date", "inflation"]]
        .rename(columns={"inflation": "cpi_rural"})
        .dropna()
    )
    urban = (
        df[(df["sector"] == "Urban") & (df["subgroup"] == "General-Overall")][["date", "inflation"]]
        .rename(columns={"inflation": "cpi_urban"})
        .dropna()
    )
    
    # Synchronized contemporaneous merge
    merged = pd.merge(rural, urban, on="date").sort_values("date").reset_index(drop=True)
    merged["cpi_rural"] = pd.to_numeric(merged["cpi_rural"])
    merged["cpi_urban"] = pd.to_numeric(merged["cpi_urban"])
    merged["wedge"] = merged["cpi_rural"] - merged["cpi_urban"]
    
    if processed_csv_path is not None:
        out_path = Path(processed_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)
        
    return merged


def compute_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rigorous descriptive moments for Rural CPI, Urban CPI, and Inter-sectoral Wedge.
    
    Returns:
    --------
    pd.DataFrame
        Statistical summary table with moments (mean, std, median, IQR, skew, kurtosis, etc.)
    """
    stats_dict = {}
    for col, label in [
        ("cpi_rural", "Rural Inflation (π^Rural)"),
        ("cpi_urban", "Urban Inflation (π^Urban)"),
        ("wedge", "Inflation Wedge (Rural - Urban)")
    ]:
        series = df[col]
        mean_val = float(series.mean())
        std_val = float(series.std(ddof=1))
        median_val = float(series.median())
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr_val = q75 - q25
        min_val = float(series.min())
        max_val = float(series.max())
        skew_val = float(stats.skew(series, bias=False))
        kurt_val = float(stats.kurtosis(series, bias=False)) # Excess kurtosis
        jb_stat, jb_pval = stats.jarque_bera(series)
        
        stats_dict[label] = {
            "N": len(series),
            "Mean (%)": mean_val,
            "Std Dev (%)": std_val,
            "Median (%)": median_val,
            "IQR (%)": iqr_val,
            "Min (%)": min_val,
            "Max (%)": max_val,
            "Skewness": skew_val,
            "Excess Kurtosis": kurt_val,
            "Jarque-Bera Stat": float(jb_stat),
            "JB p-value": float(jb_pval)
        }
        
    return pd.DataFrame(stats_dict).T


def fit_econometric_model(df: pd.DataFrame, maxlags: int = 4) -> EconometricResults:
    """
    Fits the Bivariate OLS regression model:
        \\pi_t^{Rural} = \\alpha + \\beta \\pi_t^{Urban} + \\varepsilon_t
        
    Calculates:
    1. Classical OLS standard errors
    2. Newey-West (1987) HAC standard errors (accounting for temporal serial correlation)
    3. Joint Wald test for 45° Parity hypothesis:
        H_0: \\alpha = 0 and \\beta = 1
        
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'cpi_rural' and 'cpi_urban'
    maxlags : int, default 4
        Maximum lag length for Newey-West kernel truncation (rule of thumb: ~4 for monthly data)
        
    Returns:
    --------
    EconometricResults
        Full results dataclass
    """
    y = df["cpi_rural"]
    X = sm.add_constant(df["cpi_urban"])
    
    # 1. Classical OLS
    ols_res = sm.OLS(y, X).fit()
    
    # 2. Newey-West (HAC) Robust Standard Errors
    hac_res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    
    # Parameter estimates
    alpha_hat = float(ols_res.params.iloc[0])
    beta_hat = float(ols_res.params.iloc[1])
    
    # Classical OLS SE & tests
    alpha_se_ols = float(ols_res.bse.iloc[0])
    beta_se_ols = float(ols_res.bse.iloc[1])
    alpha_t_ols = float(ols_res.tvalues.iloc[0])
    beta_t_ols = float(ols_res.tvalues.iloc[1])
    alpha_pval_ols = float(ols_res.pvalues.iloc[0])
    beta_pval_ols = float(ols_res.pvalues.iloc[1])
    
    # HAC SE & tests
    alpha_se_hac = float(hac_res.bse.iloc[0])
    beta_se_hac = float(hac_res.bse.iloc[1])
    alpha_z_hac = float(hac_res.tvalues.iloc[0])
    beta_z_hac = float(hac_res.tvalues.iloc[1])
    alpha_pval_hac = float(hac_res.pvalues.iloc[0])
    beta_pval_hac = float(hac_res.pvalues.iloc[1])
    
    # Residual diagnostics
    residuals = ols_res.resid
    dw = float(durbin_watson(residuals))
    jb_stat, jb_pval, _, _ = sm.stats.jarque_bera(residuals)
    
    # 3. 45-Degree Parity Joint Wald Test: H0: alpha = 0, beta = 1
    # Formulate linear restriction matrix: R * theta = r
    # [1  0] [alpha] = [0]
    # [0  1] [beta ]   [1]
    wald_test = ols_res.f_test("const = 0, cpi_urban = 1")
    wald_f = float(np.asarray(wald_test.fvalue).item())
    wald_p = float(np.asarray(wald_test.pvalue).item())
    wald_df_num = int(np.asarray(wald_test.df_num).item())
    wald_df_denom = int(np.asarray(wald_test.df_denom).item())
    
    return EconometricResults(
        n_obs=len(df),
        alpha_hat=alpha_hat,
        beta_hat=beta_hat,
        alpha_se_ols=alpha_se_ols,
        beta_se_ols=beta_se_ols,
        alpha_t_ols=alpha_t_ols,
        beta_t_ols=beta_t_ols,
        alpha_pval_ols=alpha_pval_ols,
        beta_pval_ols=beta_pval_ols,
        alpha_se_hac=alpha_se_hac,
        beta_se_hac=beta_se_hac,
        alpha_z_hac=alpha_z_hac,
        beta_z_hac=beta_z_hac,
        alpha_pval_hac=alpha_pval_hac,
        beta_pval_hac=beta_pval_hac,
        hac_maxlags=maxlags,
        r_squared=float(ols_res.rsquared),
        adj_r_squared=float(ols_res.rsquared_adj),
        f_stat=float(ols_res.fvalue),
        f_pval=float(ols_res.f_pvalue),
        residual_std_error=float(np.sqrt(ols_res.scale)),
        durbin_watson_stat=dw,
        jarque_bera_stat=float(jb_stat),
        jarque_bera_pval=float(jb_pval),
        wald_f_stat=wald_f,
        wald_pval=wald_p,
        wald_df_num=wald_df_num,
        wald_df_denom=wald_df_denom,
        parity_rejected_5pct=(wald_p < 0.05),
        ols_model=ols_res,
        hac_model=hac_res
    )


def get_cpi_basket_weights() -> pd.DataFrame:
    """
    Returns the official MoSPI CPI (Base 2012=100) commodity group weights
    across Rural, Urban, and Combined sectors.
    """
    weights_data = {
        "Group / Category": [
            "1. Food and Beverages",
            "2. Pan, Tobacco and Intoxicants",
            "3. Clothing and Footwear",
            "4. Housing",
            "5. Fuel and Light",
            "6. Miscellaneous (Services, Transport, Health, Education)",
            "Total General Index"
        ],
        "Rural Weight (%)": [54.18, 3.26, 7.36, 0.00, 7.94, 27.26, 100.00],
        "Urban Weight (%)": [36.29, 1.36, 5.57, 21.67, 5.58, 29.53, 100.00],
        "Combined Weight (%)": [45.86, 2.38, 6.53, 10.07, 6.84, 28.32, 100.00],
        "Rural - Urban Wedge (pp)": [+17.89, +1.90, +1.79, -21.67, +2.36, -2.27, 0.00]
    }
    return pd.DataFrame(weights_data)

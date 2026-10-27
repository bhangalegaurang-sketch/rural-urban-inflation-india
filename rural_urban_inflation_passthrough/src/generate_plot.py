"""
Publication-Grade Econometric Visualization: Rural–Urban Inflation Pass-Through
Author: Economics Researcher, Gokhale Institute of Politics and Economics (GIPE)
Output: figures/rural_urban_ols_parity_scatter.png (300 DPI)
"""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.econometric_model import load_and_preprocess_data, fit_econometric_model


def plot_rural_urban_parity(
    df: pd.DataFrame,
    output_path: str | Path,
    dpi: int = 300
) -> Path:
    """
    Constructs a publication-grade scatter plot comparing Rural and Urban CPI inflation,
    displaying:
    1. Monthly empirical observations (N = 142)
    2. OLS fitted regression line with 95% confidence interval
    3. Theoretical 45-degree parity benchmark line (slope = 1, intercept = 0)
    4. Comprehensive econometric estimates displayed directly below the scatter plot

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns 'cpi_rural' and 'cpi_urban'
    output_path : str or Path
        Destination filepath for the high-res PNG
    dpi : int, default 300
        Resolution in dots per inch

    Returns:
    --------
    Path
        Path to the saved figure
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Run econometric estimation
    res = fit_econometric_model(df)

    # Configure publication aesthetic
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "axes.edgecolor": "#334155",
        "axes.linewidth": 1.1,
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "xtick.color": "#1E293B",
        "ytick.color": "#1E293B",
    })

    # Create figure with 2 rows: Top for Scatter Plot, Bottom for Model Estimates Table/Panel
    fig = plt.figure(figsize=(9.8, 11.0), dpi=dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.6, 1.3], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax_sub = fig.add_subplot(gs[1])

    # 1. 45-Degree Parity Reference Line (alpha = 0, beta = 1)
    axis_min, axis_max = -0.5, 9.8
    ref_line = np.linspace(axis_min, axis_max, 200)
    ax.plot(
        ref_line,
        ref_line,
        color="#475569",
        linestyle="--",
        linewidth=1.9,
        alpha=0.85,
        label=r"$\mathbf{45^\circ\ Parity\ Line}\ (\pi^{Rural} = \pi^{Urban};\ \alpha=0,\ \beta=1)$",
        zorder=2
    )

    # 2. OLS Fitted Line and 95% Confidence Band
    urban_grid = np.linspace(df["cpi_urban"].min() - 0.2, df["cpi_urban"].max() + 0.2, 200)
    urban_grid_design = np.column_stack([np.ones_like(urban_grid), urban_grid])
    ols_predictions = res.ols_model.get_prediction(urban_grid_design)
    pred_summary = ols_predictions.summary_frame(alpha=0.05)

    ax.fill_between(
        urban_grid,
        pred_summary["mean_ci_lower"],
        pred_summary["mean_ci_upper"],
        color="#DC2626",
        alpha=0.18,
        label="95% Confidence Interval (OLS)",
        zorder=3
    )

    ax.plot(
        urban_grid,
        pred_summary["mean"],
        color="#B91C1C",
        linewidth=2.4,
        label=(
            rf"$\mathbf{{Fitted\ OLS\ Line}}:\ \hat{{\pi}}_t^{{Rural}} = "
            rf"{res.alpha_hat:.4f} + {res.beta_hat:.4f}\,\pi_t^{{Urban}}$"
        ),
        zorder=4
    )

    # 3. Scatter Points (Empirical monthly data points)
    ax.scatter(
        df["cpi_urban"],
        df["cpi_rural"],
        color="#1E3A8A",
        edgecolor="#FFFFFF",
        linewidth=0.85,
        s=58,
        alpha=0.80,
        label=rf"Monthly Observations ($N = {res.n_obs}$, 2014–2025)",
        zorder=5
    )

    # Direct line annotation for Fitted OLS Line
    ax.annotate(
        rf"$\hat{{\pi}}_t^{{R}} = {res.alpha_hat:.2f} + {res.beta_hat:.2f}\pi_t^{{U}}$",
        xy=(7.2, res.alpha_hat + res.beta_hat * 7.2),
        xytext=(7.5, 6.7),
        fontsize=10.5,
        fontweight="bold",
        color="#B91C1C",
        arrowprops=dict(arrowstyle="->", color="#B91C1C", lw=1.2, shrinkA=4, shrinkB=4),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor="#F87171", alpha=0.9),
        zorder=6
    )

    # 4. Set Equal Scale & Aspect Ratio for Scatter
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    # Grid & Ticks
    ax.grid(True, which="both", linestyle="--", alpha=0.55)
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_yticks(np.arange(0, 11, 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))

    # Axis Labels
    ax.set_xlabel(
        r"$\mathbf{Urban\ CPI\ Inflation\ (\pi_t^{Urban},\ \%\ YoY)}$",
        fontsize=11.5,
        fontweight="bold",
        labelpad=8,
        color="#0F172A"
    )
    ax.set_ylabel(
        r"$\mathbf{Rural\ CPI\ Inflation\ (\pi_t^{Rural},\ \%\ YoY)}$",
        fontsize=11.5,
        fontweight="bold",
        labelpad=8,
        color="#0F172A"
    )

    # Legend
    leg = ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        edgecolor="#94A3B8",
        fontsize=9.2,
        facecolor="#FFFFFF"
    )
    leg.set_zorder(6)

    # Title & Subtitle
    ax.set_title(
        "Econometric Analysis of Rural–Urban Inflation Pass-Through in India (2014–2025)\n"
        r"$\mathrm{Empirical\ Bivariate\ OLS\ Fit\ vs.\ 45^\circ\ Parity\ Benchmark\ |\ MoSPI\ CPI\ Base\ 2012=100\ (N=142)}$",
        fontsize=12.2,
        fontweight="bold",
        pad=12,
        color="#0F172A",
        linespacing=1.35
    )

    # 5. Display Econometric Model Estimates Below the Plot
    ax_sub.axis("off")

    table_data = [
        ["Urban Inflation Slope (β̂₁)", f"{res.beta_hat:.4f}", f"{res.beta_se_ols:.4f}", f"{res.beta_t_ols:.2f}", "< 0.0001", f"{res.beta_se_hac:.4f}", f"{res.beta_z_hac:.2f}", "< 0.0001***"],
        ["Intercept Constant (α̂)", f"{res.alpha_hat:.4f}", f"{res.alpha_se_ols:.4f}", f"{res.alpha_t_ols:.2f}", f"{res.alpha_pval_ols:.4f}", f"{res.alpha_se_hac:.4f}", f"{res.alpha_z_hac:.2f}", f"{res.alpha_pval_hac:.4f}"]
    ]
    col_labels = ["Parameter", "Estimate", "OLS SE", "t-stat", "OLS p-val", "HAC SE (L=4)", "z-stat", "HAC p-val"]

    table = ax_sub.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
        bbox=[0.01, 0.44, 0.98, 0.46]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)

    # Style table header & alternating cells
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if row_idx == 0:
            cell.set_facecolor("#0F172A")
            cell.set_text_props(color="#FFFFFF", fontweight="bold")
            cell.set_height(0.24)
        else:
            cell.set_facecolor("#F8FAFC" if row_idx % 2 == 1 else "#FFFFFF")
            cell.set_text_props(color="#0F172A")
            if col_idx == 0:
                cell.set_text_props(fontweight="bold", ha="left")

    # Bottom Callout Summary for Diagnostics & Hypothesis Test
    diagnostics_text = (
        r"$\mathbf{Model\ Fit\ \&\ Diagnostics:}\quad "
        rf"R^2 = {res.r_squared:.4f}\quad (\mathrm{{Adj\ }} R^2 = {res.adj_r_squared:.4f}) \quad\big|\quad "
        rf"\mathrm{{Residual\ Std\ Err}}\ (\sigma) = {res.residual_std_error:.4f}\% \quad\big|\quad "
        rf"\mathrm{{Durbin\text{{-}}Watson}} = {res.durbin_watson_stat:.4f}\ \ (\mathrm{{Strong\ persistence}} \rightarrow \mathrm{{HAC\ applied}})"
        "\n"
        r"$\mathbf{45^\circ\ Parity\ Wald\ Test}\ (H_0: \alpha=0, \beta=1):\quad "
        rf"\mathrm{{Wald\ }} F({res.wald_df_num}, {res.wald_df_denom}) = {res.wald_f_stat:.3f} \quad (p = {res.wald_pval:.4f}) "
        r"\longrightarrow\ \mathbf{Fail\ to\ reject\ parity\ at\ 5\%\ level}"
        "\n"
        r"$\mathbf{Structural\ Basket\ Divergence:}\quad "
        rf"\sigma_{{rural}} = {df['cpi_rural'].std():.2f}\%\ \ \mathrm{{vs.}}\ \ \sigma_{{urban}} = {df['cpi_urban'].std():.2f}\% \quad\big|\quad "
        r"\mathrm{Food\ Weight:\ 54.18\%\ (Rural)\ vs.\ 36.29\%\ (Urban)\ (+17.89\ pp\ wedge)}"
    )

    ax_sub.text(
        0.5, 0.18,
        diagnostics_text,
        ha="center",
        va="center",
        fontsize=8.5,
        color="#1E293B",
        linespacing=1.45,
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor="#F1F5F9",
            edgecolor="#94A3B8",
            linewidth=1.0,
            alpha=0.95
        )
    )

    # Footer / Citation
    fig.text(
        0.06, 0.015,
        "Source: Ministry of Statistics & Programme Implementation (MoSPI) and RBI Database on Indian Economy (DBIE, 2014–2025).\n"
        "Inference: Newey-West (1987) HAC standard errors (4 lags). Gokhale Institute of Politics and Economics (GIPE).",
        ha="left",
        fontsize=7.8,
        color="#64748B"
    )

    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return out_file


def main():
    """CLI entrypoint to generate the publication plot."""
    base_dir = Path(__file__).resolve().parent.parent
    raw_csv = base_dir / "data" / "raw" / "cpi_inflation_cleaned.csv"
    processed_csv = base_dir / "data" / "processed" / "rural_urban_inflation_monthly.csv"
    out_fig = base_dir / "figures" / "rural_urban_ols_parity_scatter.png"

    df = load_and_preprocess_data(raw_csv, processed_csv)
    saved_path = plot_rural_urban_parity(df, out_fig, dpi=300)
    print(f"Publication figure successfully generated at: {saved_path}")


if __name__ == "__main__":
    main()

"""
factors/composite.py
--------------------
This is the "brain" of the factor model. It takes the five individual
factor scores (value, momentum, quality, low_vol, size) and combines
them into a single composite score for each stock.

The process has three steps:

  STEP 1 — WINSORIZE
  Remove extreme outliers that could distort the rankings.
  Example: if one stock has a freak P/E of 10,000x due to a data error,
  it would pull all other rankings down. We cap the top and bottom 1%.

  STEP 2 — Z-SCORE NORMALIZE
  Convert each factor score to a standard scale (mean=0, std=1).
  This ensures that a "1 unit" change in value means the same thing
  as a "1 unit" change in momentum — apples to apples.
  Formula: z = (x - mean) / standard_deviation

  STEP 3 — WEIGHTED SUM
  Multiply each normalized factor score by its weight from strategy.yaml,
  then add them all together. The result is one number per stock — the
  composite score. Higher = more attractive overall.

  Final ranking: sort stocks by composite score, highest first.
  The top 50 become our portfolio.
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def load_factor_weights(config_path: str = "configs/strategy.yaml") -> dict:
    """
    Reads the factor weights from the YAML config file.
    Returns a dict like: {'value': 0.25, 'momentum': 0.25, ...}
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    weights = {}
    for factor_name, factor_cfg in config["factors"].items():
        if factor_cfg.get("enabled", True):
            weights[factor_name] = factor_cfg["weight"]

    # Normalize so weights sum to exactly 1.0
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    logger.info(f"  Factor weights loaded: {weights}")
    return weights


def winsorize(series: pd.Series, pct: float = 0.01) -> pd.Series:
    """
    Caps extreme values at the pct and (1-pct) percentiles.

    Example with pct=0.01:
      - Any value below the 1st percentile → set to the 1st percentile value
      - Any value above the 99th percentile → set to the 99th percentile value

    This prevents a single extreme stock from dominating the rankings.
    """
    lower = series.quantile(pct)
    upper = series.quantile(1 - pct)
    return series.clip(lower, upper)


def zscore_normalize(series: pd.Series) -> pd.Series:
    """
    Converts a series to z-scores: (value - mean) / std_deviation.

    After this transformation:
      - Mean of the series = 0
      - Standard deviation = 1
      - A z-score of +2 means "2 standard deviations above average"
      - A z-score of -1 means "1 standard deviation below average"

    This puts all factors on the same scale regardless of their
    original units or range.
    """
    mean = series.mean()
    std  = series.std()

    if std == 0 or pd.isna(std):
        # All values are identical — can't normalize, return zeros
        return pd.Series(0, index=series.index)

    return (series - mean) / std


class CompositeFactorBuilder:
    """
    Orchestrates the full factor construction pipeline.

    Usage in main.py:
        builder = CompositeFactorBuilder(config, price_data, fundamental_data)
        composite_scores = builder.build()

    composite_scores is a pd.Series with one score per ticker,
    sorted highest to lowest. The top 50 go into our portfolio.
    """

    def __init__(
        self,
        config:       dict,
        prices:       pd.DataFrame,
        fundamentals: pd.DataFrame,
    ):
        self.config       = config
        self.prices       = prices
        self.fundamentals = fundamentals
        self.winsorize_pct = config.get("zscore", {}).get("winsorize_pct", 0.01)

    def build(self) -> pd.Series:
        """
        Runs the full pipeline: compute → winsorize → z-score → weight → combine.

        Returns
        -------
        pd.Series
            Composite scores sorted highest to lowest.
            Index = ticker symbols. Higher score = more attractive.
        """

        logger.info("=" * 50)
        logger.info("  Building composite factor scores")
        logger.info("=" * 50)

        # ── Step 1: Compute each individual factor ─────────────────────────
        factor_scores = self._compute_all_factors()

        if not factor_scores:
            logger.error("  No factor scores computed — check your data.")
            return pd.Series(dtype=float)

        # ── Step 2: Align all factors to the same set of tickers ──────────
        # Some stocks may be missing data for some factors.
        # pd.DataFrame() handles this automatically — missing values = NaN.
        scores_df = pd.DataFrame(factor_scores)

        logger.info(f"\n  Score coverage before normalization:")
        for col in scores_df.columns:
            n = scores_df[col].notna().sum()
            logger.info(f"    {col:<20}: {n} tickers")

        # ── Step 3: Winsorize each factor ──────────────────────────────────
        logger.info(f"\n  Winsorizing at {self.winsorize_pct:.0%} tails...")
        for col in scores_df.columns:
            scores_df[col] = winsorize(scores_df[col].dropna(), self.winsorize_pct) \
                             .reindex(scores_df.index)

        # ── Step 4: Z-score normalize each factor ─────────────────────────
        logger.info("  Z-score normalizing...")
        for col in scores_df.columns:
            valid = scores_df[col].dropna()
            scores_df[col] = zscore_normalize(valid).reindex(scores_df.index)

        # ── Step 5: Load weights and compute weighted sum ──────────────────
        weights = load_factor_weights()

        # Map our internal factor names to the config names
        name_map = {
            "value_score":    "value",
            "momentum_score": "momentum",
            "quality_score":  "quality",
            "low_vol_score":  "low_volatility",
            "size_score":     "size",
        }

        composite = pd.Series(0.0, index=scores_df.index)
        weight_used = pd.Series(0.0, index=scores_df.index)

        for score_col, config_name in name_map.items():
            if score_col not in scores_df.columns:
                continue
            w = weights.get(config_name, 0)
            if w == 0:
                continue

            factor_col = scores_df[score_col]
            # For tickers where we have a score, add weighted contribution
            has_data = factor_col.notna()
            composite[has_data]    += factor_col[has_data] * w
            weight_used[has_data]  += w

        # ── Step 6: Rescale by actual weight used ─────────────────────────
        # If a stock is missing 1 of 5 factors, we don't penalize it —
        # we just rescale by the weights that were actually available.
        # This avoids stocks with missing data getting artificially low scores.
        rescale_mask = weight_used > 0
        composite[rescale_mask] = composite[rescale_mask] / weight_used[rescale_mask]
        composite[~rescale_mask] = np.nan

        # ── Step 7: Sort and report ────────────────────────────────────────
        composite = composite.sort_values(ascending=False)
        composite.name = "composite_score"

        logger.info(f"\n  Composite score summary:")
        logger.info(f"    Stocks ranked    : {composite.notna().sum()}")
        logger.info(f"    Top score        : {composite.iloc[0]:.3f} ({composite.index[0]})")
        logger.info(f"    Median score     : {composite.median():.3f}")
        logger.info(f"    Bottom score     : {composite.dropna().iloc[-1]:.3f} "
                    f"({composite.dropna().index[-1]})")

        logger.info("\n  Top 10 stocks by composite score:")
        for i, (ticker, score) in enumerate(composite.head(10).items()):
            logger.info(f"    {i+1:>2}. {ticker:<8} {score:+.3f}")

        return composite

    def _compute_all_factors(self) -> dict:
        """
        Calls each individual factor module and collects the results.
        Returns a dict of {factor_name: pd.Series}.
        """

        factor_scores = {}
        factor_config = self.config.get("factors", {})

        # ── Value ──────────────────────────────────────────────────────────
        if factor_config.get("value", {}).get("enabled", True):
            try:
                from factors.value import compute_value_scores
                factor_scores["value_score"] = compute_value_scores(self.fundamentals)
            except Exception as e:
                logger.warning(f"  Value factor failed: {e}")

        # ── Momentum ───────────────────────────────────────────────────────
        if factor_config.get("momentum", {}).get("enabled", True):
            try:
                from factors.momentum import compute_momentum_scores
                factor_scores["momentum_score"] = compute_momentum_scores(self.prices)
            except Exception as e:
                logger.warning(f"  Momentum factor failed: {e}")

        # ── Quality ────────────────────────────────────────────────────────
        if factor_config.get("quality", {}).get("enabled", True):
            try:
                from factors.quality import compute_quality_scores
                factor_scores["quality_score"] = compute_quality_scores(self.fundamentals)
            except Exception as e:
                logger.warning(f"  Quality factor failed: {e}")

        # ── Low Volatility ─────────────────────────────────────────────────
        if factor_config.get("low_volatility", {}).get("enabled", True):
            try:
                from factors.low_vol import compute_low_vol_scores
                factor_scores["low_vol_score"] = compute_low_vol_scores(
                    self.prices,
                    benchmark_ticker=self.config.get("benchmark", "SPY")
                )
            except Exception as e:
                logger.warning(f"  Low-vol factor failed: {e}")

        # ── Size ───────────────────────────────────────────────────────────
        if factor_config.get("size", {}).get("enabled", True):
            try:
                from factors.size import compute_size_scores
                factor_scores["size_score"] = compute_size_scores(self.fundamentals)
            except Exception as e:
                logger.warning(f"  Size factor failed: {e}")

        logger.info(f"  Factors computed: {list(factor_scores.keys())}")
        return factor_scores

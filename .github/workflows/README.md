# .github/workflows/README.md
# ─────────────────────────────────────────────────────────────
# CI/CD Workflows — Portfolio Project
# ─────────────────────────────────────────────────────────────
#
# This folder is reserved for GitHub Actions workflows.
#
# Current status: manual execution only.
# The backtest requires Yahoo Finance data downloads (~40 min first run)
# which makes automated CI impractical without a data caching layer.
#
# Planned additions for production:
#
#   ci.yml          — Run test_pipeline.py smoke test on every push
#   lint.yml        — flake8 / black formatting checks
#   scheduled.yml   — Monthly scheduled backtest refresh
#
# To run the smoke test locally:
#   python tests/test_pipeline.py
#
# To run the full backtest:
#   python main.py

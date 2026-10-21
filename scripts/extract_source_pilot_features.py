"""Reuse the fixed strategy extractor with a separate public-source cache."""

from pathlib import Path

from scripts import extract_strategy_features

if __name__ == "__main__":
    extract_strategy_features.DATA = (
        Path(__file__).resolve().parents[1] / "data/generalization_pilot_20261020/xrf55"
    )
    extract_strategy_features.main()

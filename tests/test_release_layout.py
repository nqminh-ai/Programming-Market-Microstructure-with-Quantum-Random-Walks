"""Release-layout, dependency-pin, and canonical data-path tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.data.paths import (
    DATA_ROOT,
    REPORT_KINDS,
    asset_data_dir,
    asset_report_dir,
)
from scripts.pipelines.phase2_pipeline import report_kind_dir, resolve_config_path
from scripts.research.cross_asset_benchmark import build_cross_asset_scorecard


ROOT = Path(__file__).resolve().parents[1]


def test_dependencies_are_exactly_pinned_and_locked() -> None:
    requirements = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    locked = [
        line.strip()
        for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert set(requirements) <= set(locked)
    assert len(locked) > len(requirements)
    assert all(line.count("==") == 1 for line in requirements)
    assert all(line.count("==") == 1 for line in locked)
    assert not any(operator in line for line in requirements for operator in (">=", "<=", "~=", "!="))


def test_every_asset_config_uses_canonical_data_layout() -> None:
    expected = {
        "data_config.yaml": "btcusdt",
        "data_config_eth.yaml": "ethusdt",
        "data_config_bnb.yaml": "bnbusdt",
    }
    for filename, key in expected.items():
        config = yaml.safe_load((ROOT / "config" / filename).read_text(encoding="utf-8"))
        assert config["paths"] == {
            "raw": f"data/assets/{key}/raw",
            "processed": f"data/assets/{key}/processed",
            "features": f"data/assets/{key}/features",
            "reports": f"reports/assets/{key}",
        }


def test_phase2_pipeline_writes_reports_to_the_canonical_kind_directories() -> None:
    """The pipeline must agree with ``asset_report_dir`` for every report kind.

    phase2_pipeline previously wrote (and globbed) these files flat in the asset
    report root while ``src.data.paths`` declared the per-kind subdirectory
    canonical, so BTCUSDT ended up with two divergent copies of every report.
    """
    expected = {
        "data_config.yaml": "BTCUSDT",
        "data_config_eth.yaml": "ETHUSDT",
        "data_config_bnb.yaml": "BNBUSDT",
    }
    for filename, symbol in expected.items():
        config = yaml.safe_load((ROOT / "config" / filename).read_text(encoding="utf-8"))
        reports_dir = resolve_config_path(config, "reports")
        for kind in REPORT_KINDS:
            assert report_kind_dir(reports_dir, kind) == asset_report_dir(symbol, kind)


def test_no_report_kind_files_remain_flat_in_an_asset_report_root() -> None:
    """No ``<kind>_<SYMBOL>_<date>`` file may sit directly in the asset root."""
    reports_root = ROOT / "reports" / "assets"
    stray = [
        path
        for path in reports_root.glob("*/*")
        if path.is_file()
        and any(path.name.startswith(f"{kind}_") for kind in REPORT_KINDS)
    ]
    assert stray == [], f"flat report files must live under <asset>/<kind>/: {stray}"


def test_path_helpers_match_release_layout() -> None:
    assert DATA_ROOT == ROOT / "data" / "assets"
    assert asset_data_dir("BTC/USDT", "features") == (
        ROOT / "data" / "assets" / "btcusdt" / "features"
    )
    assert asset_report_dir("ETHUSDT", "data_quality") == (
        ROOT / "reports" / "assets" / "ethusdt" / "data_quality"
    )


def test_source_and_documentation_have_no_stale_flat_feature_path() -> None:
    stale = "data" + "/features"
    candidates = [ROOT / "README.md", ROOT / "data" / "README.md"]
    for directory in ("config", "docs", "scripts", "src", "tests"):
        candidates.extend(
            path
            for path in (ROOT / directory).rglob("*")
            if path.suffix.lower() in {".md", ".py", ".yaml", ".yml"}
        )

    offenders = []
    for path in candidates:
        content = path.read_text(encoding="utf-8").replace("\\", "/")
        if stale in content:
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_confirmatory_preregistration_freezes_days_models_and_inference() -> None:
    text = (ROOT / "docs" / "data_collection_todo.md").read_text(
        encoding="utf-8"
    )
    required = {
        "10 ngày sớm nhất train",
        "5 ngày tiếp theo",
        "5 ngày cuối untouched test",
        "Logistic L2 với đúng năm feature",
        "OrderFlow AR(5)",
        "Marked Hawkes conditional-mark logit",
        "seed `2026, 2027, 2028`",
        "block ngày\n`1, 2, 5`",
        "Benjamini–Hochberg",
        "Không thêm\nmodel, feature, grid hoặc endpoint sau khi xem test",
    }

    assert not sorted(marker for marker in required if marker not in text)


def test_paper_explicitly_retracts_unsupported_empirical_claims() -> None:
    paper = (ROOT / "docs" / "paper_draft.md").read_text(encoding="utf-8")
    normalized = " ".join(paper.split())

    assert "không phải heavy-tailed unitary shift" in normalized
    assert (
        "Không có bằng chứng hợp lệ cho khoảng tail index 1,1–2,5"
        in normalized
    )
    assert "Không được mô tả kết quả đó là kiểm định 31 ngày" in normalized
    assert "chưa hỗ trợ tuyên bố QRW vượt trội" in normalized


def test_cross_asset_scorecard_preserves_all_evaluated_days(tmp_path) -> None:
    summaries = [
        {
            "asset": "BTC/USDT",
            "status": "ok",
            "scorecard": [
                {
                    "model": "QRW Adaptive",
                    "mean_marginal_crps": 0.03,
                    "overall_rank": 2.0,
                    "evaluated_days": 20,
                }
            ],
        },
        {
            "asset": "ETH/USDT",
            "status": "ok",
            "scorecard": [
                {
                    "model": "QRW Adaptive",
                    "mean_marginal_crps": 0.04,
                    "overall_rank": 3.0,
                    "evaluated_days": 21,
                }
            ],
        },
    ]

    scorecard = build_cross_asset_scorecard(
        summaries, tmp_path / "scorecard.csv"
    )

    assert scorecard.loc[0, "n_assets"] == 2
    assert scorecard.loc[0, "total_evaluated_days"] == 41
    assert scorecard.loc[0, "avg_mean_marginal_crps"] == 0.035

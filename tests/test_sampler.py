"""Unit tests for ``src.segmentation.sampler`` (Étape 2.2)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.segmentation.sampler import (
    DEFAULT_QUOTAS,
    _bucket_year,
    build_folio_index,
    stratified_sample,
)


def test_bucket_year_assigns_correct_periods() -> None:
    """Each canonical year falls into the expected stratum."""
    assert _bucket_year(1326) == "XIV"
    assert _bucket_year(1399) == "XIV"
    assert _bucket_year(1400) == "XV_early"
    assert _bucket_year(1449) == "XV_early"
    assert _bucket_year(1450) == "XV_late"
    assert _bucket_year(1499) == "XV_late"
    assert _bucket_year(1500) == "XVI"


def test_build_folio_index_minimal(tmp_path: Path) -> None:
    """A header file with two distinct folios yields a two-entry index."""
    headers = tmp_path / "headers.txt"
    headers.write_text(textwrap.dedent("""\
        <doc folio="FRAN_0393_00016_L" volume="LL105" date="1327,1327::juin" >
        <doc folio="FRAN_0393_12418_L" volume="LL119" date="1458,1458::août" >
    """), encoding="utf-8")
    index = build_folio_index(headers)
    assert index == {
        "FRAN_0393_00016_L": ("LL105", 1327),
        "FRAN_0393_12418_L": ("LL119", 1458),
    }


def _fake_index(per_stratum: int = 20) -> dict[str, tuple[str, int]]:
    """Build a synthetic folio index with enough pages in each stratum."""
    index: dict[str, tuple[str, int]] = {}
    base_year_by_stratum = {"XIV": 1350, "XV_early": 1420, "XV_late": 1470, "XVI": 1502}
    for stratum, base_year in base_year_by_stratum.items():
        for i in range(per_stratum):
            register = f"LL{stratum}_{i % 3:02d}"
            folio = f"FRAN_{stratum}_{i:03d}_L"
            index[folio] = (register, base_year + (i % 5))
    return index


def test_stratified_sample_respects_default_quotas() -> None:
    """Default quotas (5/8/8/4) are met exactly."""
    index = _fake_index(per_stratum=20)
    available = set(index)
    sample = stratified_sample(index, available, seed=42)
    counts = {stratum: 0 for stratum in DEFAULT_QUOTAS}
    for item in sample:
        counts[item["stratum"]] += 1
    assert counts == DEFAULT_QUOTAS
    assert len(sample) == sum(DEFAULT_QUOTAS.values())


def test_stratified_sample_is_deterministic() -> None:
    """Same seed produces the same sample, byte for byte."""
    index = _fake_index(per_stratum=20)
    available = set(index)
    s1 = stratified_sample(index, available, seed=42)
    s2 = stratified_sample(index, available, seed=42)
    assert s1 == s2


def test_stratified_sample_excludes_unavailable_folios() -> None:
    """Folios missing on disk are never picked."""
    index = _fake_index(per_stratum=20)
    available = set(index)
    # Make folios with index 0 unavailable across all strata.
    excluded = {f for f in index if f.endswith("_000_L")}
    available -= excluded
    sample = stratified_sample(index, available, seed=42)
    assert not (excluded & {item["folio_id"] for item in sample})


def test_stratified_sample_raises_when_quota_unmet() -> None:
    """An impossibly large quota raises a clear error."""
    index = _fake_index(per_stratum=3)
    available = set(index)
    with pytest.raises(ValueError, match="Stratum XIV"):
        stratified_sample(index, available, quotas={"XIV": 99, "XV_early": 0, "XV_late": 0, "XVI": 0})

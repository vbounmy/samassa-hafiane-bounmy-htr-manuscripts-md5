"""Build the stratified evaluation sample for the page segmentation study.

The project segments the structure of pages with SAM and compares the
result to the ground-truth zones already annotated in e-NDP. To keep
computation tractable on the project's free Colab GPU budget, this
comparison runs on a stratified sample of ~25 pages rather than the
458 pages of the full corpus.

The stratification follows our diachronic axis: pages are sampled
roughly proportionally to the four sub-periods of the corpus (early 14th
to early 16th century). Sampling is deterministic (seeded) so the same
seed always yields the same sample — a prerequisite for reproducibility.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

_FOLIO_RE = re.compile(r'folio="([^"]+)"')
_VOLUME_RE = re.compile(r'volume="([^"]+)"')
_YEAR_RE = re.compile(r'date="(\d{4})')


def build_folio_index(headers_path: Path) -> dict[str, tuple[str, int]]:
    """Parse vertical-source ``<doc>`` headers into folio→(register, year) map.

    Args:
        headers_path: Plain-text file with one ``<doc ...>`` header per line.

    Returns:
        Mapping ``folio_id -> (register_id, year)``.
    """
    index: dict[str, tuple[str, int]] = {}
    with headers_path.open(encoding="utf-8") as stream:
        for line in stream:
            fm = _FOLIO_RE.search(line)
            vm = _VOLUME_RE.search(line)
            ym = _YEAR_RE.search(line)
            if fm and vm and ym:
                index[fm.group(1)] = (vm.group(1), int(ym.group(1)))
    return index


def _bucket_year(year: int) -> str:
    """Assign a year to one of four diachronic sub-periods.

    Args:
        year: Calendar year between 1300 and 1600.

    Returns:
        One of ``"XIV"``, ``"XV_early"``, ``"XV_late"``, ``"XVI"``.
    """
    if year < 1400:
        return "XIV"
    if year < 1450:
        return "XV_early"
    if year < 1500:
        return "XV_late"
    return "XVI"


# Target number of pages per stratum. Sums to 24.
# The XVI quota is capped at 3 because only three pages of the corpus map
# to that stratum (the 16th century is heavily under-represented in e-NDP).
DEFAULT_QUOTAS: dict[str, int] = {
    "XIV":      5,
    "XV_early": 8,
    "XV_late":  8,
    "XVI":      3,
}


def stratified_sample(
    folio_index: dict[str, tuple[str, int]],
    available_xml_stems: set[str],
    quotas: dict[str, int] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Pick a deterministic stratified sample of pages for segmentation.

    Pages are first grouped by sub-period via :func:`_bucket_year`. Within
    each period, the function draws ``quotas[period]`` pages uniformly
    at random, prioritising different registers so the sample stays
    diverse even within a period.

    Args:
        folio_index: Output of :func:`build_folio_index`.
        available_xml_stems: Set of XML stems (e.g. ``"FRAN_0393_00016_L"``)
            actually present on disk, so we never pick a missing page.
        quotas: Optional override of the default per-stratum counts.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts ``{"folio_id", "register", "year", "stratum"}``,
        ordered by ``(stratum, register, folio_id)`` for stable output.

    Raises:
        ValueError: If a stratum has fewer eligible pages than its quota.
    """
    if quotas is None:
        quotas = DEFAULT_QUOTAS

    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for folio_id, (register, year) in folio_index.items():
        if folio_id not in available_xml_stems:
            continue
        by_stratum[_bucket_year(year)].append({
            "folio_id": folio_id,
            "register": register,
            "year": year,
        })

    rng = random.Random(seed)
    sample: list[dict] = []
    for stratum, target in quotas.items():
        candidates = by_stratum.get(stratum, [])
        if len(candidates) < target:
            raise ValueError(
                f"Stratum {stratum} has only {len(candidates)} pages, "
                f"cannot satisfy quota of {target}"
            )

        # Group candidates by register, then pick round-robin across registers
        # to maximise diversity within the stratum.
        by_register: dict[str, list[dict]] = defaultdict(list)
        for item in candidates:
            by_register[item["register"]].append(item)
        for reg_items in by_register.values():
            rng.shuffle(reg_items)
        registers = list(by_register)
        rng.shuffle(registers)

        picked = 0
        register_cursor = 0
        while picked < target:
            register = registers[register_cursor % len(registers)]
            if by_register[register]:
                item = by_register[register].pop()
                item = {**item, "stratum": stratum}
                sample.append(item)
                picked += 1
            register_cursor += 1

    sample.sort(key=lambda d: (d["stratum"], d["register"], d["folio_id"]))
    return sample

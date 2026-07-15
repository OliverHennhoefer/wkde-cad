from __future__ import annotations

import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SHA256 = {
    Path(
        "outputs/simulations/figure1_unweighted_power_atlas/"
        "figure1_unweighted_power_atlas.png"
    ): "61cd39da00961c75c48ba9030d4e294065cf525c3ce3e98850301cd8c9f8b071",
    Path(
        "outputs/simulations/figure2_weighted_power_atlas/"
        "figure2_weighted_power_atlas.png"
    ): "c276c9f47ac170facd29b08e522af24cef274a50066d6a838966a59f22059bc7",
}


def test_promoted_atlas_images_are_byte_identical() -> None:
    for relative_path, expected_hash in EXPECTED_SHA256.items():
        payload = (REPO_ROOT / relative_path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_hash

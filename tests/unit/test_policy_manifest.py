import hashlib
import json
from pathlib import Path

import yaml
import pytest

from jeonse_support.policy import validate_policy_artifact


POLICY_PATH = Path("policies/risk-policy-v1.yaml")
SCOPED_SIGNAL_IDS = (
    "R-TX-COVERAGE",
    "R-DEPOSIT-RATIO",
    "R-HUG-REGION",
    "R-SOURCE-CONFLICT",
)


def test_policy_parameter_checksum_matches_canonical_scoped_mapping() -> None:
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    checksum = policy["provenance"]["parameter_checksum"]
    scoped_mapping = {
        "signals": {
            signal_id: policy["signals"][signal_id]
            for signal_id in SCOPED_SIGNAL_IDS
        },
        "overall_grade": policy["overall_grade"],
    }
    canonical_input = json.dumps(
        scoped_mapping,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    assert checksum["scope"] == [
        f"signals.{signal_id}" for signal_id in SCOPED_SIGNAL_IDS
    ] + ["overall_grade"]
    assert checksum["canonical_input"] == canonical_input
    assert checksum["value"] == hashlib.sha256(
        canonical_input.encode("utf-8")
    ).hexdigest()


def test_policy_runtime_gate_rejects_tampered_mapping(tmp_path: Path) -> None:
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    policy["signals"]["R-HUG-REGION"]["population"]["minimum_rows"] = 19
    tampered = tmp_path / "risk-policy-v1.yaml"
    tampered.write_text(
        yaml.safe_dump(policy, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="INVALID_POLICY_ARTIFACT"):
        validate_policy_artifact(tampered)

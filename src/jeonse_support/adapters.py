"""Read-only snapshot adapters; they never crawl or contact live services."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import ContractModel, HugIncidentStatistic, Identifier, ListingConditions, NonEmptyText, SampleListing
from .repositories import ComparableTransactionQuery, HugIncidentQuery


class SnapshotArtifactError(RuntimeError):
    """A required local snapshot is missing or cannot be trusted as an artifact."""


class SnapshotArtifactValidationError(SnapshotArtifactError):
    """A local snapshot does not conform to the expected, explicit schema."""


class _SnapshotEnvelope(ContractModel):
    dataset_kind: str
    snapshot_notice: NonEmptyText
    snapshot_as_of: date
    provenance_id: Identifier
    content_sha256: str
    dataset_id: Identifier | None = None
    dataset_version: Identifier | None = None


class _SnapshotRowMetadata(ContractModel):
    source_name: NonEmptyText
    snapshot_as_of: date
    provenance_id: Identifier


_ROW_METADATA_FIELDS = frozenset(_SnapshotRowMetadata.model_fields)


def _read_snapshot(path: Path, collection: str) -> tuple[dict[str, Any], ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SnapshotArtifactError(f"Snapshot artifact is missing: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise SnapshotArtifactValidationError(f"Snapshot artifact is invalid: {path}") from error
    if not isinstance(payload, dict):
        raise SnapshotArtifactValidationError(f"Snapshot artifact must be an object: {path}")
    allowed_fields = set(_SnapshotEnvelope.model_fields) | {collection}
    unknown_fields = set(payload) - allowed_fields
    if unknown_fields:
        raise SnapshotArtifactValidationError(f"Snapshot artifact has unknown fields: {sorted(unknown_fields)}")
    try:
        envelope = _SnapshotEnvelope.model_validate_json(
            json.dumps({key: payload.get(key) for key in _SnapshotEnvelope.model_fields})
        )
    except ValidationError as error:
        field = ".".join(str(part) for part in error.errors()[0]["loc"])
        raise SnapshotArtifactValidationError(
            f"Snapshot artifact has invalid metadata field {field}: {path}"
        ) from error
    if envelope.dataset_kind != "snapshot":
        raise SnapshotArtifactValidationError(f"Snapshot artifact must declare dataset_kind='snapshot': {path}")
    canonical_payload = {key: value for key, value in payload.items() if key != "content_sha256"}
    canonical = json.dumps(canonical_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != envelope.content_sha256:
        raise SnapshotArtifactValidationError(f"Snapshot artifact content_sha256 does not match canonical content: {path}")
    rows = payload.get(collection)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise SnapshotArtifactValidationError(f"Snapshot artifact has no valid {collection} rows: {path}")
    for row in rows:
        if row.get("snapshot_as_of") != envelope.snapshot_as_of.isoformat():
            raise SnapshotArtifactValidationError(
                f"Snapshot row snapshot_as_of conflicts with envelope: {path}"
            )
        provenance_id = row.get("provenance_id")
        expected_prefix = f"{envelope.provenance_id}-"
        if not isinstance(provenance_id, str) or (
            provenance_id != envelope.provenance_id
            and not provenance_id.startswith(expected_prefix)
        ):
            raise SnapshotArtifactValidationError(
                f"Snapshot row provenance conflicts with envelope: {path}"
            )
        row_identity = (row.get("dataset_id"), row.get("dataset_version"))
        envelope_identity = (envelope.dataset_id, envelope.dataset_version)
        if any(value is not None for value in row_identity + envelope_identity):
            if (
                not all(value is not None for value in row_identity)
                or not all(value is not None for value in envelope_identity)
                or row_identity != envelope_identity
            ):
                raise SnapshotArtifactValidationError(
                    f"Snapshot row dataset identity conflicts with envelope: {path}"
                )
    return tuple(rows)


def _model_payload(
    row: dict[str, Any], model: type[ContractModel], *, allowed_dimensions: frozenset[str] = frozenset()
) -> dict[str, Any]:
    allowed_fields = set(model.model_fields) | _ROW_METADATA_FIELDS | set(allowed_dimensions)
    unknown_fields = set(row) - allowed_fields
    if unknown_fields:
        raise SnapshotArtifactValidationError(f"Snapshot row has unknown fields: {sorted(unknown_fields)}")
    try:
        _SnapshotRowMetadata.model_validate_json(
            json.dumps({key: row.get(key) for key in _ROW_METADATA_FIELDS})
        )
    except ValidationError as error:
        field = ".".join(str(part) for part in error.errors()[0]["loc"])
        raise SnapshotArtifactValidationError(
            f"Snapshot row has invalid provenance field {field}"
        ) from error
    return {key: value for key, value in row.items() if key in model.model_fields}






@dataclass(frozen=True, slots=True)
class JsonListingCatalog:
    """A strict, local JSON implementation of :class:`ListingCatalog`."""

    path: Path

    async def get_listing(self, listing_id: str) -> ListingConditions | None:
        matches: list[ListingConditions] = []
        listing_ids: set[str] = set()
        for row in _read_snapshot(self.path, "listings"):
            try:
                listing = ListingConditions.model_validate_json(
                    json.dumps(_model_payload(row, ListingConditions))
                )
            except ValidationError as error:
                raise SnapshotArtifactValidationError(
                    f"Invalid listing row {row.get('listing_id')!r} in {self.path}"
                ) from error
            if listing.listing_id in listing_ids:
                raise SnapshotArtifactValidationError(
                    f"Duplicate listing_id {listing.listing_id!r} in {self.path}"
                )
            listing_ids.add(listing.listing_id)
            if listing.listing_id == listing_id:
                matches.append(listing)
        return matches[0] if matches else None


@dataclass(frozen=True, slots=True)
class JsonTransactionRepository:
    """Local transaction source; deterministic policy performs final eligibility checks."""

    path: Path

    async def list_comparables(self, query: ComparableTransactionQuery) -> tuple[SampleListing, ...]:
        samples: list[SampleListing] = []
        transaction_ids: set[str] = set()
        qualified_identity_mode: bool | None = None
        stable_row_ids: set[tuple[str, str, str]] = set()
        for row in _read_snapshot(self.path, "transactions"):
            try:
                sample = SampleListing.model_validate_json(
                    json.dumps(_model_payload(row, SampleListing))
                )
            except ValidationError as error:
                raise SnapshotArtifactValidationError(f"Invalid transaction row {row.get('transaction_id')!r} in {self.path}") from error
            if sample.transaction_id in transaction_ids:
                raise SnapshotArtifactValidationError(f"Duplicate transaction_id {sample.transaction_id!r} in {self.path}")
            transaction_ids.add(sample.transaction_id)
            qualified_identity = (
                sample.dataset_id,
                sample.dataset_version,
                sample.stable_row_id,
            )
            has_qualified_identity = all(value is not None for value in qualified_identity)
            if qualified_identity_mode is None:
                qualified_identity_mode = has_qualified_identity
            elif qualified_identity_mode != has_qualified_identity:
                raise SnapshotArtifactValidationError(
                    f"Transaction snapshot mixes qualified and legacy row identities: {self.path}"
                )
            if has_qualified_identity:
                stable_key = (
                    sample.dataset_id,
                    sample.dataset_version,
                    sample.stable_row_id,
                )
                if stable_key in stable_row_ids:
                    raise SnapshotArtifactValidationError(
                        f"Duplicate qualified transaction row identity {stable_key!r} in {self.path}"
                    )
                stable_row_ids.add(stable_key)
            samples.append(sample)
        return tuple(sorted(samples, key=lambda item: (item.occurred_on, item.transaction_id)))


@dataclass(frozen=True, slots=True)
class JsonHugIncidentRepository:
    """Local fixed-period regional HUG snapshot repository; no service requests occur."""

    path: Path

    def _rows(self) -> tuple[tuple[HugIncidentStatistic, bool], ...]:
        parsed: list[tuple[HugIncidentStatistic, bool]] = []
        statistic_ids: set[str] = set()
        population_keys: set[tuple[bool, date, date, str, str]] = set()
        for row in _read_snapshot(self.path, "statistics"):
            is_subject = row.get("is_subject", False)
            if not isinstance(is_subject, bool):
                raise SnapshotArtifactValidationError(f"Invalid is_subject flag in {self.path}")
            try:
                statistic = HugIncidentStatistic.model_validate_json(
                    json.dumps(_model_payload(row, HugIncidentStatistic, allowed_dimensions=frozenset({"is_subject"})))
                )
            except ValidationError as error:
                raise SnapshotArtifactValidationError(f"Invalid HUG statistic {row.get('statistic_id')!r} in {self.path}") from error
            population_key = (is_subject, statistic.period_start, statistic.period_end, statistic.granularity, statistic.geography)
            if statistic.statistic_id in statistic_ids or population_key in population_keys:
                raise SnapshotArtifactValidationError(f"Duplicate HUG statistic key in {self.path}")
            statistic_ids.add(statistic.statistic_id)
            population_keys.add(population_key)
            parsed.append((statistic, is_subject))
        return tuple(parsed)

    async def get_subject_statistic(self, query: HugIncidentQuery) -> HugIncidentStatistic | None:
        matches = tuple(
            statistic
            for statistic, is_subject in self._rows()
            if is_subject and statistic.period_start == query.period_start and statistic.period_end == query.period_end
        )
        if len(matches) > 1:
            raise SnapshotArtifactValidationError(f"Duplicate HUG subject period in {self.path}")
        return matches[0] if matches else None

    async def list_reference_statistics(self, query: HugIncidentQuery) -> tuple[HugIncidentStatistic, ...]:
        return tuple(
            statistic
            for statistic, is_subject in self._rows()
            if not is_subject
            and statistic.period_start == query.period_start
            and statistic.period_end == query.period_end
            and statistic.period_end <= query.as_of
        )

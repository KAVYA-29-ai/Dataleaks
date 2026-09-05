from __future__ import annotations

from collections.abc import Callable

from dataleaks.schemas.finding import Finding


CompatibilityFunction = Callable[
    [Finding, Finding],
    bool,
]


class FindingAggregator:
    """Aggregate detector findings that represent the same root issue."""

    MERGEABLE_PAIRS: dict[
        tuple[str, str],
        CompatibilityFunction,
    ] = {
        (
            "target_statistical",
            "feature_suspicious",
        ): lambda first, second: _same_target_feature(
            first,
            second,
        ),
        (
            "split_overlap",
            "feature_identifier",
        ): lambda first, second: _same_affected_column(
            first,
            second,
        ),
    }

    def aggregate(
        self,
        findings: list[Finding],
    ) -> list[Finding]:
        """Return findings with eligible pairwise duplicates merged."""

        if not isinstance(findings, list):
            raise TypeError("findings must be a list")

        if not findings:
            return []

        result: list[Finding] = []
        consumed: set[int] = set()

        for index, finding in enumerate(findings):
            if index in consumed:
                continue

            merged = False

            for other_index in range(
                index + 1,
                len(findings),
            ):
                if other_index in consumed:
                    continue

                other = findings[other_index]

                if self._can_merge(
                    finding,
                    other,
                ):
                    result.append(
                        self._merge(
                            finding,
                            other,
                        )
                    )

                    consumed.add(index)
                    consumed.add(other_index)
                    merged = True
                    break

            if not merged:
                result.append(finding)
                consumed.add(index)

        return result

    def _can_merge(
        self,
        first: Finding,
        second: Finding,
    ) -> bool:
        """Return whether two findings are eligible for merging."""

        pair = (
            first.detector,
            second.detector,
        )

        reverse_pair = (
            second.detector,
            first.detector,
        )

        compatibility = self.MERGEABLE_PAIRS.get(
            pair
        )

        if compatibility is None:
            compatibility = self.MERGEABLE_PAIRS.get(
                reverse_pair
            )

        if compatibility is None:
            return False

        return compatibility(
            first,
            second,
        )

    @staticmethod
    def _merge(
        first: Finding,
        second: Finding,
    ) -> Finding:
        """Merge two compatible findings into one finding."""

        related_detectors: list[str] = []

        for detector in (
            *first.related_detectors,
            first.detector,
            *second.related_detectors,
            second.detector,
        ):
            if detector not in related_detectors:
                related_detectors.append(
                    detector
                )

        affected_columns: list[str] = []

        for column in (
            *first.affected_columns,
            *second.affected_columns,
        ):
            if column not in affected_columns:
                affected_columns.append(
                    column
                )

        evidence = {
            "type": "aggregated_findings",
            "sources": [
                first.evidence,
                second.evidence,
            ],
        }

        return Finding(
            detector=first.detector,
            category=first.category,
            severity=_highest_severity(
                first.severity,
                second.severity,
            ),
            confidence=max(
                first.confidence,
                second.confidence,
            ),
            explanation=first.explanation,
            recommendation=first.recommendation,
            affected_columns=affected_columns,
            evidence=evidence,
            metadata={
                **first.metadata,
                **second.metadata,
            },
            related_detectors=related_detectors,
        )


def _highest_severity(
    first: str,
    second: str,
) -> str:
    """Return the higher severity of two findings."""

    levels = {
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }

    first_normalized = first.strip().lower()
    second_normalized = second.strip().lower()

    if levels[first_normalized] >= levels[
        second_normalized
    ]:
        return first_normalized

    return second_normalized


def _same_target_feature(
    first: Finding,
    second: Finding,
) -> bool:
    """Check whether two findings refer to the same feature and target."""

    first_target = first.evidence.get(
        "target"
    )
    second_target = second.evidence.get(
        "target"
    )

    if (
        first_target is None
        or second_target is None
    ):
        return False

    if first_target != second_target:
        return False

    first_features = set(
        first.affected_columns
    )
    second_features = set(
        second.affected_columns
    )

    if not first_features or not second_features:
        return False

    if not first_features.intersection(
        second_features
    ):
        return False

    first_correlation = first.evidence.get(
        "absolute_correlation"
    )
    second_correlation = second.evidence.get(
        "absolute_correlation"
    )

    if (
        first_correlation is not None
        and second_correlation is not None
    ):
        return (
            abs(
                float(first_correlation)
                - float(second_correlation)
            )
            <= 1e-6
        )

    return True


def _same_affected_column(
    first: Finding,
    second: Finding,
) -> bool:
    """Check whether two findings refer to the same column."""

    first_columns = set(
        first.affected_columns
    )
    second_columns = set(
        second.affected_columns
    )

    if not first_columns or not second_columns:
        return False

    return bool(
        first_columns.intersection(
            second_columns
        )
    )
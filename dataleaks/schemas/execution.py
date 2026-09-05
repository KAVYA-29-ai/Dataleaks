from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorExecution:
    """Execution metadata for a single DataLeaks detector."""

    detector: str
    status: str
    finding_count: int = 0
    error: str | None = None
    exception_type: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.detector.strip():
            raise ValueError(
                "detector must not be empty"
            )

        if self.status not in {
            "completed",
            "failed",
            "skipped",
        }:
            raise ValueError(
                "status must be 'completed', 'failed', or 'skipped'"
            )

        if self.finding_count < 0:
            raise ValueError(
                "finding_count must be non-negative"
            )

        if self.status == "completed":
            if self.error is not None:
                raise ValueError(
                    "completed execution cannot contain an error"
                )

            if self.exception_type is not None:
                raise ValueError(
                    "completed execution cannot contain "
                    "an exception_type"
                )

            if self.reason is not None:
                raise ValueError(
                    "completed execution cannot contain a reason"
                )

        elif self.status == "failed":
            if self.error is None:
                raise ValueError(
                    "failed execution must contain an error"
                )

            if self.exception_type is None:
                raise ValueError(
                    "failed execution must contain an exception_type"
                )

            if self.reason is not None:
                raise ValueError(
                    "failed execution cannot contain a reason"
                )

        elif self.status == "skipped":
            if self.error is not None:
                raise ValueError(
                    "skipped execution cannot contain an error"
                )

            if self.exception_type is not None:
                raise ValueError(
                    "skipped execution cannot contain "
                    "an exception_type"
                )

            if self.reason is None or not self.reason.strip():
                raise ValueError(
                    "status must be 'completed', 'failed', or 'skipped'; "
                    "skipped execution must contain a reason"
                )
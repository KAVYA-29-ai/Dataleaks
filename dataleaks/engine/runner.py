from __future__ import annotations

from dataleaks.engine.registry import DetectorRegistry
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.execution import DetectorExecution
from dataleaks.schemas.finding import Finding


class DetectorRunner:
    """Run registered DataLeaks detectors safely and deterministically."""

    def __init__(self, registry: DetectorRegistry) -> None:
        self.registry = registry
        self.warnings: list[dict[str, str]] = []
        self.execution: list[DetectorExecution] = []

    def run(self, context: DatasetContext) -> list[Finding]:
        """Run enabled detectors and record skipped/failed execution."""

        findings: list[Finding] = []

        self.warnings = []
        self.execution = []

        # Record intentionally skipped detectors first.
        for detector, reason in self.registry.skipped().items():
            self.execution.append(
                DetectorExecution(
                    detector=detector,
                    status="skipped",
                    finding_count=0,
                    reason=reason,
                )
            )

        for detector in self.registry.create_all():
            try:
                result = detector.detect(context)
            except Exception as exc:
                warning = {
                    "detector": detector.name,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                }

                self.warnings.append(warning)

                self.execution.append(
                    DetectorExecution(
                        detector=detector.name,
                        status="failed",
                        finding_count=0,
                        error=str(exc),
                        exception_type=type(exc).__name__,
                    )
                )

                continue

            if not isinstance(result, list):
                raise TypeError(
                    f"Detector '{detector.name}' must return list[Finding]"
                )

            findings.extend(result)

            self.execution.append(
                DetectorExecution(
                    detector=detector.name,
                    status="completed",
                    finding_count=len(result),
                )
            )

        return findings
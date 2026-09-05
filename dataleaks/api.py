from __future__ import annotations

from typing import Any

from dataleaks.engine.aggregator import FindingAggregator
from dataleaks.engine.defaults import build_default_registry
from dataleaks.engine.registry import DetectorRegistry
from dataleaks.engine.runner import DetectorRunner
from dataleaks.reporting.report import LeakageReport
from dataleaks.schemas.config import DataLeaksConfig
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class DataLeaks:
    """Public API for running DataLeaks analysis."""

    def __init__(
        self,
        data,
        *,
        target: str | None = None,
        train=None,
        validation=None,
        test=None,
        config: DataLeaksConfig | None = None,
        metadata: dict[str, Any] | None = None,
        registry: DetectorRegistry | None = None,
    ) -> None:
        self.context = DatasetContext(
            data=data,
            target=target,
            train=train if train is not None else data,
            validation=validation,
            test=test,
            metadata=metadata or {},
        )

        self.config = config or DataLeaksConfig()

        if registry is None:
            self.registry = build_default_registry(
                self.context,
                self.config,
            )
        else:
            self.registry = registry

    def run(self) -> LeakageReport:
        """Run detectors, aggregate findings, and build a report."""

        runner = DetectorRunner(self.registry)

        findings: list[Finding] = runner.run(
            self.context
        )

        findings = FindingAggregator().aggregate(
            findings
        )

        return LeakageReport.from_findings(
            findings,
            metadata=self.context.metadata,
            execution=runner.execution,
            schema_validation=self.context.schema_validation,
        )
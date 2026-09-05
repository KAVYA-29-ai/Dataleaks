import pandas as pd
import pytest

from dataleaks import DataLeaks
from dataleaks.engine.detector import BaseDetector
from dataleaks.engine.registry import DetectorRegistry
from dataleaks.schemas.config import DataLeaksConfig
from dataleaks.schemas.finding import Finding


class TestDetector(BaseDetector):
    """Valid detector used for API tests."""

    name = "test_detector"
    category = "test"

    def detect(self, context):
        return [
            Finding(
                detector=self.name,
                category=self.category,
                severity="high",
                confidence=0.9,
                explanation="Test leakage.",
                recommendation="Fix the leakage.",
            )
        ]


def make_registry():
    registry = DetectorRegistry()
    registry.register(TestDetector)
    return registry


def test_public_api_can_be_imported():
    from dataleaks import DataLeaks as ImportedDataLeaks

    assert ImportedDataLeaks is DataLeaks


def test_api_creates_dataset_context():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )

    analyzer = DataLeaks(
        data,
        target="target",
    )

    assert analyzer.context.data.equals(data)
    assert analyzer.context.target == "target"


def test_api_accepts_train_test_splits():
    data = pd.DataFrame(
        {"feature": [1, 2]}
    )

    train = pd.DataFrame(
        {"feature": [1]}
    )

    test = pd.DataFrame(
        {"feature": [2]}
    )

    analyzer = DataLeaks(
        data,
        train=train,
        test=test,
    )

    assert analyzer.context.train.equals(train)
    assert analyzer.context.test.equals(test)


def test_api_accepts_metadata():
    data = pd.DataFrame(
        {"feature": [1, 2]}
    )

    analyzer = DataLeaks(
        data,
        metadata={"source": "test"},
    )

    assert analyzer.context.metadata == {
        "source": "test"
    }


def test_api_accepts_config():
    data = pd.DataFrame(
        {"feature": [1, 2]}
    )

    config = DataLeaksConfig(
        enable_target_checks=False,
    )

    analyzer = DataLeaks(
        data,
        config=config,
    )

    assert analyzer.config is config


def test_api_uses_default_registry():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )

    analyzer = DataLeaks(
        data,
        target="target",
    )

    assert analyzer.registry.names()


def test_api_uses_custom_registry():
    data = pd.DataFrame(
        {"feature": [1, 2]}
    )

    registry = make_registry()

    analyzer = DataLeaks(
        data,
        registry=registry,
    )

    assert analyzer.registry is registry


def test_run_returns_leakage_report():
    data = pd.DataFrame(
        {"feature": [1, 2]}
    )

    analyzer = DataLeaks(
        data,
        registry=make_registry(),
    )

    report = analyzer.run()

    assert report.finding_count == 1
    assert report.has_leakage


def test_run_executes_registered_detector():
    data = pd.DataFrame(
        {"feature": [1, 2]}
    )

    report = DataLeaks(
        data,
        registry=make_registry(),
    ).run()

    finding = report.findings[0]

    assert finding.detector == "test_detector"
    assert finding.severity == "high"
    assert finding.confidence == 0.9


def test_run_builds_risk():
    data = pd.DataFrame(
        {"feature": [1, 2]}
    )

    report = DataLeaks(
        data,
        registry=make_registry(),
    ).run()

    assert report.risk_score == pytest.approx(0.675)
    assert report.risk_level == "medium"


def test_run_builds_recommendations():
    data = pd.DataFrame(
        {"feature": [1, 2]}
    )

    report = DataLeaks(
        data,
        registry=make_registry(),
    ).run()

    assert report.recommendations == [
        "Fix the leakage."
    ]


def test_run_preserves_metadata():
    data = pd.DataFrame(
        {"feature": [1, 2]}
    )

    report = DataLeaks(
        data,
        metadata={"source": "unit_test"},
        registry=make_registry(),
    ).run()

    assert report.metadata == {
        "source": "unit_test"
    }


def test_default_registry_runs_real_target_detector():
    data = pd.DataFrame(
        {
            "leaked_feature": [0, 1, 0, 1],
            "target": [0, 1, 0, 1],
        }
    )

    report = DataLeaks(
        data,
        target="target",
    ).run()

    assert report.has_leakage

    detectors = {
        finding.detector
        for finding in report.findings
    }

    assert "target_direct" in detectors


def test_default_registry_detects_identifier_feature():
    data = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
            ],
            "target": [0, 1, 0],
        }
    )

    report = DataLeaks(
        data,
        target="target",
    ).run()

    detectors = {
        finding.detector
        for finding in report.findings
    }

    assert "feature_identifier" in detectors


def test_default_registry_respects_target_config():
    data = pd.DataFrame(
        {
            "leaked_feature": [0, 1, 0, 1],
            "target": [0, 1, 0, 1],
        }
    )

    config = DataLeaksConfig(
        enable_target_checks=False,
    )

    report = DataLeaks(
        data,
        target="target",
        config=config,
    ).run()

    detectors = {
        finding.detector
        for finding in report.findings
    }

    assert "target_direct" not in detectors
    assert "target_statistical" not in detectors
    assert "target_derived" not in detectors


def test_default_registry_handles_train_test_data():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    train = pd.DataFrame(
        {
            "feature": [1, 2],
            "target": [0, 1],
        }
    )

    test = pd.DataFrame(
        {
            "feature": [1, 3],
            "target": [0, 0],
        }
    )

    report = DataLeaks(
        data,
        target="target",
        train=train,
        test=test,
    ).run()

    assert report.finding_count >= 0


def test_default_registry_handles_temporal_configuration():
    data = pd.DataFrame(
        {
            "prediction_time": [
                "2025-01-01",
                "2025-01-02",
            ],
            "feature_time": [
                "2025-01-02",
                "2025-01-01",
            ],
            "timestamp": [
                "2025-01-01",
                "2025-01-02",
            ],
            "target": [0, 1],
        }
    )

    report = DataLeaks(
        data,
        target="target",
        metadata={
            "temporal": {
                "prediction_time_column": "prediction_time",
                "feature_time_columns": ["feature_time"],
                "time_column": "timestamp",
            }
        },
    ).run()

    detectors = {
        finding.detector
        for finding in report.findings
    }

    assert "temporal_future_features" in detectors


def test_default_registry_handles_reference_dataset():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )

    reference = pd.DataFrame(
        {
            "feature": [1, 10],
            "target": [0, 1],
        }
    )

    report = DataLeaks(
        data,
        target="target",
        metadata={
            "reference_data": reference,
        },
    ).run()

    detectors = {
        finding.detector
        for finding in report.findings
    }

    assert "cross_dataset_overlap" in detectors


def test_invalid_target_is_rejected():
    data = pd.DataFrame(
        {"feature": [1, 2]}
    )

    with pytest.raises(ValueError):
        DataLeaks(
            data,
            target="missing",
        )


def test_invalid_dataframe_is_rejected():
    with pytest.raises(TypeError):
        DataLeaks(
            "not a dataframe",
        )
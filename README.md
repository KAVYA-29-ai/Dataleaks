# 🔍 DataLeaks

**Automatic data leakage detection for machine learning workflows.**

> Find leakage before it silently invalidates your model.

[![PyPI](https://img.shields.io/pypi/v/dataleaks)](https://pypi.org/project/dataleaks/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![Tests](https://img.shields.io/badge/tests-521%20passed-success)](https://github.com/KAVYA-29-ai/Dataleaks)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

DataLeaks analyzes datasets and ML workflow metadata for signals of target,
split, temporal, preprocessing, identifier, feature, and cross-dataset leakage.
It returns structured, explainable findings with severity, confidence, evidence,
risk scoring, and remediation recommendations.

---

## Why DataLeaks?

A model can achieve excellent validation performance while being fundamentally
unreliable because information from the target, future, evaluation data, or a
downstream workflow has leaked into the training process.

DataLeaks is designed to surface these signals **before they silently distort
model evaluation or production performance**.

## What it detects

| Area | Checks |
| --- | --- |
| 🎯 Target leakage | Direct, statistical, and derived-target relationships |
| 🔀 Split leakage | Exact duplicates, near-duplicates, overlapping values, entity overlap |
| 🆔 Identifier leakage | Identifier-like features and cross-split identifier overlap |
| 🕵️ Suspicious features | Strong feature-target relationships and semantic leakage signals |
| ⏱️ Temporal leakage | Invalid timestamps, ordering issues, and future feature timestamps |
| 🚧 Preprocessing leakage | Fit-before-split and held-out-data contamination |
| 🌐 Cross-dataset leakage | Shared entities and values across dataset boundaries |
| 🧪 Schema validation | Target presence, missing columns, and dtype mismatches |
| 📊 Risk scoring | Severity × confidence assessment |
| 💡 Recommendations | Actionable remediation guidance |
| 📋 Reporting | Console and JSON output |
| ⚙️ Execution tracking | Completed, failed, and skipped detectors |
| 🐍 Python API | Programmatic integration |
| 💻 CLI | One-command analysis |

---

## Installation

```bash
pip install dataleaks
```

**Python:** 3.10+

## Quick start

```python
import pandas as pd
from dataleaks import DataLeaks

df = pd.read_csv("train.csv")

report = DataLeaks(df, target="target").run()

print("Risk:", report.risk_level)
print("Score:", report.risk_score)
print("Findings:", report.finding_count)

for finding in report.findings:
    print(finding.detector, finding.severity)
    print(finding.explanation)
```

```text
Dataset
  ↓
DataLeaks(...)
  ↓
.run()
  ↓
LeakageReport
```

---

## CLI

```bash
# Basic analysis
dataleaks train.csv --target target

# Train / validation / test
dataleaks train.csv --target target --validation validation.csv --test test.csv

# JSON output
dataleaks train.csv --target target --output json
```

---

## Target leakage

Target leakage occurs when a feature directly or indirectly reveals the value
being predicted.

### Direct

```text
target = churn
leaky_feature = churn
```

### Statistical

Extremely strong numerical relationships between a feature and the target are
flagged for investigation.

### Derived target

```text
leaky_feature = 1 × target + 0
```

Deterministic relationships can make model performance appear unrealistically
strong.

---

## Split leakage

DataLeaks analyzes relationships between training, validation, and test data.
It checks for duplicate, near-duplicate, overlapping, and entity-level patterns.

```text
TRAIN                    TEST

customer_id              customer_id
CUST1001   ────────────► CUST1001
CUST1002   ────────────► CUST1002
CUST1003                  CUST1007
```

When the same real-world entities appear across training and evaluation data,
model evaluation can become unreliable.

---

## Identifier leakage

Identifiers can become leakage channels when entities overlap across dataset
boundaries.

```text
customer_id
user_id
patient_id
transaction_id
device_id
```

High cardinality alone is **not** treated as proof of identifier leakage.
Identifier-like semantics and explicit metadata keep this detection conservative.

---

## Suspicious and post-outcome features

Some features contain information that only becomes available after the event
being predicted.

```text
future_purchase_value
post_outcome_refund
after_event_status
forecast_revenue
subsequent_transaction
```

DataLeaks combines statistical evidence with semantic signals to identify
features that deserve investigation.

```text
Prediction
    │
    ├──► Model Input
    │
    └──► Refund Processed
             ↑
        happens later
```

Using a later event as an input for an earlier prediction introduces leakage.

---

## Temporal leakage

Temporal leakage occurs when information that would not have been available at
prediction time becomes part of the model input.

DataLeaks can detect:

- Invalid or unparsable timestamps
- Chronological ordering problems
- Feature timestamps after prediction timestamps
- Future-feature relationships
- Conditional timestamp presence

### Chronological validation

```bash
dataleaks train.csv --target target --time-column event_time
```

### Future-feature detection

```bash
dataleaks train.csv --target target \
  --prediction-time-column prediction_time \
  --feature-time-columns signup_time outcome_time
```

### Conditional timestamps

```bash
dataleaks train.csv --target target \
  --conditional-time-column outcome_time \
  --conditional-target-column target \
  --conditional-present-when 1
```

Temporal metadata is supplied explicitly rather than guessed automatically.

---

## Preprocessing leakage

Leakage can happen before model training even begins.

### Incorrect

```text
Full Dataset
     │
     ▼
Preprocessing.fit()
     │
     ├──► Train
     └──► Test
```

### Correct

```text
Full Dataset
     │
     ▼
Train / Test Split
     │
     ├──► Train → fit()
     │
     └──► Test  → transform()
```

DataLeaks can analyze preprocessing workflow metadata:

```python
metadata = {
    "preprocessing": {
        "fitted_on": "full_dataset",
        "contaminated_by": ["test"],
    }
}

report = DataLeaks(
    df,
    target="target",
    metadata=metadata,
).run()
```

This can identify preprocessing fitted before the split and contamination by
held-out data.

---

## Cross-dataset leakage

DataLeaks can analyze overlap between datasets when reference data is supplied
through the supported dataset context.

This is useful for detecting shared entities or values that should remain
isolated between datasets.

---

## Risk scoring

Every finding includes:

- Severity
- Confidence
- Evidence
- Affected columns
- Explanation
- Recommendation

The finding score is:

```text
severity_weight × confidence
```

| Severity | Weight |
| --- | ---: |
| 🟢 Low | 0.25 |
| 🟡 Medium | 0.50 |
| 🟠 High | 0.75 |
| 🔴 Critical | 1.00 |

The overall risk score is driven by the **strongest finding**.

---

## Structured findings

```python
for finding in report.findings:
    print("Detector:", finding.detector)
    print("Category:", finding.category)
    print("Severity:", finding.severity)
    print("Confidence:", finding.confidence)
    print("Columns:", finding.affected_columns)
    print("Evidence:", finding.evidence)
    print("Explanation:", finding.explanation)
    print("Recommendation:", finding.recommendation)
```

This makes the output suitable for Python applications, notebooks, CI/CD
pipelines, automated quality gates, internal ML platforms, and dashboards.

---

## Detector execution

DataLeaks tracks detector execution states:

```text
completed
failed
skipped
```

```python
print(report.completed_detectors)
print(report.failed_detectors)
print(report.skipped_detectors)
```

Detector failures are visible rather than silently interpreted as a clean run.

---

## Schema validation

DataLeaks validates important dataset assumptions, including:

- Supported input type
- Target column existence
- Train/test schema compatibility
- Missing columns
- Dtype mismatches
- Target presence

Example:

```text
Error: Target column 'churn' does not exist in the dataset
```

---

## JSON reporting

```bash
dataleaks train.csv --target target --output json
```

The report contains structured information about risk, findings,
recommendations, detector execution, schema validation, and metadata.

---

## Python API

### Basic

```python
from dataleaks import DataLeaks

report = DataLeaks(
    data,
    target="target",
).run()
```

### Train / validation / test

```python
report = DataLeaks(
    train,
    target="target",
    validation=validation,
    test=test,
).run()
```

### Configuration

```python
from dataleaks import DataLeaks
from dataleaks.schemas import DataLeaksConfig

config = DataLeaksConfig(
    enable_target_checks=True,
    enable_split_checks=True,
    enable_temporal_checks=True,
    enable_preprocessing_checks=True,
    enable_feature_checks=True,
    enable_cross_dataset_checks=True,
)

report = DataLeaks(
    df,
    target="target",
    config=config,
).run()
```

---

## Architecture

```text
Input
  ↓
DatasetContext
  ↓
DetectorRegistry
  ↓
DetectorRunner
  ↓
┌──────────┬────────┬──────────┬─────────┬────────────────┐
│ Target   │ Split  │ Temporal │ Feature │ Preprocessing  │
│ Checks   │ Checks │ Checks   │ Checks  │ Checks         │
└──────────┴────────┴──────────┴─────────┴────────────────┘
  ↓
Findings
  ↓
Aggregation
  ↓
Scoring
  ↓
Recommendations
  ↓
LeakageReport
  ├── Console
  └── JSON
```

Detectors implement a common interface and are managed through a registry,
allowing the system to grow without redesigning the reporting and scoring
pipeline.

---

## Configuration

```python
from dataleaks.schemas import DataLeaksConfig

config = DataLeaksConfig(
    duplicate_threshold=0.0,
    near_duplicate_threshold=0.95,
    confidence_threshold=0.5,
)
```

Configuration values are validated to remain within supported ranges.

---

## Example detection

A dataset containing features such as:

```text
customer_id
leaky_id
tenure_months
monthly_charges
target_leak_direct
post_outcome_refund
target
```

can produce findings such as:

```text
target_statistical
    HIGH
    Strong feature-target relationship detected

split_overlap
    HIGH
    Entity values overlap between train and test

feature_suspicious
    HIGH
    Post-outcome semantic signal detected

feature_identifier
    MEDIUM
    Identifier-like feature detected
```

---

## Design principles

**Execution-first**  
Analyze actual dataset and workflow evidence rather than relying exclusively on
static assumptions.

**Conservative detection**  
A suspicious signal is not automatically treated as proof of leakage.

**Explainability**  
Every finding should answer what happened, why it matters, and what to do next.

**Structured output**  
Findings are machine-readable objects that can be consumed by other systems.

**Extensibility**  
New detectors can be added without redesigning the entire architecture.

**Production awareness**  
Leakage can originate from datasets, splits, features, timestamps,
preprocessing, identifiers, and workflow metadata.

---

## Current scope and limitations

DataLeaks `0.1.0` focuses on dataset- and workflow-level leakage detection.

Some workflow-level checks require explicit metadata because operations
performed outside the dataset cannot always be inferred reliably. Preprocessing
contamination is one example.

Temporal metadata is explicitly supplied by the user rather than guessed
automatically.

> DataLeaks prefers explicit, explainable evidence over unsupported assumptions.

---

## Testing

**521 tests passed.**

The suite includes unit, integration, regression, adversarial, temporal,
schema, execution, scoring, reporting, CLI, and packaging coverage.

```bash
pytest -q -W error
```

---

## Building from source

```bash
python -m build
python -m twine check --strict dist/*
```

---

## Roadmap

Future releases will expand workflow-level capabilities, including:

- Scikit-learn pipeline adapters
- Additional ML framework adapters
- Richer workflow introspection
- Expanded temporal analysis
- Additional entity-resolution strategies
- Broader cross-dataset analysis
- CI/CD integrations
- Additional reporting formats
- ML workflow integrations

The planned adapter architecture will allow DataLeaks to inspect complete ML
workflows while preserving the same detector, scoring, and reporting system.

---

## Contributing

Contributions, detector ideas, bug reports, and adversarial regression cases are
welcome.

When adding a detector, include the implementation, tests, clear evidence, an
explanation, and an actionable recommendation.

Before submitting changes:

```bash
pytest -q -W error
```

---

## License

DataLeaks is released under the **MIT License**.

See [LICENSE](LICENSE).

---

**DataLeaks** — find leakage before it finds your model.

<div align="center">

# 🔍 DataLeaks

### Automatic Data Leakage Detection for Machine Learning

**Find leakage before it silently invalidates your model.**

[![PyPI](https://img.shields.io/pypi/v/dataleaks?style=for-the-badge&logo=pypi)](https://pypi.org/project/dataleaks/)
[![Python](https://img.shields.io/pypi/pyversions/dataleaks?style=for-the-badge&logo=python)](https://pypi.org/project/dataleaks/)
[![Tests](https://img.shields.io/badge/tests-521%20passed-success?style=for-the-badge)](#testing)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)

**Dataset → Detection → Findings → Risk → Recommendations**

</div>

---

## 🚨 Why DataLeaks?

A machine learning model can achieve excellent validation performance while
being fundamentally unreliable because information from the target, future,
evaluation data, or downstream workflow has leaked into the training process.

Common examples:

- A feature directly contains the target.
- A feature is mathematically derived from the target.
- The same entities appear in train and test.
- A feature contains information from the future.
- Preprocessing is fitted on the complete dataset before splitting.
- Training preprocessing is contaminated by held-out data.
- An identifier creates cross-split information leakage.
- A post-outcome feature is accidentally used for prediction.

**DataLeaks automatically analyzes these patterns and converts suspicious
signals into structured, explainable findings.**

---

# ✨ Features

| Capability | What DataLeaks Checks |
|---|---|
| 🎯 **Target Leakage** | Direct, statistical, and derived-target relationships |
| 🔀 **Split Leakage** | Duplicate, near-duplicate, and overlapping values |
| 🆔 **Identifier Leakage** | Identifier-like features and cross-split entity overlap |
| 🕵️ **Suspicious Features** | Strong feature-target relationships and semantic signals |
| ⏱️ **Temporal Leakage** | Future timestamps and chronological ordering |
| 🚧 **Preprocessing Leakage** | Fit-before-split and held-out-data contamination |
| 🌐 **Cross-Dataset Leakage** | Overlap between reference datasets |
| 🧪 **Schema Validation** | Missing columns, dtype mismatches, and target validation |
| 📊 **Risk Scoring** | Severity × confidence based risk assessment |
| 💡 **Recommendations** | Actionable remediation guidance |
| 📋 **Reporting** | Console and machine-readable JSON |
| ⚙️ **Execution Tracking** | Completed, failed, and skipped detectors |
| 🐍 **Python API** | Programmatic integration |
| 💻 **CLI** | One-command dataset analysis |

---

# 📦 Installation

Install DataLeaks directly from PyPI:

```bash
pip install dataleaks

⚡ Quick Start
import pandas as pd

from dataleaks import DataLeaks

df = pd.read_csv("train.csv")

report = DataLeaks(
    df,
    target="target",
).run()

print("Risk:", report.risk_level)
print("Score:", report.risk_score)
print("Findings:", report.finding_count)

for finding in report.findings:
    print(
        finding.detector,
        finding.severity,
        finding.explanation,
    )

The workflow is intentionally simple:

Dataset
   ↓
DataLeaks(...)
   ↓
.run()
   ↓
LeakageReport
💻 CLI

DataLeaks can also be used directly from the terminal.

Basic analysis
dataleaks train.csv --target target
Train / validation / test
dataleaks train.csv --target target --validation validation.csv --test test.csv
JSON output
dataleaks train.csv --target target --output json
🎯 Target Leakage

Target leakage occurs when a feature contains information that directly or
indirectly reveals the prediction target.

DataLeaks checks multiple forms of target leakage.

Direct leakage
target = churn
leaky_feature = churn

A feature that directly reproduces the target is a strong leakage signal.

Statistical leakage

Extremely strong numerical relationships between a feature and the target are
flagged for investigation.

Derived-target leakage

DataLeaks can identify deterministic relationships such as:

leaky_feature = 1 × target + 0

These relationships can make model performance appear unrealistically strong.

🔀 Split Leakage

DataLeaks analyzes relationships between training, validation, and test data.

It checks for:

Exact duplicate rows
Near-duplicate rows
Overlapping values
Entity overlap
Suspicious identifier overlap

Example:

TRAIN                    TEST

customer_id              customer_id
CUST1001   ────────────► CUST1001
CUST1002   ────────────► CUST1002
CUST1003                  CUST1007

When the same real-world entities appear across training and evaluation data,
model evaluation can become unreliable.

🆔 Identifier Leakage

Identifiers can become leakage channels when entities overlap across dataset
boundaries.

Examples include:

customer_id
user_id
patient_id
transaction_id
device_id

DataLeaks intentionally does not treat high cardinality alone as proof of
identifier leakage.

Identifier-like semantics and explicit metadata are used to make this detection
more conservative.

🕵️ Suspicious & Post-Outcome Features

Some features contain information that only becomes available after the event
being predicted.

Examples include:

future_purchase_value
post_outcome_refund
after_event_status
forecast_revenue
subsequent_transaction

DataLeaks combines statistical evidence with semantic signals to identify
features that deserve investigation.

Example:

Prediction
    │
    ├──► Model Input
    │
    └──► Refund Processed
             ↑
        happens later

Using a later event as an input for an earlier prediction introduces leakage.

⏱️ Temporal Leakage

Temporal leakage occurs when information that would not have been available at
prediction time becomes part of the model input.

DataLeaks can detect:

Invalid timestamps
Missing or unparsable timestamps
Chronological ordering problems
Feature timestamps occurring after prediction timestamps
Future-feature relationships
Conditional timestamp presence
Chronological validation
dataleaks train.csv --target target --time-column event_time
Future-feature detection
dataleaks train.csv --target target \
  --prediction-time-column prediction_time \
  --feature-time-columns signup_time outcome_time

DataLeaks compares feature timestamps with the prediction timestamp to identify
information that would not have been available when the prediction was made.

Conditional timestamps

Some timestamp fields are legitimately absent depending on the state or outcome
of a record.

dataleaks train.csv --target target \
  --conditional-time-column outcome_time \
  --conditional-target-column target \
  --conditional-present-when 1

This allows expected conditional absence to be distinguished from genuinely
invalid temporal data.

🚧 Preprocessing Leakage

Leakage can happen before model training even begins.

❌ Incorrect
Full Dataset
     │
     ▼
Preprocessing.fit()
     │
     ├──► Train
     └──► Test
✅ Correct
Full Dataset
     │
     ▼
Train / Test Split
     │
     ├──► Train → fit()
     │
     └──► Test  → transform()

DataLeaks can analyze preprocessing workflow metadata.

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

This can identify:

Preprocessing fitted before the split
Training preprocessing contaminated by held-out data
🌐 Cross-Dataset Leakage

DataLeaks can analyze overlap between datasets when reference data is supplied
through the supported dataset context.

This is useful for detecting shared entities or values that should remain
isolated between datasets.

📊 Risk Scoring

Every finding contains:

Severity
Confidence
Evidence
Explanation
Affected columns
Recommendation

The finding score is based on:

severity_weight × confidence
Severity weights
Severity	Weight
🟢 Low	0.25
🟡 Medium	0.50
🟠 High	0.75
🔴 Critical	1.00

The overall risk score is driven by the strongest finding.

This prevents many low-impact findings from hiding a single serious,
high-confidence leakage problem.

🔎 Structured Findings

DataLeaks returns structured finding objects instead of forcing users to parse
terminal output.

for finding in report.findings:
    print("Detector:", finding.detector)
    print("Category:", finding.category)
    print("Severity:", finding.severity)
    print("Confidence:", finding.confidence)
    print("Columns:", finding.affected_columns)
    print("Evidence:", finding.evidence)
    print("Explanation:", finding.explanation)
    print("Recommendation:", finding.recommendation)

This makes the output suitable for:

Python applications
Notebooks
CI/CD pipelines
Automated quality gates
Internal ML platforms
Custom dashboards
⚙️ Detector Execution

DataLeaks tracks the execution state of detectors.

Supported states:

completed
failed
skipped

Inspect execution information through the report:

print(report.completed_detectors)
print(report.failed_detectors)
print(report.skipped_detectors)

Detector failures are therefore visible instead of being silently interpreted as
a clean analysis.

🧪 Schema Validation

DataLeaks validates important dataset assumptions before analysis.

Validation includes:

Supported input type
Target column existence
Train/test schema compatibility
Missing columns
Dtype mismatches
Target presence

For example:

Error: Target column 'churn' does not exist in the dataset

Invalid input should fail clearly rather than producing misleading leakage
results.

📋 JSON Reporting

For automation and CI/CD:

dataleaks train.csv --target target --output json

The JSON report contains structured information about:

Risk score
Risk level
Findings
Recommendations
Detector execution
Schema validation
Metadata
🐍 Python API

The primary API:

from dataleaks import DataLeaks

report = DataLeaks(
    data,
    target="target",
).run()

Train / validation / test:

report = DataLeaks(
    train,
    target="target",
    validation=validation,
    test=test,
).run()

Custom configuration:

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
🧩 Architecture

DataLeaks follows an extensible detector architecture:

                         ┌───────────────┐
                         │     Input     │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │DatasetContext │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │DetectorRegistry│
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ DetectorRunner│
                         └───────┬───────┘
                                 │
          ┌──────────────┬───────┼────────┬──────────────┐
          ▼              ▼       ▼        ▼              ▼
       Target          Split  Temporal  Feature     Preprocessing
       Checks         Checks   Checks    Checks        Checks
          │              │       │        │              │
          └──────────────┴───────┴────────┴──────────────┘
                                 │
                                 ▼
                            ┌──────────┐
                            │ Findings │
                            └────┬─────┘
                                 │
                                 ▼
                           ┌───────────┐
                           │Aggregation│
                           └─────┬─────┘
                                 │
                                 ▼
                            ┌─────────┐
                            │ Scoring │
                            └────┬────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │ Recommendations│
                        └───────┬────────┘
                                │
                                ▼
                         ┌───────────────┐
                         │ LeakageReport │
                         └───────┬───────┘
                                 │
                       ┌─────────┴─────────┐
                       ▼                   ▼
                    Console              JSON

Detectors implement a common interface and are managed through a detector
registry, allowing the system to grow without redesigning the complete
reporting and scoring pipeline.

⚙️ Configuration

Detector categories and analysis thresholds can be configured:

from dataleaks.schemas import DataLeaksConfig

config = DataLeaksConfig(
    duplicate_threshold=0.0,
    near_duplicate_threshold=0.95,
    confidence_threshold=0.5,
)

Configuration values are validated to remain within supported ranges.

🧪 Example Detection

Consider a dataset containing:

customer_id
leaky_id
tenure_months
monthly_charges
target_leak_direct
post_outcome_refund
target

DataLeaks can produce findings such as:

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

The findings are then aggregated into a single leakage risk assessment.

🧠 Design Principles
Execution-first

Analyze actual dataset and workflow evidence rather than relying exclusively on
static assumptions.

Conservative detection

A suspicious signal is not automatically treated as proof of leakage.

Explainability

Every finding should answer:

What happened? Why does it matter? What should I do?

Structured output

Findings are machine-readable objects that can be consumed by other systems.

Extensibility

New detectors can be added without redesigning the entire architecture.

Production awareness

Leakage can originate from datasets, splits, features, timestamps,
preprocessing, identifiers, and workflow metadata.

⚠️ Current Scope & Limitations

DataLeaks 0.1.0 focuses on dataset- and workflow-level leakage detection.

Some workflow-level checks require explicit metadata because operations performed
outside the dataset cannot always be inferred reliably.

For example, preprocessing contamination requires information about how the
preprocessing workflow was fitted.

Temporal metadata is also explicitly supplied by the user rather than guessed
automatically.

This is intentional.

DataLeaks prefers explicit, explainable evidence over unsupported assumptions.

🧪 Testing

DataLeaks is heavily test-driven.

<div align="center">
✅ 521 Tests Passed
</div>

The test suite includes:

Unit tests
Integration tests
Regression tests
Adversarial datasets
Target leakage cases
Split leakage cases
Temporal edge cases
Timestamp parsing
Preprocessing contamination
Schema validation
Detector execution
Scoring
Reporting
CLI behavior
Packaging

Run the complete suite:

pytest -q -W error
📦 Building From Source

Build the distributions:

python -m build

This produces:

dist/
├── dataleaks-0.1.0-py3-none-any.whl
└── dataleaks-0.1.0.tar.gz

Validate them before publishing:

python -m twine check --strict dist/*
🛣️ Roadmap

DataLeaks 0.1.0 establishes the core leakage-detection engine.

Future releases will expand workflow-level capabilities:

🔌 scikit-learn pipeline adapters
🔌 Additional ML framework adapters
🔍 Richer workflow introspection
⏱️ Expanded temporal analysis
🧬 Additional entity-resolution strategies
🌐 Broader cross-dataset analysis
🤖 CI/CD integrations
📊 Additional reporting formats
🔗 ML workflow integrations

The planned adapter architecture will allow DataLeaks to inspect complete ML
workflows while preserving the same detector, scoring, and reporting system.

🤝 Contributing

Contributions, detector ideas, bug reports, and adversarial regression cases are
welcome.

When adding a detector, include:

Detector implementation
Unit tests
Regression tests where appropriate
Adversarial / false-positive tests where appropriate
Clear evidence
Explanation
Actionable recommendation

Before submitting changes:

pytest -q -W error
📄 License

DataLeaks is released under the MIT License.

See LICENSE.

<div align="center">
🔍 DataLeaks
Find leakage before it finds your model.

Reliable leakage detection for machine learning workflows.

</div> ```
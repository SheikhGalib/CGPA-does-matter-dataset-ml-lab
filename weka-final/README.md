# WEKA final evidence package

This folder supports the WEKA-only PowerPoint release.

- `evidence/` contains ARFF datasets, raw logs, J48 model files, exact tree text and DOT graphs, result screenshots, confusion matrices inside the logs, split indices and the slide-result audit.
- `scripts/prepare_data.py` verifies the raw-to-clean transformation and exports the three target-specific datasets.
- `scripts/WekaEvidence.java` runs all classifiers and builds the fixed-split J48 models.
- `scripts/RenderWeka.java` renders the exact WEKA trees and test-output captures.
- `scripts/assemble_evidence.py` validates confusion-matrix arithmetic, extracts result tables and writes the audit files.

Read [the detailed operation record](../docs/WEKA_LATEST_OPERATIONS.md) before quoting any metric.

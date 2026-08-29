# Prayan - ISAS 2026 BLE Indoor Localization Challenge

This repository contains the reproducible Python implementation used by **Team Prayan** for the **ISAS 2026 Challenge** on BLE-based indoor location recognition in a nursing-care facility.

The associated study is:

> **Multi-Resolution BLE Indoor Localization for Nursing Care under Signal Loss, Class Imbalance, and Temporal Shift**

The implementation was developed and executed as **Python scripts inside Docker**, not as Jupyter or Colab notebooks. The challenge submission form uses the word *notebook* for the training and test-prediction links; however, no `.ipynb` notebook was used in the actual experiments. The scripts in this repository are the code used for model evaluation, validation-artifact generation, and final challenge prediction.

---

## 1. Repository purpose

The repository has three main roles:

1. **Training and validation**  
   `phase24_fresh_cross_protocol_benchmark.py` performs the fresh development-data benchmark used for the manuscript.

2. **Confusion matrix and classification artifacts**  
   `generate_validation_artifacts.py` reconstructs the final strict-forward confusion matrix, per-class classification report, and classification figures from the structured CSV outputs produced by the Phase 24 benchmark.

3. **Final Apr14 challenge prediction**  
   `prayan_final_submission.py` trains the frozen selected architecture on all labeled development days and generates the final prediction column for the organizer-provided Apr14 test file.

The intended provenance chain is therefore:

```text
Raw development BLE + labels
            |
            v
phase24_fresh_cross_protocol_benchmark.py
            |
            +--> protocol/per-class/confusion CSV outputs
            |
            v
generate_validation_artifacts.py
            |
            +--> confusion matrix table
            +--> confusion matrix figure
            +--> classification report
            +--> classification figure

Frozen selected architecture
            |
            v
prayan_final_submission.py
            |
            v
Organizer Apr14 test template + prediction column
```

No manually typed confusion values or manually edited performance values are used in the validation-artifact generation script.

---

## 2. Important note about the challenge test labels

The official Apr14 challenge ground-truth location labels were **not provided to participants**.

Therefore:

- the Apr14 test prediction file can be generated;
- Apr14 accuracy, Macro-F1, Weighted-F1, confusion matrix, and classification report **cannot** be computed by participants;
- the confusion matrix and classification figures included in this repository are generated from the **labeled development data** under the primary **strict expanding-forward validation protocol** used to evaluate the final model.

The test-prediction script does **not** use Apr14 ground-truth labels.

---

## 3. Main files

### `phase24_fresh_cross_protocol_benchmark.py`

This is the fresh benchmark used for the results reported in the manuscript.

The script rebuilds the experiment from raw development BLE and location-label data. It deliberately does **not** read:

- previous model files;
- saved probability caches;
- saved prediction files;
- previous Phase 24 outputs;
- previous metric/result CSV files.

Features, augmentation samples, models, probabilities, predictions, metrics, class-level results, and confusion matrices are generated during the fresh run.

The benchmark evaluates the frozen candidate architectures under several validation protocols and writes structured result files to the output directory.

---

### `generate_validation_artifacts.py`

This script does **not** train the model.

It reads the structured CSV outputs produced by the Phase 24 benchmark and regenerates the validation tables and figures for the final selected system:

```text
MethodKey   : ensemble_class_aware
ProtocolKey : forward
Scope       : pooled
```

Before saving figures, the script independently reconstructs the following metrics from the pooled confusion matrix:

- Precision;
- Recall;
- F1;
- Macro-F1;
- Weighted-F1;
- Accuracy;
- class support.

It then verifies these reconstructed values against the corresponding Phase 24 protocol-summary and per-class-summary CSV files.

If a metric does not match, the script stops instead of generating the figures.

The verified run reported:

| Metric | Value |
|---|---:|
| Evaluation samples | 36,265 |
| Macro-F1 | 0.315929849342 |
| Weighted-F1 | 0.597298740714 |
| Accuracy | 0.605349510547 |

These values correspond to the pooled strict-forward development evaluation of the final selected system.

---

### `prayan_final_submission.py`

This is the final deployment script used to generate the challenge test prediction file.

It:

1. loads the labeled Apr10-Apr13 development data;
2. reconstructs the continuous one-second BLE state;
3. loads the organizer-provided Apr14 test template;
4. constructs the 10 s and 60 s feature representations;
5. trains the three frozen final-system branches on all labeled development days;
6. performs probability-level fusion;
7. applies the class-aware temporal decoder;
8. maps the one-second predictions back to every original test row;
9. verifies that the organizer-provided columns and row order remain unchanged;
10. appends the prediction column and writes the final CSV.

The submitted team filename is:

```text
Prayan_prediction.csv
```

The final test template contained:

- 62,222 raw BLE rows;
- 5,721 observed BLE seconds;
- one Apr14 test day;
- 25,769 seconds in the reconstructed continuous test grid.

The script's strict-template mode checks these values before writing the final submission.

---

## 4. Dataset and task representation

The challenge task is multiclass indoor location recognition from Bluetooth Low Energy (BLE) Received Signal Strength Indicator (RSSI) observations.

The model uses a fixed 25-beacon feature space:

```text
PROX_1 ... PROX_25
```

The official location vocabulary contains 23 classes:

```text
501
502
503
505
506
508
510
511
512
513
515
516
517
518
520
521
522
523
cafeteria
cleaning
hallway
kitchen
nurse station
```

The fresh development benchmark used labeled data from:

```text
2023-04-10
2023-04-11
2023-04-12
2023-04-13
```

The challenge test prediction is for:

```text
2023-04-14
```

After the required 10-second history condition, the common development evaluation population contains:

```text
47,121 labeled one-second observations
```

---

## 5. BLE preprocessing

The BLE preprocessing used by the final system is based on a one-second signal representation.

For each beacon:

1. raw detections are aggregated to **one-second mean RSSI**;
2. short missing intervals are forward-filled for at most **3 seconds**;
3. an exponentially weighted mean is applied with **span = 3**;
4. remaining unavailable RSSI values are represented as **-120 dBm**;
5. beacon availability/activity is retained as a feature rather than treating signal magnitude alone as the complete observation.

This design uses both:

- RSSI magnitude;
- signal availability/detection behavior.

The `power` field in the organizer's Apr14 template is preserved in the submitted CSV but is not used as a model feature.

---

## 6. Multi-resolution feature representation

The selected system combines a short 10-second representation with two 60-second representations.

### 6.1 10-second Random Forest branch

For every beacon, the basic feature representation contains:

- rolling mean RSSI;
- rolling standard deviation;
- rolling maximum RSSI;
- beacon activity/availability.

With 25 beacons:

```text
25 beacons x 4 features = 100 features
```

Training stride:

```text
1 second
```

---

### 6.2 60-second rich XGBoost branch

The rich representation contains, for each beacon:

- mean;
- standard deviation;
- maximum;
- activity;
- first temporal difference of rolling mean;
- RSSI relative to the strongest beacon.

It also includes three global descriptors:

- total active-beacon count;
- strongest RSSI;
- second-strongest RSSI.

Total:

```text
153 features
```

Training stride:

```text
1 second
```

---

### 6.3 60-second full-statistics XGBoost branch

The full-statistics representation contains, for each beacon:

- mean;
- standard deviation;
- variance;
- minimum;
- maximum;
- median;
- sum;
- activity;
- first temporal difference;
- relative RSSI.

It also contains the same global activity and strongest-signal descriptors.

Total:

```text
253 features
```

Training stride:

```text
10 seconds
```

---

## 7. Class imbalance handling

The development data are strongly imbalanced across room classes and days.

The selected model therefore uses training-fold-local augmentation. Held-out labels are not used to generate augmentation samples.

Two mechanisms are used in the selected branches:

### Signal-pattern relabeling

Minority patient-room classes can receive synthetic training evidence from eligible donor-room signal sequences.

The relabeling process uses the fixed floor-plan topology mapping and requires valid contiguous donor sequences.

Synthetic signal sequences are perturbed only on observed entries using Gaussian RSSI noise with:

```text
standard deviation = 1.5 dB
```

The final feature representation is then recomputed from the synthetic signal sequence.

### SMOTE

For the branches configured with SMOTE:

1. classes with fewer than six examples are first raised to at least six examples using random oversampling;
2. features are standardized;
3. SMOTE is applied with:

```text
k_neighbors = 4
```

4. generated features are returned to the original feature scale.

Common areas are not treated as minority patient rooms for the relabeling threshold.

---

## 8. Final selected prediction architecture

The final model is a three-branch probability ensemble.

| Branch | Window | Representation | Augmentation | Fusion weight |
|---|---:|---|---|---:|
| Random Forest | 10 s | basic, 100D | symmetric relabeling + SMOTE | 0.70 |
| XGBoost | 60 s | rich, 153D | symmetric relabeling + SMOTE | 0.15 |
| XGBoost | 60 s | full statistics, 253D | KL-partial relabeling | 0.15 |

### Random Forest

The selected augmented RF uses:

```text
n_estimators = 100
max_depth = 12
class_weight = balanced
random_state = 42
n_jobs = -1
```

The augmented RF uses the scikit-learn default `max_features`.

### XGBoost

The selected XGBoost branches use:

```text
n_estimators = 150
learning_rate = 0.07
max_depth = 7
objective = multi:softprob
random_state = 42
n_jobs = -1
```

Class-balanced sample weights are used during fitting.

---

## 9. Probability fusion and temporal decoding

Let the three class-probability vectors be:

```text
p_rich
p_full
p_RF
```

The rich 60-second XGBoost probabilities are temperature-adjusted using:

```text
T = 1.25
```

The full-statistics XGBoost and Random Forest branches use:

```text
T = 1.0
```

The final probability mixture is:

```text
p_ensemble =
    0.15 * p_rich
  + 0.15 * p_full
  + 0.70 * p_RF
```

### Class 510 recovery

A conservative recovery rule is applied when class 510 receives supporting evidence from the full XGBoost branch.

If:

```text
class-510 rank <= 2
and
p(510) > 0.15
```

then the fused class-510 score is multiplied by:

```text
1.5
```

This is a score adjustment, not a forced 510 prediction.

### Temporal voting

Final hard predictions are processed using an:

```text
11-second centered temporal vote
```

During this vote, class 518 receives a vote multiplier of:

```text
3.5
```

The centered vote is appropriate for the offline challenge setting. A live online implementation would require buffering or a causal replacement.

---

## 10. Validation protocols

The benchmark evaluates the methods under five protocols.

### 10.1 Row-random 70/30

Stratified row-level random split using seeds:

```text
42, 43, 44, 45, 46
```

Because overlapping temporal windows share signal history, this protocol is treated as an optimistic diagnostic rather than the primary deployment estimate.

### 10.2 45-second block-group random

Complete 45-second blocks are assigned to training or testing with a 59-second symmetric purge around the held-out blocks.

### 10.3 Event-group random

Complete labeled location events are assigned to training or testing with the same 59-second purge.

### 10.4 Leave-one-day-out (LODO)

Each development day from Apr11-Apr13 is held out while the other available development days are used for training.

### 10.5 Strict expanding-forward validation

This is the primary deployment-oriented evaluation:

```text
Train Apr10                  -> validate Apr11
Train Apr10 + Apr11          -> validate Apr12
Train Apr10 + Apr11 + Apr12  -> validate Apr13
```

No future labeled day is used to train an earlier validation fold.

The final pooled strict-forward result of the selected system is:

```text
Macro-F1    = 0.315929849342
Weighted-F1 = 0.597298740714
Accuracy    = 0.605349510547
Eval N      = 36,265
```

---

## 11. Why strict-forward evaluation is emphasized

BLE observations are temporally dependent. Consecutive windows can contain much of the same signal history.

For this reason, random row splitting and chronological future-day evaluation answer different prediction questions.

The repository retains the random and grouped protocols for comparison, but the strict expanding-forward protocol is emphasized because the challenge requires predicting a future unlabeled day after training on previous development days.

The repository does **not** claim that random cross-validation is universally invalid. Rather, it documents that validation choice has a large effect for this specific temporally structured BLE dataset.

---

## 12. Regenerating the training and validation benchmark

The benchmark expects the organizer data to be available locally.

The Docker paths used in the study were:

```text
/app/data/5f_label_loc_train.csv
/app/data/BLE Data
/app/data/phase21_floorplan_topology_v1.csv
/app/output
```

Example command:

```bash
docker compose run --rm \
  -e PYTHONHASHSEED=42 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  isas python phase24_fresh_cross_protocol_benchmark.py \
  --label-file /app/data/5f_label_loc_train.csv \
  --ble-dir "/app/data/BLE Data" \
  --topology-file /app/data/phase21_floorplan_topology_v1.csv \
  --output-dir /app/output \
  --seeds 42,43,44,45,46 \
  --protocols row_random,block45_random,event_random,lodo,forward \
  --methods all
```

The benchmark is computationally much more expensive than the validation-artifact script because it trains all requested models under all requested protocols.

Do not use `--skip-tcn` when reproducing the complete final manuscript benchmark, because that option is intended only for engineering smoke tests.

---

## 13. Phase 24 structured outputs used for validation artifacts

The verified manuscript run used:

```text
Run ID: 20260828_165704
```

The following structured outputs are sufficient to regenerate the validation tables and figures included here:

```text
phase24_confusion_detail_20260828_165704.csv
phase24_perclass_summary_20260828_165704.csv
phase24_protocol_summary_20260828_165704.csv
phase24_class_support_20260828_165704.csv
```

They are placed in:

```text
benchmark_outputs/
```

These are aggregate evaluation outputs, not hidden Apr14 labels.

---

## 14. Regenerating the confusion matrix and classification figures

The validation-artifact generator reads the Phase 24 CSV files and performs internal consistency checks.

Example command for a normal Python environment:

```bash
python generate_validation_artifacts.py \
  --input-dir benchmark_outputs \
  --output-dir results
```

The exact one-off Docker command used when Matplotlib was not already installed in the project image was:

```bash
docker compose run --rm \
  -e PYTHONHASHSEED=42 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  isas bash -lc "
    pip install matplotlib==3.11.1 &&
    python generate_validation_artifacts.py \
      --input-dir /app/output \
      --output-dir /app/output/validation_artifacts
  "
```

A successful run prints:

```text
VALIDATION ARTIFACT REGENERATION: PASS

Method       : ensemble_class_aware
Protocol     : forward (pooled)
Eval N       : 36,265
Macro F1     : 0.315929849342
Weighted F1  : 0.597298740714
Accuracy     : 0.605349510547
```

---

## 15. Generated validation artifacts

The `results/` directory contains:

### Confusion matrix

```text
strict_forward_confusion_matrix_counts.csv
strict_forward_confusion_matrix_normalized.csv
strict_forward_confusion_matrix.png
strict_forward_confusion_matrix.pdf
```

`strict_forward_confusion_matrix_counts.csv` is the 23 x 23 pooled strict-forward confusion matrix.

The normalized matrix is normalized by the true-class support.

---

### Classification report and figure

```text
strict_forward_classification_report.csv
strict_forward_classification_figure.png
strict_forward_classification_figure.pdf
```

The classification report contains, for every one of the 23 classes:

- Precision;
- Recall;
- F1;
- true support;
- predicted support.

The classification figure is generated directly from these reconstructed development-evaluation metrics.

---

### Basic RF versus final system

```text
strict_forward_basic_vs_final_f1.csv
strict_forward_basic_vs_final_f1.png
strict_forward_basic_vs_final_f1.pdf
```

This comparison shows per-class strict-forward F1 for:

- the 10-second Basic Random Forest baseline;
- the final selected multi-resolution system;
- the corresponding delta F1 for every class.

---

### Overall metrics

```text
strict_forward_metrics_summary.csv
```

This file contains the final pooled strict-forward Macro-F1, Weighted-F1, Accuracy, evaluation sample count, and number of classes.

---

## 16. Generating the final Apr14 challenge prediction

The final challenge prediction is generated using the frozen architecture after model selection.

For the final prediction run, the three selected branches are retrained using all available labeled development days:

```text
Apr10 + Apr11 + Apr12 + Apr13
```

and then applied to the organizer-provided Apr14 test file.

The exact command used was:

```bash
docker compose run --rm \
  -e PYTHONHASHSEED=42 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  isas python prayan_final_submission.py \
  --label-file /app/data/5f_label_loc_train.csv \
  --ble-dir "/app/data/BLE Data" \
  --test-file "/app/data/BLE_Test_predict.csv" \
  --output /app/output/Prayan_prediction.csv \
  --strict-test-template
```

The output contains every organizer-provided test row plus one additional column:

```text
prediction
```

The deployment script checks that:

- the output has exactly 62,222 rows;
- the Apr14 test date is correct;
- all 5,721 observed test seconds are represented;
- the reconstructed continuous grid contains 25,769 seconds;
- every prediction belongs to the official 23-class vocabulary;
- there are no missing predictions;
- all original organizer-provided columns retain the same values;
- the original row order is unchanged.

A separate diagnostics JSON is written for local verification and is not the challenge submission file.

---

## 17. Final Apr14 prediction is separate from validation metrics

The workflow intentionally separates model evaluation from final challenge prediction.

```text
Development labels available:
Apr10-Apr13
    |
    +--> validation metrics and confusion matrices

Apr14 labels hidden:
Apr14 BLE only
    |
    +--> final prediction CSV only
```

Consequently, the Apr14 prediction script does not report Apr14 F1, accuracy, or a confusion matrix.

Any such metric would require access to the hidden challenge labels.

---

## 18. Software environment

The verified fresh Phase 24 benchmark recorded the following environment:

```text
Python           3.11.0rc1
NumPy            1.26.4
Pandas           2.2.2
scikit-learn     1.5.1
imbalanced-learn 0.12.3
XGBoost          2.1.1
LightGBM         4.5.0
PyTorch          2.12.0.dev20260408+cu128
PyTorch CUDA     12.8
```

The GPU visible during that benchmark run was:

```text
NVIDIA GeForce RTX 5070 Laptop GPU
```

The validation-artifact regeneration additionally used:

```text
Matplotlib 3.11.1
```

PyTorch/LightGBM are included because the Phase 24 benchmark evaluates TCN and LightGBM comparison methods. They are not required by the three-branch final RF-XGBoost deployment architecture itself.

---

## 19. Suggested Python dependencies

A minimal requirements file for the benchmark and artifact generation should include the versions recorded by the verified run where practical:

```text
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
scikit-learn==1.5.1
imbalanced-learn==0.12.3
xgboost==2.1.1
lightgbm==4.5.0
matplotlib==3.11.1
```

The exact PyTorch build used by the benchmark was:

```text
torch==2.12.0.dev20260408+cu128
```

Because this is a development CUDA build, installation depends on the corresponding PyTorch package index/container environment. It should not be replaced silently by a different build when exact TCN reproduction is required.

The selected final RF-XGBoost challenge prediction does not require the TCN model.

---

## 20. Recommended repository layout

```text
prayan_isas_26/
|
|-- README.md
|-- requirements.txt
|
|-- phase24_fresh_cross_protocol_benchmark.py
|-- generate_validation_artifacts.py
|-- prayan_final_submission.py
|
|-- benchmark_outputs/
|   |-- phase24_confusion_detail_20260828_165704.csv
|   |-- phase24_perclass_summary_20260828_165704.csv
|   |-- phase24_protocol_summary_20260828_165704.csv
|   `-- phase24_class_support_20260828_165704.csv
|
`-- results/
    |-- strict_forward_confusion_matrix_counts.csv
    |-- strict_forward_confusion_matrix_normalized.csv
    |-- strict_forward_confusion_matrix.png
    |-- strict_forward_confusion_matrix.pdf
    |
    |-- strict_forward_classification_report.csv
    |-- strict_forward_classification_figure.png
    |-- strict_forward_classification_figure.pdf
    |
    |-- strict_forward_basic_vs_final_f1.csv
    |-- strict_forward_basic_vs_final_f1.png
    |-- strict_forward_basic_vs_final_f1.pdf
    |
    `-- strict_forward_metrics_summary.csv
```

---

## 21. Data availability and redistribution

The raw challenge BLE data, organizer test file, and location-label files are **not included in this repository**.

Participants should use the files distributed through the official ISAS 2026 Challenge.

This repository contains code and aggregate development-evaluation outputs required to document and regenerate the reported validation artifacts.

Expected local Docker paths used by the scripts are:

```text
/app/data/5f_label_loc_train.csv
/app/data/BLE Data/
/app/data/phase21_floorplan_topology_v1.csv
/app/data/BLE_Test_predict.csv
```

Paths can also be supplied through command-line arguments.

---

## 22. Reproducibility notes

### Randomness

The study uses:

```text
random_state = 42
```

for the frozen model configurations where applicable.

The random-split benchmark uses:

```text
42, 43, 44, 45, 46
```

Environment variables used during reproducible runs include:

```text
PYTHONHASHSEED=42
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

---

### No historical prediction-cache reuse

The Phase 24 benchmark was designed specifically as a fresh-run audit.

Its run metadata records:

```text
fresh_run                    = true
reads_previous_result_files  = false
reads_saved_probability_files = false
reads_saved_prediction_files = false
```

Therefore the reported Phase 24 results are not filled from old score files or cached predictions.

---

### Validation artifact verification

`generate_validation_artifacts.py` reconstructs the pooled strict-forward classification metrics directly from the confusion counts and compares them with the benchmark summary.

The script generates the figures only after those checks pass.

This is intended to make the link between the benchmark results and the published figures transparent.

---

## 23. Interpreting the results

The final strict-forward Macro-F1 is substantially lower than scores obtained under the easiest random-row evaluation.

This is expected for this challenge because:

- the task contains 23 fixed classes;
- some patient-room classes are extremely scarce;
- class availability changes across days;
- overlapping RSSI windows create substantial temporal dependence;
- some classes appearing on a held-out day have little or no real training evidence in the preceding days;
- Apr14 is a future-day prediction task rather than an independent random sample from the same rows.

For this reason, the repository reports several validation protocols rather than presenting only the highest random-split score.

The final challenge architecture was selected primarily for its behavior under the strict future-day setting, not for maximizing the easiest row-random result.

---

## 24. Model limitations

The repository should not be interpreted as demonstrating a clinically validated tracking system.

Important limitations include:

- data from a single nursing facility;
- a short development period;
- strong and time-varying class imbalance;
- extremely scarce observations for some rooms;
- unobserved environmental variation affecting BLE propagation;
- geometry-constrained signal-pattern relabeling;
- development-derived minority recovery constants;
- a centered temporal decoder that introduces offline/future-neighbor dependence;
- hidden Apr14 labels, preventing participant-side measurement of final test performance.

The work is intended as a challenge localization study and as an analysis of BLE localization under signal loss, imbalance, and temporal shift.

---

## 25. Quick start

### A. Re-run the complete Phase 24 development benchmark

```bash
docker compose run --rm \
  -e PYTHONHASHSEED=42 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  isas python phase24_fresh_cross_protocol_benchmark.py \
  --label-file /app/data/5f_label_loc_train.csv \
  --ble-dir "/app/data/BLE Data" \
  --topology-file /app/data/phase21_floorplan_topology_v1.csv \
  --output-dir /app/output
```

### B. Regenerate the confusion matrix and classification figures

```bash
python generate_validation_artifacts.py \
  --input-dir benchmark_outputs \
  --output-dir results
```

### C. Generate the final challenge test prediction

```bash
docker compose run --rm \
  -e PYTHONHASHSEED=42 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  isas python prayan_final_submission.py \
  --label-file /app/data/5f_label_loc_train.csv \
  --ble-dir "/app/data/BLE Data" \
  --test-file "/app/data/BLE_Test_predict.csv" \
  --output /app/output/Prayan_prediction.csv \
  --strict-test-template
```

---

## 26. Challenge submission clarification

The challenge form requested links to the *notebook* used for training and to the *notebook* used for test prediction.

This project did not use Jupyter/Colab notebooks for the final pipeline.

The repository link was supplied because the actual reproducible workflow consists of these Python scripts:

```text
Training / validation:
phase24_fresh_cross_protocol_benchmark.py

Confusion matrix / classification artifacts:
generate_validation_artifacts.py

Final Apr14 prediction:
prayan_final_submission.py
```

These files expose the actual executable pipeline rather than a separate notebook reimplementation.

---

## 27. Contact / review-period note

During double-anonymous manuscript review, this repository intentionally avoids placing author affiliations, personal contact information, or an unredacted manuscript in the README.

The repository is organized around **Team Prayan** and the challenge implementation.

---

## 28. Summary

The core reproducibility chain is:

```text
Development BLE + labels
        |
        v
Fresh Phase 24 benchmark
        |
        +--> strict-forward predictions/metrics
        +--> confusion/per-class CSV outputs
        |
        v
Validation artifact generator
        |
        +--> confusion matrix
        +--> classification report
        +--> classification figures

Frozen selected model
        |
        +--> retrain on Apr10-Apr13
        |
        v
Apr14 BLE test data
        |
        v
Prayan_prediction.csv
```

The validation artifacts are derived from labeled development evaluation, while the final Apr14 file contains predictions only because the challenge test labels are hidden.

# Developer Handover Report

## Robust Product Attribute-Value Extraction and Normalization from E-Commerce Product Descriptions

**Course:** CS F429 — Natural Language Processing\
**Project Type:** NLP / Information Extraction / Sequence Labeling\
**Primary Dataset Completed:** WDC-PAVE\
**Secondary Dataset Planned:** MAVE\
**Current Stage:** Classical CRF baseline complete; BERT baseline setup started\
**Handover Date:** 8 October 2026\
**Status:** Active development

---

# 1. Executive Summary

This project develops an NLP system for extracting structured product attribute-value pairs from noisy e-commerce titles and descriptions.

Given an input such as:

```text
Samsung Galaxy S24 with 8GB RAM, 256GB storage and
6.2-inch AMOLED display
```

the target output is conceptually:

```text
Brand          → Samsung
Model          → Galaxy S24
RAM            → 8GB
Storage        → 256GB
Display Size   → 6.2 inch
Display Type   → AMOLED
```

The project is structured as a progression from classical NLP to contextual deep-learning models:

```text
Raw e-commerce data
        ↓
Dataset inspection
        ↓
Span matching
        ↓
BIO sequence construction
        ↓
Classical CRF baseline
        ↓
BERT token-classification baseline
        ↓
DeBERTa / stronger Transformer
        ↓
Attribute-value normalization
        ↓
Hybrid extraction + normalization system
        ↓
Long-tail and cross-dataset analysis
```

The WDC-PAVE preprocessing and CRF baseline are complete.

The final CRF achieved:

```text
Official WDC-PAVE Test Micro F1 = 0.7934
Precision                       = 0.8408
Recall                          = 0.7510
Macro F1                        = 0.7483
Token Accuracy                  = 0.9282
```

The CRF performs especially well for structured attributes such as product identifiers, model numbers, UPC values and rotational speeds, but struggles with semantically contextual attributes such as product type, material, flavor and some dimension-related properties.

The next development stage is BERT-based token classification using exactly the same BIO data so that the classical and Transformer models can be compared fairly.

---

# 2. Project Objective

The primary objective is to construct a robust product attribute-value extraction system for e-commerce data.

The model should identify spans of text corresponding to product attributes such as:

```text
Brand
Product Type
Manufacturer
Capacity
Material
Color
Model Number
Part Number
Interface
Dimensions
Processor properties
Jewelry properties
Food properties
etc.
```

The project additionally aims to normalize extracted values.

Example:

```text
8192 MB      → 8 GB
8GB          → 8 GB
13.6"        → 13.6 inch
10000 RPM    → 10K RPM
```

The final system is intended to combine:

```text
contextual extraction
+
rule-based or structured normalization
```

rather than stopping at raw sequence labeling.

---

# 3. Research Questions

The project is currently organized around three main research questions.

## RQ1 — Classical vs Transformer Extraction

How much improvement do pretrained Transformer representations provide over a classical CRF for product attribute-value extraction?

---

## RQ2 — Normalization

Can a normalization layer improve the consistency and usability of extracted product values?

---

## RQ3 — Long-Tail Robustness

How do classical and Transformer models perform on:

```text
frequent attributes
vs
rare / long-tail attributes?
```

This is especially important because WDC-PAVE contains several attributes with fewer than 20 positive training examples.

MAVE is planned as the larger-scale dataset for strengthening the long-tail analysis.

---

# 4. Current Development Status

## Completed

```text
✓ HPC environment exploration
✓ GPU allocation procedure identified
✓ Conda environment created
✓ WDC-PAVE extraction dataset downloaded
✓ Dataset schema inspected
✓ Dataset statistics generated
✓ Train / validation split constructed
✓ Official test set protected
✓ Exact span matching analyzed
✓ Character-offset aware tokenization implemented
✓ BIO dataset constructed
✓ BIO dataset validated
✓ Artificial duplicate-span issue discovered
✓ Span-selection algorithm corrected
✓ Overlap handling analyzed
✓ Classical CRF implemented
✓ CRF validation baseline trained
✓ Hyperparameter search performed
✓ CRF validation error analysis performed
✓ Error taxonomy implemented and corrected
✓ Final CRF hyperparameters frozen
✓ Final CRF trained on train + validation
✓ CRF evaluated once on official WDC-PAVE test set
✓ Classical baseline documented
```

## In Progress

```text
→ BERT environment preparation
→ BERT WordPiece/BIO alignment
```

## Not Yet Started

```text
□ BERT fine-tuning
□ BERT validation analysis
□ BERT final test evaluation
□ DeBERTa baseline
□ Transformer vs CRF comparison
□ Normalization module
□ Hybrid extraction + normalization system
□ MAVE reconstruction
□ MAVE experiments
□ Final CLI / GUI
□ Final paper and presentation
```

---

# 5. HPC Environment

The project currently runs on the institute HPC system.

## Login Host

```text
hpc.bits-hyderabad.ac.in
```

Account used:

```text
csisnlp_09
```

The login node observed during development was:

```text
hpc01
```

Operating system:

```text
Rocky Linux 8.10
```

---

# 6. Important HPC Policy

Heavy computation, model training, compilation and package installation should not be performed on the login node.

The accelerator node should be requested through SLURM.

The user has indicated that normal compute-node access is not available for this account, so the course accelerator allocation is used for both preprocessing/model development and GPU training when required.

---

# 7. Course GPU Allocation

The working GPU partition is:

```text
gpu_rtx_pro_6000_6_csis_hyd
```

Account:

```text
bits
```

QoS:

```text
gpu_csis_course
```

Observed QoS limit:

```text
CPU      : 4
Memory   : up to 80G
GPU MPS  : 50
```

The important detail is that the course allocation requires:

```text
--gres=mps:50
```

and not:

```text
--gres=gpu:1
```

---

# 8. Working SLURM Command

Use:

```bash
srun \
  --partition=gpu_rtx_pro_6000_6_csis_hyd \
  --account=bits \
  --qos=gpu_csis_course \
  --gres=mps:50 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=01:00:00 \
  --pty bash
```

Previous successful allocation produced:

```text
Node: gpunode8
GPU: NVIDIA RTX PRO 6000 Blackwell
GPU memory: approximately 98 GB
Driver-supported CUDA: 13.0
```

---

# 9. Current GPU Queue Situation

At the time of handover, a new accelerator request was submitted.

SLURM reported:

```text
srun: job 385414 queued and waiting for resources
```

Therefore the BERT stage has not yet proceeded to actual GPU training.

Check it with:

```bash
squeue -j 385414
```

Detailed reason:

```bash
scontrol show job 385414
```

If necessary:

```bash
scancel 385414
```

Do not submit many duplicate interactive GPU jobs.

---

# 10. Conda Environment

Environment name:

```text
nlp_project
```

Typical activation:

```bash
module load anaconda3-2022.05-gcc-11.2.0-od5lltp
conda activate nlp_project
```

Then:

```bash
cd ~/nlp_project
```

The environment was originally created as a minimal classical NLP environment.

Core dependencies used for the CRF stage include:

```text
numpy
pandas
scikit-learn
sklearn-crfsuite
seqeval
tqdm
matplotlib
```

PyTorch/Transformers installation for the BERT stage was planned but was not confirmed completed at the time of this handover.

---

# 11. Storage Layout

Code is stored in the home directory.

```text
~/nlp_project/
```

Large datasets and model artifacts are stored in scratch.

```text
/scratch/csisnlp_09/
```

This is intentional because the home quota is approximately:

```text
40 GiB
```

while scratch is intended for larger working data.

Important HPC note:

Scratch data may be automatically removed after prolonged inactivity.

Therefore:

```text
source code
documentation
reports
configuration
```

should remain in the home directory or another persistent location.

---

# 12. Project Directory Structure

Current structure is approximately:

```text
~/nlp_project/
│
├── scripts/
│   ├── inspect_dataset.py
│   ├── create_split.py
│   ├── check_span_matches.py
│   ├── build_bio_dataset.py
│   ├── validate_bio_dataset.py
│   ├── train_crf.py
│   ├── tune_crf.py
│   ├── analyze_crf_errors.py
│   ├── analyze_error_types.py
│   ├── train_final_crf.py
│   └── check_bert_data.py          # planned / current BERT step
│
├── src/
├── configs/
├── results/
└── notebooks/
```

Scratch structure:

```text
/scratch/csisnlp_09/
│
├── data/
│   └── wdc_pave/
│       ├── raw/
│       │   └── extraction/
│       ├── interim/
│       │   └── splits/
│       └── processed/
│           └── crf/
│
└── models/
```

---

# 13. Dataset Strategy

Two datasets are part of the intended final project.

## Dataset 1 — WDC-PAVE

Current primary dataset.

Used to establish the complete end-to-end pipeline.

Advantages:

```text
ready product text
attribute annotations
manageable scale
multiple product domains
suitable for CRF and Transformer experiments
```

All current numerical results refer to WDC-PAVE unless explicitly stated otherwise.

---

# 14. Dataset 2 — MAVE

MAVE remains planned for the final project.

The motivation is to test:

```text
larger-scale behavior
greater category diversity
long-tail extraction
generalizability
```

MAVE has not yet been reconstructed locally.

An important implementation complication discovered earlier is that the public MAVE release does not directly provide the complete product-text dataset in the same ready-to-train form as the WDC-PAVE files used here.

The text needs to be reconstructed using the associated Amazon metadata and MAVE labels.

Therefore MAVE was intentionally postponed until the WDC-PAVE pipeline became stable.

Recommended strategy:

```text
Finish:
WDC-PAVE CRF
WDC-PAVE BERT
WDC-PAVE DeBERTa

Then reuse the stable pipeline for MAVE.
```

The CRF should also eventually be run on the chosen MAVE experimental subset to ensure fair comparison.

Do not compare:

```text
CRF on WDC-PAVE
vs
BERT on MAVE
```

as the primary classical-vs-modern experiment.

---

# 15. WDC-PAVE Raw Dataset

Data directory:

```text
/scratch/csisnlp_09/data/wdc_pave/raw/extraction/
```

Files:

```text
train_1.0.jsonl
test.jsonl
```

Parsed product counts:

```text
Official train = 1066
Official test  = 354
Total          = 1420
```

A shell `wc -l` count may show one fewer test row because the final JSONL line does not necessarily contain a trailing newline.

Python parsing confirmed 354 test records.

---

# 16. WDC-PAVE Record Schema

Each product contains approximately:

```text
id
cluster_id
category
input_title
input_description
target_scores
url
```

Example concept:

```text
Category:
Office Products

Title:
Xerox TONER CARTRIDGE FOR WORKCENTRE ...

Description:
4250 4260MFP SERIES 25K PAGE YIELD ...

Attributes:
Brand
Product Type
Page Yield
Color(s)
Retail UPC
Manufacturer Stock Number
...
```

---

# 17. Annotation Representation

Example:

```json
{
  "Brand": {
    "Xerox": {
      "pid": [0],
      "score": 1
    }
  }
}
```

Interpretation:

```text
pid = 0 → title
pid = 1 → description
```

Some annotations may reference both.

---

# 18. Missing Attributes

Many attribute entries contain:

```text
n/a
```

These represent absence of an extractable value.

They are useful for dataset statistics but are not BIO spans.

Decision:

```text
n/a values are ignored during BIO sequence construction.
```

This is essential because approximately:

```text
57.74%
```

of WDC-PAVE attribute annotations are `n/a`.

---

# 19. Full Dataset Statistics

Observed WDC-PAVE statistics:

```text
Products                : 1420
Unique product IDs      : 1420
Unique clusters         : 1420
Categories              : 5
Observed attributes     : 51
Annotated attributes    : 21904
Extractable values      : 11935
n/a values              : 12648
n/a percentage          : 57.74%
```

Extractable + n/a values may exceed annotated-attribute count because an attribute can contain more than one candidate value.

---

# 20. Category Distribution

```text
Computers And Accessories    436
Home And Garden              356
Office Products              297
Jewelry                      250
Grocery And Gourmet Food      81
```

This imbalance matters when interpreting model performance by category.

---

# 21. Text Statistics

Observed full-dataset text statistics:

## Titles

```text
Mean    ≈ 12.36 tokens/words
Median  ≈ 13
Maximum ≈ 25
```

## Descriptions

```text
Mean    ≈ 45.67
Maximum ≈ 172
```

These sequences are relatively short, which is favorable for BERT.

---

# 22. Train / Validation / Test Protocol

The original WDC-PAVE training split was divided into:

```text
Train       906 products
Validation  160 products
```

using:

```text
test_size = 0.15
random_state = 42
```

with category stratification.

Official test remained untouched:

```text
354 products
```

Therefore:

```text
Development:
906 + 160 = 1066

Official test:
354
```

---

# 23. Leakage Checks

Train/test and train/validation leakage was checked using product and cluster identifiers.

Observed official train/test overlap:

```text
Product IDs  : 0
Cluster IDs  : 0
```

The created development split similarly prevents product/cluster overlap.

---

# 24. Span Matching

Before converting annotations to BIO sequences, positive values were searched in their declared title or description.

Initial positive-span analysis:

```text
Total positive values : 7627
Matched values        : 7592
Unmatched values      : 35
Exact match rate      : 99.54%
```

Because exact matching already achieved approximately 99.5%, fuzzy matching was deliberately avoided for the classical baseline.

This keeps preprocessing simpler and reduces heuristic contamination.

---

# 25. Examples of Unmatched Values

Some unmatched annotations are caused by non-contiguous realization.

Example:

```text
Gold:
Ultra320 SCSI

Text:
Ultra320 3.5-inch SCSI
```

Another:

```text
Gold:
Hewlett-Packard Proliant

Text:
...Proliant...Hewlett-Packard...
```

Such values cannot be represented cleanly as one contiguous BIO span.

These are retained as known dataset/model limitations.

---

# 26. BIO Representation

Attribute extraction is treated as token-level sequence labeling.

Labels:

```text
B-X = beginning of attribute X
I-X = continuation of attribute X
O   = outside
```

Example:

```text
Xerox       B-BRAND
TONER       B-PRODUCT_TYPE
CARTRIDGE   I-PRODUCT_TYPE
```

---

# 27. Attribute Label Normalization

Human-readable attribute names are converted to machine-friendly labels.

Examples:

```text
Product Type
→ PRODUCT_TYPE

Color(s)
→ COLOR_S

Manufacturer Stock Number
→ MANUFACTURER_STOCK_NUMBER
```

---

# 28. Tokenization

Plain whitespace tokenization was insufficient because WDC-PAVE contains noisy strings such as:

```text
BlackALTERNATE
```

when the gold value might be only:

```text
Black
```

The preprocessing therefore performs character-offset-aware tokenization.

Conceptually:

```text
raw text
   ↓
gold character spans
   ↓
force span boundaries into tokenization
   ↓
token offsets
   ↓
BIO alignment
```

---

# 29. Initial Duplicate-Occurrence Problem

The first version of BIO conversion labeled every occurrence of an annotated value.

Example:

```text
Processor Quantity = "1"
```

If a description contained several unrelated instances of the digit `1`, all were initially considered candidates.

This produced too many artificial overlaps.

Initial train rejected overlaps:

```text
627
```

---

# 30. Improved Span Selection

The BIO converter was modified so that each:

```text
(attribute, value, source)
```

uses one representative occurrence.

Selection preference:

```text
1. fewer possible textual occurrences
2. longer/more-specific values
3. non-overlapping occurrence
```

After correction:

```text
Train overlaps      627 → 120
Validation          103 → 15
Test                255 → 62
```

This was a major preprocessing improvement.

---

# 31. Final BIO Dataset

Directory:

```text
/scratch/csisnlp_09/data/wdc_pave/processed/crf/
```

Files include:

```text
train.jsonl
val.jsonl
test.jsonl

train_unmatched.jsonl
val_unmatched.jsonl
test_unmatched.jsonl

train_overlaps.jsonl
val_overlaps.jsonl
test_overlaps.jsonl
```

---

# 32. Final Training BIO Statistics

```text
Products             906
Sequences            1812
Tokens               64962
Positive tokens      14050
Positive-token rate  21.63%
Unique BIO labels    91
Positive attributes  51
Unmatched values     40
Rejected overlaps    120
BIO errors           0
Length errors        0
```

---

# 33. Final Validation BIO Statistics

```text
Products             160
Sequences            320
Tokens               11536
Positive tokens      2598
Positive-token rate  22.52%
Positive attributes  50
Unmatched values     7
Rejected overlaps    15
BIO errors           0
Length errors        0
```

---

# 34. Final Test BIO Statistics

```text
Products             354
Sequences            708
Tokens               24952
Positive tokens      5526
Positive-token rate  22.15%
Positive attributes  51
Unmatched values     16
Rejected overlaps    62
BIO errors           0
Length errors        0
```

---

# 35. Unseen BIO Forms

All attributes occurring in validation and test are represented in training.

However, some particular `I-` forms are absent from training.

Validation:

```text
I-CACHE
I-MODEL_NUMBER
```

Test additionally includes:

```text
I-STONE_SHAPE
I-THERMAL_DESIGN_POWER
```

Interpretation:

The attribute exists in training, but training may contain only single-token examples.

Example:

```text
8MB
→ B-CACHE
```

while test might contain:

```text
8   B-CACHE
MB  I-CACHE
```

This was not treated as a preprocessing error.

---

# 36. Long-Tail Training Attributes

Examples of rare positive training attributes:

```text
STONES_TOTAL_WEIGHT       6
BUS_SPEED                 6
SHEET_PERFORATION        10
PAGE_YIELD               13
COOLING                  15
CLOSURE                  18
PAPER_WEIGHT             18
THERMAL_DESIGN_POWER     19
MOUNTING                 22
VOLTAGE                  22
PROCESSOR_CORE           23
SHELVES                  23
```

This makes the dataset suitable for long-tail analysis.

---

# 37. Classical Baseline

The classical baseline is a linear-chain Conditional Random Field implemented using:

```text
sklearn-crfsuite
```

Feature groups:

```text
token identity
lowercase token
prefixes
suffixes
capitalization
digit features
character structure
hyphen/slash/dot indicators
word shape
previous token
next token
title vs description
BOS/EOS
```

No pretrained embeddings or neural components are used.

---

# 38. CRF Initial Configuration

```text
algorithm                 = lbfgs
c1                        = 0.1
c2                        = 0.1
max_iterations            = 100 initially
all_possible_transitions  = True
```

The first run generated approximately:

```text
67,829 CRF features
```

and hit the maximum 100 iterations.

The final configuration therefore uses:

```text
max_iterations = 150
```

---

# 39. Initial CRF Validation Results

```text
Precision       0.8344
Recall          0.7481
Entity F1       0.7889
Token Accuracy  0.9248
Macro F1        0.7650
```

Primary metric:

```text
entity-level Micro F1
```

Token accuracy is secondary because approximately 78% of tokens are `O`.

---

# 40. CRF Hyperparameter Search

Evaluated:

```text
c1      c2

0.01    0.01
0.01    0.10
0.10    0.01
0.10    0.10
0.10    1.00
1.00    0.10
1.00    1.00
```

Best validation configuration:

```text
c1 = 0.1
c2 = 0.1

Precision = 0.8340
Recall    = 0.7494
Micro F1  = 0.7894
Macro F1  = 0.7596
```

The small improvement over the initial run indicates that the remaining limitation is mostly representational rather than hyperparameter-related.

---

# 41. Frozen CRF Configuration

Final:

```text
algorithm                  = lbfgs
c1                         = 0.1
c2                         = 0.1
max_iterations             = 150
all_possible_transitions   = True
```

No modification was allowed after examining official test results.

---

# 42. Validation Performance by Source

```text
Title:
Precision  0.8351
Recall     0.7704
F1         0.8014

Description:
Precision  0.8331
Recall     0.7314
F1         0.7789
```

Titles are easier than descriptions.

---

# 43. Validation Performance by Category

```text
Jewelry                       0.9327
Computers And Accessories     0.8837
Home And Garden               0.7338
Office Products               0.6628
Grocery And Gourmet Food      0.5739
```

This reveals substantial domain variation.

---

# 44. CRF Error Analysis

Strict validation errors originally consisted of:

```text
False negatives  400
False positives  238
```

A more detailed taxonomy was created.

---

# 45. Final Error Taxonomy

After fixing one-to-one matching between gold and predicted entities:

```text
Complete miss          266   52.78%
Spurious prediction    104   20.63%
Boundary error          76   15.08%
Attribute confusion     58   11.51%
```

The reconstructed totals exactly match:

```text
False negatives = 400
False positives = 238
```

---

# 46. Major Complete-Miss Attributes

```text
PRODUCT_TYPE                     65
MATERIAL                         33
BRAND                            20
PART_NUMBER                      16
COLOR_S                          14
MANUFACTURER                     12
CAPACITY                         10
INTERFACE                         9
MANUFACTURER_STOCK_NUMBER         8
```

---

# 47. Boundary Errors

Most boundary errors occur for:

```text
PRODUCT_TYPE  41
MATERIAL      10
CAPACITY       5
```

Example:

```text
Gold:
Plastic Step Trash Can

Prediction:
Step Trash Can
```

Another:

```text
Gold:
metal and plastic

Prediction:
plastic
```

---

# 48. Attribute Confusions

Common confusions:

```text
BRAND         → MANUFACTURER        7
MANUFACTURER  → BRAND               6
LENGTH        → WIDTH               5
COLOR         → COLOR_S             4
HEIGHT        → WIDTH               4
WIDTH         → HEIGHT              4
MATERIAL      → PRODUCT_TYPE        4
```

These are semantically interpretable mistakes.

---

# 49. Final CRF Training

After all decisions were frozen, train and validation were combined.

```text
906 + 160 = 1066 products
```

Sequences:

```text
1812 + 320 = 2132
```

The model was retrained using all 1,066 development products.

It was then evaluated once on the untouched official test set.

---

# 50. Final Official CRF Result

## Main Metrics

```text
Entity Precision   0.8408
Entity Recall      0.7510
Micro F1           0.7934
Macro F1           0.7483
Token Accuracy     0.9282
Weighted F1        0.7865
```

The primary classical baseline is:

# CRF Micro F1 = 0.7934

---

# 51. Final CRF Source Performance

```text
Title
P = 0.8513
R = 0.7888
F1 = 0.8189

Description
P = 0.8310
R = 0.7179
F1 = 0.7703
```

The title/description gap therefore generalizes from validation to test.

---

# 52. Final CRF Category Performance

```text
Jewelry
F1 = 0.9302

Computers And Accessories
F1 = 0.8847

Home And Garden
F1 = 0.7402

Office Products
F1 = 0.6700

Grocery And Gourmet Food
F1 = 0.6224
```

The same category ranking observed on validation largely persists on test.

---

# 53. Important High-Support Test Attributes

## Product Type

```text
Support    626
Precision  0.7212
Recall     0.5990
F1         0.6545
```

This is arguably the most important CRF weakness because it has very high support.

---

## Material

```text
Support    181
Precision  0.7123
Recall     0.5746
F1         0.6361
```

Another major semantic challenge.

---

## Brand

```text
Support    239
F1         0.8363
```

---

## Manufacturer

```text
Support    273
F1         0.8659
```

---

# 54. Strong Structured Attributes

Examples:

```text
RETAIL_UPC
F1 = 1.0000

PROCESSOR_CORE
F1 = 1.0000

ROTATIONAL_SPEED
F1 = 0.9942

METAL_TYPE
F1 = 0.9821

MANUFACTURER_STOCK_NUMBER
F1 = 0.9446

MODEL_NUMBER
F1 = 0.9383

INTERFACE
F1 = 0.9100
```

These demonstrate where classical handcrafted features work extremely well.

---

# 55. Weak / Rare Attributes

Examples:

```text
FLAVOR
F1 = 0.0833

PAGE_YIELD
F1 = 0.1429

PAPER_WEIGHT
F1 = 0.2222

SHELVES
F1 = 0.2222

SHEET_PERFORATION
F1 = 0.3333
```

Many of these also have limited training/test support.

Support must always be considered when interpreting their F1.

---

# 56. CRF Conclusions

The classical model provides a strong benchmark.

It performs best when values have:

```text
strong lexical patterns
digits
identifier formatting
units
consistent neighboring syntax
```

It struggles more when extraction requires:

```text
semantic understanding
longer phrase boundaries
attribute-role interpretation
domain-specific contextual understanding
rare labels
```

This provides a strong scientific motivation for Transformer experiments.

---

# 57. Key Scripts from the CRF Stage

## `inspect_dataset.py`

Dataset statistics and schema analysis.

## `create_split.py`

Creates the 906/160 train-validation split.

## `check_span_matches.py`

Checks exact textual recoverability of positive values.

## `build_bio_dataset.py`

Converts WDC-PAVE to BIO sequences.

## `validate_bio_dataset.py`

Checks BIO consistency, label coverage, rare attributes and overlap behavior.

## `train_crf.py`

Trains and validates the baseline CRF.

## `tune_crf.py`

Searches CRF `c1/c2`.

## `analyze_crf_errors.py`

Analyzes source/category performance and FP/FN counts.

## `analyze_error_types.py`

Separates errors into:

```text
complete miss
boundary error
attribute confusion
spurious prediction
```

## `train_final_crf.py`

Retrains frozen CRF on train + validation and evaluates official test.

---

# 58. CRF Outputs

Model:

```text
/scratch/csisnlp_09/models/wdc_pave_crf_final.pkl
```

Test results:

```text
~/nlp_project/results/crf_test_results.txt
```

Test predictions:

```text
~/nlp_project/results/crf_test_predictions.jsonl
```

Validation results:

```text
~/nlp_project/results/crf_validation_results.txt
```

Hyperparameter search:

```text
~/nlp_project/results/crf_hyperparameter_search.csv
```

Validation predictions:

```text
~/nlp_project/results/crf_val_predictions.jsonl
```

---

# 59. Current Transformer Plan

The next model is:

```text
google-bert/bert-base-uncased
```

The Transformer experiment must reuse exactly the same underlying WDC-PAVE BIO annotations.

Do not regenerate or change the gold data merely to improve BERT.

This ensures:

```text
CRF gold data
=
BERT gold data
```

and therefore allows direct comparison.

---

# 60. Planned BERT Development Protocol

Return to the development split:

```text
Train       906 products
Validation  160 products
```

Do not immediately train BERT using train + validation.

Development protocol:

```text
906 train
   ↓
BERT training
   ↓
160 validation
   ↓
model/hyperparameter selection
   ↓
freeze configuration
   ↓
1066 train + validation
   ↓
official test evaluation
```

This mirrors the CRF methodology.

---

# 61. BERT Label Vocabulary

There are:

```text
51 attributes
```

The CRF observed only 91 BIO states because some `I-X` states never occurred in training.

For BERT, define the complete schema from the training attributes:

```text
O

B-ATTRIBUTE_1
I-ATTRIBUTE_1

...

B-ATTRIBUTE_51
I-ATTRIBUTE_51
```

Therefore:

```text
1 + 51 × 2
=
103 labels
```

This does not introduce validation information because the attribute identities already come from training.

It only establishes a complete output schema.

---

# 62. BERT WordPiece Alignment

Current BIO labels are word-level.

BERT operates on WordPieces.

Example:

```text
Original:
508232-B21

BIO:
B-PART_NUMBER
```

BERT may split it into multiple pieces.

The recommended approach for this project is:

```text
first WordPiece of original word
→ original BIO label

remaining WordPieces
→ -100
```

Special tokens also get:

```text
-100
```

so they are ignored by the loss.

Evaluation must be converted back to the original word level so BERT and CRF are directly comparable.

---

# 63. Planned BERT Data Check

A script named:

```text
scripts/check_bert_data.py
```

was planned to verify:

```text
51 training attributes
103 classifier labels
no unknown labels
WordPiece lengths
number of sequences >128
number >256
number >512
example word-to-subword label alignment
```

This check should be run before actual fine-tuning.

At the time of handover, this validation had not yet been completed.

---

# 64. Planned Initial BERT Configuration

Recommended first experiment:

```text
Model               google-bert/bert-base-uncased

Learning rate       2e-5
Batch size          16
Epochs              up to 5
Weight decay        0.01
Seed                 42

Primary metric      validation entity Micro F1
```

Use dynamic batch padding.

Mixed precision may be used if supported by the installed PyTorch/GPU stack.

Given the Blackwell GPU, BF16 should be checked explicitly before enabling it.

---

# 65. GPU/PyTorch Verification Required

Once the SLURM allocation starts, activate:

```bash
module load anaconda3-2022.05-gcc-11.2.0-od5lltp
conda activate nlp_project
```

Check whether PyTorch is already installed:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16:", torch.cuda.is_bf16_supported())
PY
```

Desired result:

```text
CUDA available: True
GPU: NVIDIA RTX PRO 6000 Blackwell
```

If CUDA is `False`, do not begin model training.

Resolve PyTorch installation first.

---

# 66. BERT Success Criterion

The first major comparison will be:

```text
CRF validation Micro F1
0.7894

vs

BERT validation Micro F1
?
```

After freezing BERT:

```text
CRF official test Micro F1
0.7934

vs

BERT official test Micro F1
?
```

A useful Transformer result is not only one with higher aggregate F1.

The analysis should also identify where gains come from.

---

# 67. Specific BERT Hypotheses

## Overall

BERT should ideally outperform:

```text
CRF Test Micro F1 = 0.7934
```

---

## Descriptions

CRF:

```text
Title F1       0.8189
Description F1 0.7703
```

BERT may provide larger gains on descriptions because contextual information is more important.

---

## Product Type

CRF:

```text
PRODUCT_TYPE
F1 = 0.6545
Support = 626
```

This should be a major Transformer target.

---

## Material

CRF:

```text
MATERIAL
F1 = 0.6361
Support = 181
```

---

## Difficult Domains

CRF:

```text
Office Products
F1 = 0.6700

Grocery And Gourmet Food
F1 = 0.6224
```

A contextual model may improve these domains.

---

# 68. Recommended BERT Evaluation

Record at minimum:

```text
entity precision
entity recall
Micro F1
Macro F1
token accuracy
```

Also repeat the CRF analyses:

```text
per attribute
per source
per category
```

For meaningful comparison, produce a table such as:

| Metric             |    CRF | BERT |
| ------------------ | -----: | ---: |
| Overall Micro F1   | 0.7934 |  TBD |
| Title F1           | 0.8189 |  TBD |
| Description F1     | 0.7703 |  TBD |
| Product Type F1    | 0.6545 |  TBD |
| Material F1        | 0.6361 |  TBD |
| Office Products F1 | 0.6700 |  TBD |
| Grocery F1         | 0.6224 |  TBD |

---

# 69. BERT Error Analysis

Reuse the same conceptual taxonomy:

```text
complete miss
boundary error
attribute confusion
spurious prediction
```

This will allow direct statements such as:

```text
BERT reduced PRODUCT_TYPE boundary errors by X%.

BERT reduced complete misses on descriptions by Y%.

BERT improved Grocery category F1 by Z points.
```

This is much more valuable academically than only comparing one aggregate metric.

---

# 70. DeBERTa Stage

After BERT is complete, train a stronger contextual baseline.

Recommended candidate:

```text
DeBERTa-v3-base
```

Do not start DeBERTa until the BERT pipeline is stable.

The benefit of this sequence is that:

```text
data loading
label mapping
WordPiece/subword alignment
metrics
prediction export
error analysis
```

can all be reused.

---

# 71. Normalization Stage

After establishing extraction quality, implement deterministic or structured normalization.

Potential attribute classes:

```text
storage/capacity
dimensions
weights
page yield
processor frequency
rotational speed
pack quantity
```

Example:

```text
8192 MB
→ 8 GB
```

Example:

```text
13.6"
→ 13.6 inch
```

Example:

```text
10,000 RPM
→ 10K RPM
```

The normalization component should be evaluated separately from extraction where possible.

---

# 72. Hybrid System

Planned final architecture:

```text
Product title/description
        ↓
Transformer extractor
        ↓
attribute + raw value
        ↓
attribute-aware normalization rules
        ↓
normalized structured output
```

Example:

```text
Input:
"8 GB RAM"

Extractor:
RAM = "8 GB"

Normalizer:
RAM = "8 GB"
```

Another:

```text
Input:
8192 MB memory

Extractor:
RAM = 8192 MB

Normalizer:
RAM = 8 GB
```

---

# 73. MAVE Stage

MAVE should be introduced after the WDC-PAVE Transformer pipeline is functional.

Recommended sequence:

```text
Obtain required Amazon metadata
        ↓
Reconstruct product text
        ↓
Join MAVE labels
        ↓
validate spans
        ↓
decide manageable experimental subset
        ↓
BIO construction
        ↓
CRF baseline
        ↓
BERT / DeBERTa
        ↓
frequent-vs-long-tail comparison
```

Do not attempt all of MAVE immediately unless the compute/storage/time budget clearly permits it.

A controlled and reproducible subset is preferable to an unfinished full-scale experiment.

---

# 74. Long-Tail Evaluation

For each dataset, attribute frequencies should be computed using positive training spans.

One potential grouping:

```text
Head:
>100 positive instances

Medium:
20–100

Tail:
5–19

Extreme tail:
<5
```

Exact thresholds can be adjusted before the final experiment but must then be frozen.

Compare:

```text
CRF
BERT
DeBERTa
```

within each group.

This directly addresses RQ3.

---

# 75. Demo Requirement

The course eventually requires a working demonstration.

Recommended interface:

```text
CLI first
GUI only if time permits
```

Example CLI:

```text
Enter category:
Computers And Accessories

Enter product title:
HP 400GB 10K SAS Hard Drive

Enter description:
...

Predicted attributes:

BRAND        HP
CAPACITY     400GB
INTERFACE    SAS
PRODUCT_TYPE Hard Drive
```

Later:

```text
Normalized:
CAPACITY = 400 GB
```

A CLI is sufficient to demonstrate the complete inference pipeline reliably.

---

# 76. Reproducibility Requirements

Before final submission, record:

```text
Python version
PyTorch version
Transformers version
CUDA/PyTorch CUDA version
GPU model
random seed
training batch size
learning rate
epochs
model checkpoint name
max sequence length
best epoch
training time
```

These have not yet been fully captured for BERT because Transformer training has not started.

Do not fabricate these values later.

Record them directly from the final environment.

---

# 77. Recommended Experiment Logging

For every Transformer run, save a machine-readable record containing:

```text
run ID
date
model
seed
learning rate
batch size
weight decay
epochs
max length
precision
recall
micro F1
macro F1
best epoch
checkpoint path
```

Example:

```text
results/
├── bert_run_001_metrics.json
├── bert_run_001_predictions.jsonl
├── bert_run_002_metrics.json
└── ...
```

This avoids losing experiment history.

---

# 78. Avoid Test Leakage

The official WDC-PAVE test results for the CRF are final.

For BERT:

```text
train → 906
validation → 160
```

Use only these for model development.

Only after the entire BERT setup is frozen should:

```text
train + validation
```

be used to retrain, followed by official test evaluation.

Do not repeatedly change BERT settings in response to official test performance.

The same rule applies to DeBERTa.

---

# 79. Key Experimental Benchmark

The single number that the next developer should remember is:

# 0.7934

This is the official WDC-PAVE test Micro-F1 for the CRF.

Any claim that a modern model improves on the classical baseline should be evaluated against this value using the same processed gold data.

---

# 80. Main CRF Diagnostic Benchmarks

Additional values worth preserving:

```text
Overall test:
Micro F1       0.7934

Title:
F1             0.8189

Description:
F1             0.7703

Computers:
F1             0.8847

Jewelry:
F1             0.9302

Home/Garden:
F1             0.7402

Office:
F1             0.6700

Grocery:
F1             0.6224

PRODUCT_TYPE:
F1             0.6545

MATERIAL:
F1             0.6361
```

These are the major comparison points for BERT.

---

# 81. Known Limitations

## Flat BIO Cannot Represent Overlapping Attributes

A single token cannot simultaneously be:

```text
B-BRAND
and
B-MANUFACTURER
```

This causes unavoidable information loss in overlapping cases.

---

## Non-Contiguous Values

BIO cannot directly represent values such as:

```text
Ultra320 ... SCSI
```

as one contiguous entity.

---

## Repeated Identical Mentions

The annotation indicates values and source but does not necessarily identify the one unique textual occurrence intended for training.

A representative occurrence is selected heuristically.

---

## Exact Span Evaluation

Predictions such as:

```text
Gold:
Solid State Drive ( SSD )

Pred:
Solid State Drive
```

are considered incorrect under strict entity evaluation despite semantic similarity.

This should remain the primary evaluation because changing it after seeing results would compromise comparability.

---

## Rare Attributes

Some attributes have extremely limited positive training data.

F1 values for these attributes can therefore vary dramatically based on only one or two predictions.

---

# 82. Risks Going Forward

## GPU Availability

The course GPU queue may delay interactive jobs.

Use SLURM responsibly.

---

## Package Compatibility

Blackwell GPU support depends on an appropriate PyTorch/CUDA build.

Always verify:

```text
torch.cuda.is_available()
```

before training.

---

## Hugging Face Download Access

The first model load requires access to the BERT tokenizer/model files unless they are already cached.

---

## Scratch Retention

Important checkpoints may disappear if left inactive in scratch for too long.

Copy final models/results somewhere persistent before project completion.

---

## MAVE Complexity

MAVE reconstruction is substantially more involved than WDC-PAVE and should not interrupt the working WDC-PAVE pipeline prematurely.

---

# 83. Immediate Next Actions for the New Developer

Once GPU resources are available:

## Step 1

Enter allocated node.

Verify:

```bash
hostname
nvidia-smi
```

---

## Step 2

Load environment.

```bash
module load anaconda3-2022.05-gcc-11.2.0-od5lltp
conda activate nlp_project
cd ~/nlp_project
```

---

## Step 3

Verify PyTorch/CUDA.

```bash
python - <<'PY'
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_bf16_supported())
PY
```

---

## Step 4

Install missing Transformer dependencies if necessary.

Required:

```text
torch
transformers
datasets
accelerate
safetensors
seqeval
```

Do not reinstall packages unnecessarily if the environment already contains compatible versions.

---

## Step 5

Run:

```bash
python scripts/check_bert_data.py
```

Expected checks:

```text
Attributes              51
Classifier labels       103
Unknown train labels    {}
Unknown val labels      {}
Unknown test labels     {}
```

Also inspect BERT token lengths.

---

## Step 6

Confirm whether any sequence exceeds BERT's usable maximum.

If none exceeds the selected maximum, proceed directly.

If sequences exceed the selected maximum, do not silently truncate gold entities.

Implement chunking or another explicit strategy.

---

## Step 7

Implement:

```text
scripts/train_bert.py
```

Initial configuration:

```text
bert-base-uncased
LR = 2e-5
batch = 16
epochs ≤ 5
weight decay = 0.01
seed = 42
```

Train on 906 products only.

Evaluate on 160 validation products.

---

## Step 8

Save:

```text
best checkpoint
validation metrics
validation predictions
per-attribute report
source/category results
```

---

## Step 9

Compare against:

```text
CRF validation F1 = 0.7894
```

Analyze where BERT gains or loses.

---

## Step 10

Tune only if necessary.

Potential parameters:

```text
learning rate
batch size
epochs
weight decay
```

Avoid an unnecessarily large search because the dataset is small.

---

## Step 11

Freeze BERT configuration.

Combine:

```text
906 + 160
```

and retrain.

Evaluate official test once.

---

## Step 12

Compare final BERT test score against:

```text
CRF = 0.7934
```

---

# 84. Recommended Subsequent Roadmap

```text
CURRENT
│
├── WDC-PAVE preprocessing ................. COMPLETE
├── CRF baseline ........................... COMPLETE
│
├── BERT setup ............................. IN PROGRESS
│   ├── CUDA verification
│   ├── label schema
│   ├── WordPiece alignment
│   ├── validation training
│   ├── error analysis
│   └── official test
│
├── DeBERTa
│   ├── reuse BERT pipeline
│   ├── validation
│   └── test
│
├── Normalization
│
├── Hybrid model
│
├── MAVE
│   ├── reconstruction
│   ├── subset/split
│   ├── CRF
│   ├── Transformer
│   └── long-tail analysis
│
├── CLI
│
├── Final report
└── Presentation / viva
```

---

# 85. Suggested Final Paper Results Table

Ultimately target a table such as:

| Dataset  | Model       | Precision | Recall |   Micro F1 | Macro F1 |
| -------- | ----------- | --------: | -----: | ---------: | -------: |
| WDC-PAVE | CRF         |    0.8408 | 0.7510 | **0.7934** |   0.7483 |
| WDC-PAVE | BERT        |       TBD |    TBD |        TBD |      TBD |
| WDC-PAVE | DeBERTa     |       TBD |    TBD |        TBD |      TBD |
| WDC-PAVE | Hybrid      |       TBD |    TBD |        TBD |      TBD |
| MAVE     | CRF         |       TBD |    TBD |        TBD |      TBD |
| MAVE     | Transformer |       TBD |    TBD |        TBD |      TBD |

Additional tables should report:

```text
source performance
category performance
head/tail performance
per-attribute performance
error taxonomy
```

---

# 86. Suggested Research Narrative

The current evidence supports the following narrative:

> A classical CRF can extract highly structured e-commerce attributes effectively using lexical, orthographic and local-context features, achieving 0.7934 Micro-F1 on the WDC-PAVE official test set. However, performance varies considerably by attribute and product domain. Structured properties such as UPCs, model identifiers and rotational speeds are extracted accurately, whereas semantically dependent attributes such as product type, material and flavor remain challenging. Error analysis further reveals complete misses, entity-boundary errors and confusion between related attributes such as brand/manufacturer and length/width. These findings motivate the use of pretrained contextual Transformer representations and an attribute-aware normalization layer.

This should remain the backbone of the final report unless later experiments provide contrary evidence.

---

# 87. Handover Checkpoint

The incoming developer should consider the project healthy if the following files exist:

```text
/scratch/csisnlp_09/data/wdc_pave/processed/crf/train.jsonl

/scratch/csisnlp_09/data/wdc_pave/processed/crf/val.jsonl

/scratch/csisnlp_09/data/wdc_pave/processed/crf/test.jsonl

/scratch/csisnlp_09/models/wdc_pave_crf_final.pkl

~/nlp_project/results/crf_test_results.txt

~/nlp_project/results/crf_test_predictions.jsonl
```

Validation:

```bash
python scripts/validate_bio_dataset.py
```

should continue reporting:

```text
Length errors = 0
BIO errors    = 0
```

---

# 88. Final Handover Summary

At handover:

```text
DATASET ENGINEERING
===================
WDC-PAVE inspected             ✓
Train/val/test split           ✓
Span recovery analysis        ✓
BIO generation                ✓
BIO validation                ✓
Overlap handling              ✓


CLASSICAL NLP
=============
CRF implementation            ✓
Feature engineering           ✓
Validation                    ✓
Hyperparameter search         ✓
Error analysis                ✓
Final test evaluation         ✓

Official CRF Micro F1:
0.7934


TRANSFORMERS
============
BERT selected                 ✓
Experimental design           ✓
BIO/WordPiece plan            ✓
GPU allocation                pending
BERT training                 not started


SECOND DATASET
==============
MAVE identified               ✓
MAVE reconstruction           pending
MAVE CRF                      pending
MAVE Transformer              pending


FINAL SYSTEM
============
Normalization                 pending
Hybrid model                  pending
Demo                          pending
Final paper                   pending
```

The immediate priority is therefore:

# Establish the BERT baseline on WDC-PAVE without modifying the existing gold annotations or CRF benchmark.

Once BERT is stable, the same architecture, evaluation and error-analysis framework should be reused for DeBERTa and subsequently for the reconstructed MAVE dataset.

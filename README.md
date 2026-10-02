Business Entity Resolution — blocking pipeline
This repository holds our solution for the ML Challenge 2026 Business Entity Resolution task. The blocking / candidate-generation stage lives in src/business_entity_resolution/; the design, the EDA evidence behind every decision and the tunable defaults are in BLOCKING_PLAN.md. The original challenge statement is kept below for reference.

Quick start (AWS or any Linux box with uv)
# 1. environment (python 3.12 managed by uv, locked dependencies)
make setup                      # = uv venv --python 3.12 && uv sync --frozen --extra dev
# 2. train split: prepare -> learn aliases -> block -> evaluate against the ground truth
make block-train                # artifacts/train/{normalized_*.pkl, aliases.pkl, candidates.pkl, blocking_metrics.json, ...}
# 3. test split: prepare -> block -> export
make block-test                 # artifacts/test/candidates.pkl + output/candidate_pairs.tsv
make validate                   # utils/validate_submission.py --check-ids on output/
Without GNU make: ./run.sh setup, ./run.sh train, ./run.sh test, ./run.sh validate. Docker: docker build -t ber-blocking . then docker run --rm -v $PWD/dataset:/app/dataset -v $PWD/artifacts:/app/artifacts -v $PWD/output:/app/output ber-blocking --split both.

The single runner is python -m business_entity_resolution.blocking_main (also uv run blocking.py):

--split train|test|both      --stages prepare,aliases,block,evaluate,export   --resume (skip stages whose artifacts exist)
--sample N --pool-fraction F (development runs on a subset; true matches of the sampled S1 entities stay in the pool)
--threads N (sparse matmul)  --workers N (normalisation processes)  --cap N  --k-word/--k-char-name/--k-char-addr N
--no-shard (disable state sharding)  --dense (optional embedding blocker B4, needs `uv sync --extra dense`)
--data-dir/--artifacts-dir/--output-dir   --parquet (also write candidates.parquet)
Every stage prints a banner and progress bars (readable in a log: nohup make all > run.log 2>&1 &). Run the train split before the test split: the test run reuses artifacts/train/aliases.pkl (state/city alias tables learned from the training ground truth).

What the pipeline does
prepare — reads the three TSVs (tab-separated, no quoting), normalises names and addresses (normalize.py: unidecode + lowercase + junk/web-tail removal, honorific and legal-suffix stripping with a suffix vocabulary learned from the split itself, address components, ordinal/abbreviation/number normalisation), flags the name script (latin / indic / other), writes normalized_{source1,source2,source3}.pkl and, for train, splits.pkl (ml_train / ml_valid, 90/10 stratified by country, seed 42).
aliases (train) — learns per-country state vocabularies and component -> state alias tables from the ground-truth pairs (aliases.pkl; e.g. tx <- texas, houston, dallas, maharashtra <- mh, mumbai, mhaaraassttr).
block — per country: B0 exact core-name key, B1 word TF-IDF top-K over name+address (state-sharded), B2 char-3gram TF-IDF top-K over the name, B3 char-3gram TF-IDF top-K over the address, optional B4 dense embeddings; weighted Reciprocal Rank Fusion; cap per S1 entity. Writes candidates.pkl (one row per candidate pair with provenance: per-blocker rank/cosine, n_blockers, rrf_score, rrf_rank, state_match, and label on train), vectorizers.pkl, final_addr_*.pkl, block_summary.json.
evaluate (train) — recall against the ground truth: pair / entity / full-recall, by country, by name script, by fold, per blocker (unique contribution, recall without it), recall at caps, candidates per S1, reduction ratio, F0.5 ceiling for a perfect matcher; writes blocking_metrics.json and missed_pairs.pkl.
export (test) — output/candidate_pairs.tsv (one row per test S1 entity) and an empty output/matching_results_stub.tsv so the official validator can run. The real matching_results.tsv comes from the matching model, which trains on artifacts/train/candidates.pkl (label column) and scores artifacts/test/candidates.pkl.
All artifacts are pickles (pd.read_pickle(...) gives pandas DataFrames with pyarrow-backed columns) so they can be moved between machines / S3. ARTIFACTS.md documents every file and the protocol for the matching model. GPU: make setup-gpu (torch cu128) and add --gpu; the char 3-gram passes then run on CUDA (4-5x faster). EDA scripts that produced the evidence in BLOCKING_PLAN.md are in scripts/eda/. Tests: make test (uv run pytest).

Matching model: stage-A CatBoost pair classifier (CATBOOST_PLAN.md)
CatBoost (GPU on the laptop, CPU prediction) scores every (S1, candidate) pair with 47 base features (blocking provenance, name / address similarities, per-entity context) plus the label-free "orthogonal" families of ml/features_extra.py (candidate graph, address and name structure, block ambiguity, IDF of shared tokens). Graph features are computed on whole candidate parts and context features on complete entities, so train, ml_valid and test see identical feature definitions. Results and the feature ablations: CATBOOST_RESULTS.md.

make setup-gbm                                    # + catboost, jellyfish
make gbm-mini                                     # the whole chain on dev/mini_artifacts, validator included
make gbm-full                                     # full data: features -> fit -> predict valid/test -> report -> decide -> safety submission
#   gbm-features build  train/ml/gbm_fit.parquet  (50% of ml_train entities, row filter + weighted tail; resumable per part)
#   gbm fit             gbm_model_{42,43,44}.cbm  (quantised in RAM or from a TSV when RAM is short: --loader auto|numpy|file)
#   gbm predict         gbm_valid.parquet, gbm_rule.json, gbm_test.parquet, gbm_entity_{valid,test}.parquet (uncertainty order)
#   gbm report          gbm_metrics.json: F0.5 overall/country/Indic/hard, singletons, calibration, loss split, drift, France check
make feat-explore FAMILIES=explore BASE=measured  # feature-family ablation harness on mini (3 seeds, paired bootstrap)
output/matching_results.tsv from CatBoost alone (make gbm-safety) is the safety submission; stage B then re-scores only the most uncertain test entities (--entity-list ... --top-k, SAGEMAKER_GUIDE.md section 3).

Matching model: stage-B LLM block re-ranker (GEMMA_PLAN.md)
Gemma 4 E4B (text tower only, LoRA + a 1-logit answer head) reads one block per S1 entity (the S1 record and its top 12-16 candidates) and scores every candidate in one forward pass; stage C turns the probabilities into per-entity sets. One runner, five commands (python -m business_entity_resolution.ml ...):

make setup-llm                                                        # + transformers / peft / bitsandbytes
uv run python -m business_entity_resolution.ml blocks --artifacts-dir artifacts --n-train 400000   # WP-B, CPU
# training / scoring run on SageMaker (SAGEMAKER_GUIDE.md): any GPU instance, dataset size chosen at launch
uv run --no-project --with "sagemaker>=3.20,<4" python aws/sagemaker_launch.py train --bucket B --run-name night1 --time-budget-hours 7
uv run python -m business_entity_resolution.ml decide --artifacts-dir artifacts   # rule on valid_llm, stage A vs A+B
uv run python -m business_entity_resolution.ml export --artifacts-dir artifacts --test-dir dataset/test
Development on the mini copies: make ml-mini, make llm-train-local (4-bit E2B), make llm-score-local. The blocks' prompts carry the provenance-only out-of-fold proxy p the LLM was trained with; decisions, routing and the stage-C blend use CatBoost's ml/gbm_*.parquet when present (else the proxy, with a warning).

ML Challenge 2026 Problem Statement
Business Entity Resolution Challenge
In large-scale commercial platforms, business identity data arrives from multiple independent sources — each contributing partial, noisy fragments of information about the same real-world entities. These fragments share no common identifiers, and the challenge of determining which records refer to the same business is known as Entity Resolution (ER). Your challenge is to build an ML solution that, given business records from 3 independent data sources with noisy and inconsistent fields, determines which records across sources refer to the same real-world business entity.

Source 1 is the deduplicated reference source. Your task is to find all matching records from Source 2 and Source 3 for each Source 1 entity. A Source 1 entity may match zero, one, or many records from Source 2 and Source 3.

File Format
All files in this challenge are tab-separated (.tsv), and your submissions must be tab-separated too. Tabs are used because business addresses and the ID list columns both contain commas. Read them with an explicit tab separator, for example:

import pandas as pd
df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
Reading a .tsv without sep="\t" will silently produce a single column containing the whole line.

Data Description:
Each source file (*_source1.tsv, *_source2.tsv, *_source3.tsv) has the following columns:

entity_id: Unique identifier for the record. The prefix indicates the source — S1-, S2-, or S3-.
business_name: Name of the business entity (may contain abbreviations, legal suffixes, typos, transliterations)
business_address: Address of the business (may contain partial addresses, format variations, missing components, landmark-based references)
country: Country label for the record. The training data covers US and India. The test set additionally contains a third country, France, that does not appear in the training data. Treat country as an open set of string labels: do not hard-code, filter, or one-hot your pipeline to only {US, India}, and remember that every test entity — France included — must appear in your submission.
There is no separate source column — a record's source is given by its entity_id prefix (S1-/S2-/S3-) and by which file it appears in.

The ground truth file (train_ground_truth.tsv) has two columns:

source1_entity_id: The entity_id of a Source 1 record
matched_entity_ids: Comma-separated list of matching entity_ids from Source 2 and/or Source 3 (empty when the entity has no matches)
Noise Patterns to Expect:

Name variations: Abbreviations (Corp vs. Corporation, Pvt vs. Private, Ltd vs. Limited), legal suffix inconsistencies, DBA/trade names, punctuation differences (& vs. "and"), word-order transpositions, typos
Address variations: Abbreviations (Rd vs. Road, St vs. Street), transliteration variants, missing components (no PIN code, no state), landmark-based references (Near SBI ATM), municipal numbering formats, component reordering
Dataset Details:
Training Dataset: Business records across 3 sources with ground truth matching labels
Test Set: Business records across 3 sources without matching labels
File Descriptions:
Training files

dataset/train/train_source1.tsv: Source 1 training records (the deduplicated reference source)
dataset/train/train_source2.tsv: Source 2 training records
dataset/train/train_source3.tsv: Source 3 training records
dataset/train/train_ground_truth.tsv: Ground truth matching labels for the training set
Test files

dataset/test/test_source1.tsv: Source 1 test records. Generate matches for every entity in this file.
dataset/test/test_source2.tsv: Source 2 test records
dataset/test/test_source3.tsv: Source 3 test records
No ground truth is provided for the test set. To measure your own performance, hold out a validation split from the training data and score it yourself using the F_0.5 formula given below.

Output Format:
Your solution produces two tab-separated files, both placed in the output/ folder of your final submission package (see Final Submission Package below):

matching_results.tsv — your final entity matches. This is the only file scored on the leaderboard — it is what you upload to the Portal during the challenge.
candidate_pairs.tsv — the candidate set your blocking / candidate-generation stage produced, before your final matching model narrowed it down.
matching_results.tsv
Your final entity matches:

Column	Description
source1_entity_id	The entity_id of a Source 1 record
matched_entity_ids	Comma-separated list of matching entity_ids from Source 2 and/or Source 3
Example (columns separated by a single tab, ID lists separated by commas with no quoting):

source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
Important:

Every Source 1 entity in the test set must have exactly one row
Leave matched_entity_ids empty for entities with no matches (singletons)
No duplicate entity IDs within a single ID list
ID lists must only contain Source 2 or Source 3 IDs that exist in the test set
candidate_pairs.tsv
The candidate set from your blocking stage — every Source 2 / Source 3 record you considered a plausible match for each Source 1 entity, before your final matching model narrowed it down. This is the exact set of records you feed into your matching model for inference — the final candidate list just before the ML model scores them, not the raw output of an early blocking pass you later filter further. If your pipeline has several blocking/filtering stages, candidate_pairs.tsv is the last one: whatever your model actually runs inference over. Every ID in matching_results.tsv should therefore appear here.

It is not scored on the leaderboard; we use it to analyse blocking quality (recall ceiling, reduction ratio) and to verify your pipeline.

Column	Description
source1_entity_id	The entity_id of a Source 1 record
candidate_entity_ids	Comma-separated list of candidate entity_ids from Source 2 and/or Source 3
Example:

source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003	
Same rules as matching_results.tsv: one row per Source 1 entity, candidate_entity_ids empty when blocking found no candidates, S2-/S3- IDs only, no duplicates within a list. Your final matches should be a subset of your candidates (a matched ID that never appeared as a candidate signals a pipeline bug — the validator warns about it).

Validate before submitting: a helper script utils/validate_submission.py (stdlib only, no dependencies) checks both files against every rule above so you can catch a rejection locally instead of spending a submission on it. Run it from this student_resource/ directory:

python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
It prints PASS (exit 0) when the files are safe to submit, or a numbered list of issues to fix (exit 1). It only reads your output files and the test source files; it does not compute your score.

Final Submission Package:
In addition to your live leaderboard uploads, every team submits a single zip archive with your code and outputs. We use it to reproduce your results, audit your blocking, and check the fair-play and model-license rules — the top teams' packages are reviewed in detail before the final rankings are confirmed.

Structure:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches (same file you upload to the leaderboard)
│   └── candidate_pairs.tsv         # your blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all your source code
│       ├── README.md               # how to reproduce end-to-end (data → blocking → matching → output)
│       └── requirements.txt        # pinned dependencies / environment
└── Documentation_template.md       # your methodology write-up (this filled-in template)
output/ — the two TSV files described above: matching_results.tsv and candidate_pairs.tsv.
code/business_entity_resolution/ — a self-contained, runnable copy of your pipeline. Put all source under src/, and include a README.md with exact run instructions plus a requirements.txt (or equivalent environment file) pinning versions. Anyone should be able to regenerate both output files from the training/test data using only what is in this folder.
Methodology document — fill in the provided Documentation_template.md and drop it straight into the zip (the filled-in .md is fine; a .pdf export works too). No need to rename it.
Constraints:
Format your output exactly as described above. Submissions that fail validation will not be evaluated. You should see a SCORED status with your F_0.5 score if the output is correctly formatted.
matched_entity_ids must only reference entities from Source 2 or Source 3. Self-matches to Source 1, and IDs that do not exist in the test set, will be rejected.
Every Source 1 entity must appear in your submission. Missing entities will cause rejection.
Duplicate entity IDs in any ID list will cause rejection, as will duplicate source1_entity_id rows.
Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters.
Evaluation Criteria:
Submissions are evaluated using F_β Score (β = 0.5) — a precision-heavy metric that penalizes false merges (matching two different businesses) more than missed matches.

Formula:

F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
Computed as a macro-average: F_0.5 is calculated per Source 1 entity, then averaged across all Source 1 entities in the evaluation set.

Singletons are included in that average. A Source 1 entity with no true matches scores 1.0 when you correctly predict an empty list, and 0.0 when you predict any match for it. Correctly identifying singletons therefore earns credit, and false merges on them are penalised.

Why precision-heavy? In real-world entity resolution, merging two distinct businesses (false positive) is more damaging than missing a link (false negative). F_0.5 weights precision 2× over recall.

Example:

Your model predicts S1-00001 matches [S2-00047, S2-00193, S3-00812]
Ground truth says S1-00001 matches [S2-00047, S3-00812]
Precision = 2/3, Recall = 2/2 = 1.0
F_0.5 = (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) = 0.714
Leaderboard Information:
Public Leaderboard: During the challenge, rankings will be based on a subset of the test set to provide real-time feedback on your model's performance.
Private Leaderboard: After the challenge ends, the private leaderboard will be revealed, which uses the remaining portion of the test set for evaluation.
Final Rankings: The final decision will be based on the private leaderboard.
You submit predictions for the full test set in both cases; the split is applied during scoring.

Submission Requirements:
Leaderboard (during the challenge): upload matching_results.tsv in the Portal — tab-separated, with the exact column names described above. This is what drives the public and private leaderboards.

Final submission package: submit the single zip described in Final Submission Package above — output/ with both matching_results.tsv (final matches) and candidate_pairs.tsv (your candidate-generation / blocking set fed to the model), code/business_entity_resolution/ (runnable pipeline), and your methodology document. All teams must submit it; the top teams' packages are reviewed before the final rankings are confirmed.

Your methodology document must describe:

Methodology used
Candidate generation / blocking strategy
Model architecture and feature engineering
Any other relevant information about the approach
A template for this documentation is provided in Documentation_template.md. There is no page limit — prioritise clarity and technical depth over brevity.

Academic Integrity and Fair Play:
⚠️ STRICTLY PROHIBITED: External Data Lookup

Participants are STRICTLY NOT ALLOWED to use external databases, APIs, or services to look up business identities or resolve entities. This includes but is not limited to:

Using commercial entity resolution APIs or services
Looking up business registrations from government databases
Using geocoding APIs to normalize addresses
Any external data augmentation from internet sources
Enforcement:

All submitted approaches, methodologies, and code pipelines will be thoroughly reviewed and verified
Any evidence of external data lookup will result in immediate disqualification
Fair Play: This challenge is designed to test your machine learning and data science skills using only the provided training data.

Tips for Success:
Invest in a strong blocking/candidate generation strategy — it determines the upper bound of your recall
Explore string similarity features (Jaccard, Levenshtein, TF-IDF cosine) for name and address matching
Pay attention to country specific address patterns
Consider the precision-recall trade-off carefully — F_0.5 rewards precision more than recall
Do not neglect singletons — correctly predicting "no match" is worth a full 1.0 on that entity
Validate your own output format against the rules above before submitting

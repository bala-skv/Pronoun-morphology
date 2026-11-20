#!/usr/bin/env bash
# Run every stage in order on one machine (a GPU is strongly recommended).
# Env: PRONOUN_DEVICE=cuda|cpu, PRONOUN_NDEVICES (GPUs to split the model over), GEMMA_REPO (default: auto)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
NP=${NPAIRS:-200}

$PY scripts/00_behaviour.py
for spec in "c1 hi" "c1 pa" "c2 hi"; do
  set -- $spec
  $PY scripts/01_lens_dla.py --contrast "$1" --lang "$2"
done
$PY scripts/02_patching.py --contrast c1 --lang hi --n-pairs "$NP"
$PY scripts/02_patching.py --contrast c1 --lang pa --n-pairs "$NP"
$PY scripts/02_patching.py --contrast c2 --lang hi --n-pairs 40 --keep distinguishes
$PY scripts/05_attention.py --lang hi --n 304
$PY scripts/05_attention.py --lang pa --n 304
$PY scripts/03_probe_gender.py
$PY scripts/03_probe_case.py --max-sents 6000
$PY scripts/04_sae.py --layers 4,8,12,16,18,20,21,22,23,24
$PY scripts/00b_orthography.py
$PY scripts/06_figures.py
$PY scripts/99_summarize.py

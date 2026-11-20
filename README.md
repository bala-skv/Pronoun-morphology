# Where Does a Model Decide "Mine"? Pronoun Morphology in Gemma-2-2B

Course project for CL3.412 (Mechanistic Interpretability of Language Models), IIIT Hyderabad.
Amisha (2023114003) and Balasubramanian (2025121002).

In Hindi and Punjabi the possessive "my" agrees in gender with the possessed noun (मेरी किताब "my book",
मेरा घर "my house"); English and Tamil use one form. We look at where Gemma-2-2B makes this choice and
whether Hindi and Punjabi, which share the rule but not the script, use the same mechanism. We also look at
the Hindi ergative (मैंने vs. मैं), which changes the form of the following verb.

The report is [`report.pdf`](report.pdf).

## Main results

| | Hindi | Punjabi |
|---|---|---|
| Agreement accuracy, with two example sentences | 0.905 | 0.793 |
| Layer where the gender gap appears (logit lens) | 21 | 21 |
| Patching: final position takes over at layer | 22 | 22 |
| Recovery from head L21H4 alone | 0.86 | 0.81 |
| Recovery from MLP 21 alone | 0.60 | 0.54 |
| Gender gap removed by editing 10 Hindi SAE features at layer 21 | 175% | 147% |

- The noun keeps its gender until layer 21. Head L21H4 copies it to the final position and MLPs 21-23 write
  the output. Hindi and Punjabi use the same head, layers and SAE features.
- Before the native word, the logit lens shows English "my"/"mine" (probability up to 0.99 at layers 20-23).
- The Hindi ergative is decided at a similar depth but by several heads and MLPs, with no single dominant head.
- Probes find gender from layer 0, but control tasks are as easy and a Hindi probe also transfers to English
  and Tamil, so the probing results alone are not informative about use.

`RESULTS.md` has the full numbers for every stage.

## Repository layout

```
src/pronoun/        model loading, prompts and tokenisation, metrics, patching, probing, SAE loader
scripts/            one script per stage (run in numeric order)
  00_behaviour.py     tokenisation of the forms and behavioural accuracy
  00b_orthography.py  accuracy by whether the noun's ending shows its gender
  01_lens_dla.py      logit lens and direct logit attribution
  02_patching.py      activation and attribution patching (residual, attention/MLP, heads)
  03_probe_gender.py  gender probes on the templates, control tasks, cross-lingual transfer
  03_probe_case.py    pronoun case probes on Universal Dependencies treebanks
  04_sae.py           Gemma Scope SAE features: sharing across languages and steering
  05_attention.py     attention patterns of the top heads
  06_figures.py       figures used in the report (written to results/figures/)
  99_summarize.py     collects all stage outputs into RESULTS.md
data/               lexicon, nouns with genders, ergative object-verb rows, UD treebanks
results/            outputs of every stage (CSV and PNG), from the GPU run
run_all.sh          runs all stages on a single machine
```

## Setup

Python 3.11 or 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch            # pick the CUDA build that matches your GPU
pip install -r requirements.txt
```

We use TransformerLens 2.18 (`HookedTransformer` was removed in later versions), which needs numpy < 2.
Gemma Scope SAEs are loaded directly from Hugging Face (`src/pronoun/sae.py`), so `sae_lens` is not needed.
`google/gemma-2-2b` is gated on Hugging Face; if the licence has not been accepted, the code loads the same
weights from `unsloth/gemma-2-2b`. Set `GEMMA_REPO` to choose explicitly.

Environment variables: `PRONOUN_DEVICE` (`cuda` or `cpu`), `PRONOUN_DTYPE` (default `float32`),
`PRONOUN_NDEVICES` (split the model over several GPUs).

## Running

```bash
bash run_all.sh
```

All stages together take a few hours on two 11 GB GPUs. Stage 0, the logit lens and patching with fewer
pairs (`--n-pairs 24`) also run on a CPU. Each script writes to `results/<stage>/`; `06_figures.py` and
`99_summarize.py` only read those outputs.

## Notes on the setup

- Prompts include two example sentences (one feminine, one masculine possessive) before the test noun.
  With bare prompts the model rarely predicts a possessive and agreement is at chance.
- Hindi मेरी/मेरा and Punjabi ਮੇਰੀ/ਮੇਰਾ are several tokens long. We append the shared prefix to the prompt
  and score the logit difference at the first token where the two forms differ.
- Ergative prompts start with "हाँ," so that मैंने and मैं are both single tokens and clean and corrupted
  prompts align for patching.
- Recovery is computed from mean logit differences over pairs (ratio of means).
- The forms, nouns, genders, verb forms and frames were checked by native speakers of Hindi, Punjabi and
  Tamil. Punjabi ਕਮੀਜ਼ takes either gender and is excluded from Punjabi scoring. The Punjabi probing, lens,
  patching and SAE outputs in `results/` were produced with ਅਖ਼ਬਾਰ labelled feminine and ਕਮੀਜ਼ included
  (2 of 76 nouns); the behavioural results use the final labels.

## Data

`data/ud/` contains Universal Dependencies treebanks (English GUM, Hindi HDTB, Tamil TTB, Punjabi), each
under its own licence; see https://universaldependencies.org. The templates in `data/` are our own.

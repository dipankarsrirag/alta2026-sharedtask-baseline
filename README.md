# ALTA 2026 Shared Task Baseline — BESSTIE Sentiment & Sarcasm Classification

Baseline system for the ALTA 2026 shared task on sentiment and sarcasm
classification for Australian (en-AU) and British (en-UK) English, using
the en-AU and en-UK subsets of [BESSTIE](https://aclanthology.org/2025.findings-acl.432/).

Full task description, rules, and timeline: **[ALTA 2026 Shared Task page](http://www.alta.asn.au/events/sharedtask2026/)**

This baseline is provided as a starting point only. Participants are expected
to beat it — it is not competitive with the systems we expect to see submitted.

## Task

Given a text (Google Places review or Reddit post/comment) labelled with its
`source` and `variety`, predict:

- `sentiment`: negative (`0`) or positive (`1`)
- `sarcasm`: not sarcastic (`0`) or sarcastic (`1`)

Systems are scored with macro-F1 per task and per dialect (`sent-AU`,
`sent-UK`, `sarc-AU`, `sarc-UK`). To reward dialect robustness, each task is
scored by its **weakest**-performing variety:

```
score = ( min(sent-AU, sent-UK) + min(sarc-AU, sarc-UK) ) / 2
```

See the shared task page for the full evaluation description and participation rules.

## Approach

Both scripts fine-tune a pretrained transformer encoder independently for
the sentiment and sarcasm tasks, using class-balanced cross-entropy loss to
handle the sarcasm class imbalance (~19% positive in training data). Only
open-source models/weights are used, consistent with the task's
participation rules.

- `train_baseline.py` — default model `distilbert-base-uncased`. The
  official baseline for this task; scores below are from this script.
- `train_transformer.py` — same training code, default model `roberta-base`.
  A stronger but slower alternative.

Both accept `--model_name` to swap in any other Hugging Face encoder.

## Data

Download `alta2026_public_data.zip` from the shared task page and unzip it
into a `data/` folder at the repo root:

```
data/
  train.csv
  valid.csv
  evaluate.py
  sample_valid.csv
```

`valid.csv` is the public dev/test set — the one used to produce
`answer.csv` submissions during development. It is **not** used for final
ranking.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python train_baseline.py --data_dir data --out answer.csv
# or, for the roberta-base variant:
python train_transformer.py --data_dir data --out answer.csv
```

This trains both task heads, writes predictions for `data/valid.csv` to
`answer.csv`, and prints per-dialect macro-F1 plus the final score.

To submit, zip the output:

```bash
zip -j answer.zip answer.csv
```

You can also score any `answer.csv` directly against the official script:

```bash
mkdir -p eval/res eval/ref
cp answer.csv eval/res/answer.csv
cp data/valid.csv eval/ref/truth.csv
python data/evaluate.py eval eval/output
cat eval/output/scores.txt
```

## Baseline results

`train_baseline.py` (`distilbert-base-uncased`) scores **0.7740** final
score on the public `valid.csv` set (dev-only, not used for final ranking):

| sent-AU | sent-UK | sarc-AU | sarc-UK | **Final score** |
|---|---|---|---|---|
| 0.9106 | 0.9532 | 0.7135 | 0.6374 | **0.7740** |

`train_transformer.py` (`roberta-base`) is provided as a stronger but
slower alternative; participants are encouraged to benchmark it themselves.

## License

MIT — see [LICENSE](LICENSE).

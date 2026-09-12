# Budget Transaction Categorizer — Fine-Tuned Qwen3-1.7B

A LoRA fine-tuned version of Qwen3-1.7B-Base that classifies raw bank/card transaction line items (e.g. `SQ *BLUE BOTTLE COFFEE DENVER CO`) into 7 personal budget categories, compared against an untuned baseline.

This started as the Gen AI Academy Week 5 project ("Fine-Tune a Support Ticket Router"), adapted from IT ticket routing to personal finance transaction categorization — same LoRA fine-tuning pipeline (LLaMA Factory), different, more universally relatable dataset.

## The problem

Budgeting apps (Mint, YNAB, Copilot, your bank's own app) need to categorize every transaction a user makes, continuously, at scale. Calling a frontier LLM API per transaction doesn't hold up for three reasons:

- **Cost** — a fine-tuned small model runs on a single CPU instance for a few dollars a month; per-call API pricing doesn't scale to millions of transactions/day.
- **Latency** — categorization needs to happen in near-real-time as transactions post.
- **Privacy** — transaction data is sensitive financial data; routing it through a third-party API is a real compliance concern for any fintech, independent of cost.

## Categories

| Category | Examples |
|---|---|
| Groceries | Supermarkets, grocery delivery, warehouse clubs |
| Dining & Coffee | Restaurants, cafes, fast food, food delivery apps |
| Transportation | Rideshare, gas, tolls, parking, transit |
| Subscriptions & Entertainment | Streaming, gym memberships, app subscriptions |
| Utilities & Bills | Electric, water, internet, phone, insurance |
| Shopping & Retail | Online/in-store retail, electronics, clothing |
| Health & Wellness | Pharmacy, doctor copay, dental, vision |

## Dataset

`data/budget_transactions.csv` — 570 synthetic but realistic transaction line items (`category_truth,text`), styled after real bank statement formatting (merchant codes, POS/ACH prefixes, city/state, reference numbers), stratified 80/20 into train/validation (456 train / 114 validation).

## Pipeline

1. Install LLaMA-Factory, confirm T4 GPU
2. Prepare dataset — stratified split, ShareGPT JSON conversion, registered as `budget_transactions`
3. LoRA fine-tune `Qwen/Qwen3-1.7B-Base` — 10 epochs, learning rate 1e-4, run directly via the `llamafactory-cli train` CLI (the LLaMA Board Gradio UI hung on a multiprocessing deadlock in this environment, so the CLI was used instead — same underlying training code and config)
4. Review training loss curve
5. Merge LoRA adapter into base weights via `llamafactory-cli export`, smoke test on 5 hand-picked examples (one per category)
6. Evaluate on held-out validation split; compare against an untrained baseline using the same prompt format

## Results

**Loss curve:** converged cleanly over 290 steps (10 epochs) — starting loss 7.19, final loss 2.62, a total drop of 4.56. Per-step loss plateaued around 2.5–2.6 in the final epochs. See `results/loss_curve.png`.

**Classification report (fine-tuned model, full validation split, n=114):**
```
=== Fine-tuned model — validation set classification report ===

                               precision    recall  f1-score   support

                    Groceries      1.000     1.000     1.000        20
              Dining & Coffee      1.000     1.000     1.000        18
               Transportation      1.000     1.000     1.000        12
Subscriptions & Entertainment      1.000     1.000     1.000        18
            Utilities & Bills      1.000     1.000     1.000        16
            Shopping & Retail      1.000     1.000     1.000        16
            Health & Wellness      1.000     1.000     1.000        14

                     accuracy                          1.000       114
                    macro avg      1.000     1.000     1.000       114
                 weighted avg      1.000     1.000     1.000       114
```

**Confusion matrix:** see `results/confusion_matrix.png` — a clean diagonal, 100% recall on every category, no confusion pairs at all. Two categories I expected to be genuinely ambiguous going in (Shopping & Retail vs. Groceries, e.g. a Costco/Target charge; Subscriptions & Entertainment vs. Utilities & Bills) turned out not to be a problem in practice on this dataset.

**Baseline vs. fine-tuned accuracy:**
| Model | Accuracy |
|---|---|
| Baseline (untrained Qwen3-1.7B-Base, same prompt format) | 26.3% |
| Fine-tuned (LoRA, merged) | 100.0% |
| Delta | +73.7 pts |

See `results/baseline_vs_finetuned.png` for the per-class breakdown — the baseline model collapsed toward a couple of categories (Shopping & Retail, Groceries) while getting others like Subscriptions & Entertainment and Health & Wellness wrong 100% of the time.

## Lesson learned (the interesting bug)

After the first two full training runs, evaluation was still showing every prediction collapsing into `Shopping & Retail`, which looked like a training/undertraining problem — the obvious response is "train longer" or "tune hyperparameters," which is what I tried first (twice). It turned out to be a parsing bug, not a model problem: the base model's raw generated text frequently included a stray leading token (an odd Unicode character, a stray subword) before the actual category name, and the classification helper matched labels with `generated.startswith(label)`, which only matches when the label is literally the first thing generated. Switching that check to `label in generated` (substring match) fixed it immediately, with the *already-trained* model, no retraining needed — validation accuracy went from effectively 0% recall on most classes to 100% in one edit. Worth remembering: when a fine-tuned model's outputs look uniformly wrong in a suspicious pattern (always the same class), check the post-processing/parsing logic before assuming the model needs more training.

## What I'd try next

- Fix the confidence-score calculation, which is a separate, smaller bug: it reads the logits of the first generated token position, which — now that generation sometimes starts with that same stray leading token — occasionally reports a misleadingly low confidence (e.g. 0.0%) on a correct prediction. Purely cosmetic; it doesn't affect the predicted label.
- Constrain generation to a fixed label vocabulary (e.g. logit-biasing to just the 7 category strings, or a small classification head instead of open-ended generation) to make the parsing bug above structurally impossible rather than just fixed in post-processing.
- Add a few more adversarial/ambiguous examples (e.g. a wholesale club purchase that's actually groceries, a subscription billed like a utility) to stress-test whether the clean 100% holds up outside this particular validation split.

## Demo

Loom walkthrough: [link]

## Credits

Base assignment: [Gen AI Academy Week 5 — Fine-Tune a Support Ticket Router](https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router)

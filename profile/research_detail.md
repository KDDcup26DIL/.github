# KDD_submission Experimental Summary

> This document summarizes the `KDD_submission` repository in a reusable form for future CTR and recommender-system competitions. It covers the submission framework, local validation protocol, model variants, experimental logs, failure modes, and empirical lessons.

## 1. Executive Summary

The main objective was to build a single-model submission pipeline compliant with the KDD Cup rules, benchmark representative CTR/RecSys backbones, and improve the HyFormer family through controlled architectural variants.

| Topic | Finding | Implication |
| --- | --- | --- |
| Best model family | `HyFormer tref` and `HyFormer UniMixer` | Stable sequence summaries and user/item global summaries were the most effective signals. |
| Fast baselines | DCNv2 and WuKong | These models are useful for early sanity checks and parallel baseline evaluation. |
| Auxiliary towers | WuKong, GCN, user-init, and large fusion mostly failed | Internal diversity is useful only when the main tower is preserved. |
| Sequence signal | Removing `MeanPool(seq)` reduced submission AUC | Simple sequence summaries were more robust than complex sequence encoders. |
| Sparse regularization | Removing sparse re-initialization caused the largest ablation drop | High-cardinality embedding regularization is essential. |
| Local vs. submission | Local ranking did not fully match public submission ranking | Local validation should be used for filtering, not final model selection. |
| Framework | `run.local.sh` and `run.submit.sh` enabled reproducible experiments | The directory, logging, checkpoint, and packaging pattern is reusable. |

The central empirical lesson is that small modifications inside the strongest HyFormer path were safer than attaching larger external modules. LightGCN, clicked-user initialization, and large fusion appeared plausible but collapsed under submission evaluation.

## 2. Framework Design

### 2.1 Repository Layout

The repository separates local experimentation from submission packaging.

```text
KDD_submission/
├── sample/
│   ├── model_training/
│   └── model_evaluation/
├── {model_name}/
│   ├── {model_name}_local/
│   │   ├── model_training/
│   │   └── model_evaluation/
│   └── {model_name}_submission/
│       ├── {model_name}_training/
│       └── {model_name}_evaluation/
├── data/{dataset_name}/
│   ├── train.parquet
│   ├── valid.parquet
│   ├── test.parquet
│   └── schema.json
├── log/
├── checkpoint/
└── local_runs/
```

The `sample/` directory was treated as immutable. Each model was developed under its own local directory and later copied into the official submission layout. This design reduced accidental contract violations while enabling many parallel variants.

### 2.2 Scripts

| Script | Function | Note |
| --- | --- | --- |
| `make_dir.sh <model>` | Creates local and submission directories | Minimal skeleton generator. |
| `run.local.sh <model> <gpu> <dataset> ...` | Trains on train/valid and evaluates on held-out test | Creates run-scoped logs, checkpoints, and local run artifacts. |
| `run.submit.sh <model> [num_epochs]` | Copies local code into the submission structure | Produces the required training and evaluation files. |
| `run.arc.sh` | Visualizes an architecture from a model or checkpoint | Used for inspection. |

`run.local.sh` computes the validation ratio from parquet row groups, trains on `train.parquet + valid.parquet`, runs inference on `test.parquet`, and reports held-out AUC and LogLoss. Labels are converted by treating `label_type == 2` as positive.

### 2.3 Model Interface

All models were aligned with the competition sample interface.

| Component | Role |
| --- | --- |
| `forward()` | Training forward path |
| `predict()` | Evaluation forward path |
| `get_sparse_params()` | Sparse optimizer parameter group |
| `get_dense_params()` | Dense optimizer parameter group |
| `reinit_high_cardinality_params()` | High-cardinality embedding re-initialization |
| `num_ns` | Number of non-sequential tokens |

Sparse embeddings were trained with Adagrad, while dense parameters were trained with AdamW. High-cardinality sparse re-initialization became a key stabilizer.

## 3. Dataset and Server Analysis

### 3.1 Data Scale

| Item | Value |
| --- | ---: |
| Parquet files | 1,000 |
| Total rows | 1,010,000 |
| Total row groups | 1,000 |
| Columns | 120 |
| Train row groups | 900 |
| Valid row groups | 100 |
| Train rows | 907,381 |
| Valid rows | 102,619 |

Each parquet file contained one row group. File-level and row-group-level splitting were therefore almost equivalent.

### 3.2 Feature Groups

| Group | Structure | Interpretation |
| --- | --- | --- |
| Metadata/label | `user_id`, `item_id`, `label_type`, `label_time`, `timestamp` | CTR label and row context |
| User integer | 35 scalar features, 11 list features | User profile and categorical attributes |
| User dense | 10 list-float features, total dimension 1,057 | Pretrained or statistical user representations |
| Item integer | 13 scalar features, 1 list feature | Item identity, category, and source attributes |
| Sequence | Four domains: a/b/c/d | Multi-domain user behavior history |

Several sequence vocabularies were extremely large, e.g., `seq_b` fid 69 with 64,710,562 IDs and `seq_c` fid 47 with 86,335,515 IDs. Dense features `user_dense_feats_61` and `87` appeared to be high-dimensional pretrained representations.

### 3.3 NaN Failure Analysis

Early HyFormer runs produced repeated NaNs. Dense feature scaling was initially suspected, but `log1p` scaling of large dense fields did not fully resolve the issue.

The final cause was fully masked attention. Some rows contained zero-length sequences, which produced all-padding masks. Passing fully masked queries into `scaled_dot_product_attention` caused NaNs during backward propagation, even when forward outputs were later sanitized.

The fix was to detect fully masked queries inside `RoPEMultiheadAttention.forward()`, temporarily open a dummy key for SDPA stability, and zero out the output afterward. Future sequence models should include an explicit unit test for empty-sequence attention masks.

## 4. Local Benchmarks

Local datasets were used to filter candidates and inspect inductive biases. They did not reliably predict submission ranking.

### 4.1 Toss Tiny

Toss tiny contained 110,257 rows and was used for fast candidate screening.

| Model | Setting | AUC | LogLoss | Interpretation |
| --- | --- | ---: | ---: | --- |
| RF | 10 epochs | 0.556976 | 0.104141 | Weak baseline. |
| MLP | 10 epochs | 0.620905 | 0.100910 | Reasonable dense baseline. |
| LSTM | 10 epochs | 0.576957 | 0.105185 | Sequence-only bias was insufficient. |
| DCNv2 | 10 epochs | 0.647718 | 0.098608 | Strong tabular cross baseline. |
| WuKong | 10 epochs | 0.675493 | 0.097939 | Best local baseline. |
| HyFormer block 2 | 10 epochs | 0.596690 | 0.100134 | Weak local result. |
| HyFormer block 4 | 10 epochs | 0.600007 | 0.100018 | Limited gain from depth. |
| HyFormerResidual | 10 epochs | 0.595316 | 0.100052 | No improvement. |
| HyFormerResidualGate | 10 epochs | 0.598816 | 0.100003 | Limited improvement. |
| HyFormerNSgate | 10 epochs | 0.615006 | 0.099876 | Mild gain. |
| EulerNet(b64) | 10 epochs | 0.664916 | 0.100351 | Strong AUC but weaker calibration. |
| SCV(b64) | 10 epochs | 0.636060 | 0.107081 | Unstable. |
| FinalMLP | 10 epochs | 0.597006 | 0.101975 | Below expectation. |
| HyFormer-WuKong simple fusion | 10 epochs | 0.664605 | 0.098509 | Looked strong locally. |
| HyFormer-WuKong bilinear fusion | 10 epochs | 0.658183 | 0.098024 | Lower AUC, better LogLoss. |
| HyFormer-WuKong-DCNv2 fusion | 10 epochs | 0.661140 | 0.097766 | Strong local LogLoss. |
| HyFormer-UniMixer token | 10 epochs | 0.600537 | 0.100137 | Tokenizer-only change was weak locally. |
| HyFormer-UniMixer block | 10 epochs | 0.593689 | 0.100272 | Block-only change degraded. |
| HyFormer-UniMixer | 10 epochs | 0.602282 | 0.100493 | Limited local gain. |
| HyFormer-UniMixer RankUp split | 10 epochs | 0.603180 | 0.100466 | Small local gain. |
| HyFormer-UniMixer NLIR | 10 epochs | 0.603881 | 0.099851 | Small local gain. |
| HyFormer-UniMixer auxloss | 10 epochs | 0.605133 | 0.100142 | Small local gain. |

Although WuKong, DCNv2, and fusion models looked strong locally, submission results favored HyFormer variants. Thus Toss tiny was useful for failure detection but insufficient for final selection.

### 4.2 Other Local Datasets

| Dataset | Main Observation |
| --- | --- |
| Frappe | Calibration was unstable; AUC alone was misleading. |
| Criteo | MLP, DCNv2, and WuKong were strong when sequence signals were weak. |
| Avazu | WuKong and EulerNet were useful early baselines. |
| KDD sample | Variance was high; very strong sample scores often indicated overfitting. |

The KDD sample result for `HyFormer two exact` reached 0.859375 AUC, but this should not be interpreted as reliable evidence of submission performance.

## 5. Submission Results

Submission AUC was the primary selection criterion.

| Model | Setting | AUC | Interpretation |
| --- | --- | ---: | --- |
| DCNv2 | 1 epoch | 0.790978 | Fast and stable tabular baseline. |
| WuKong | 1 epoch | 0.791163 | Similar to DCNv2. |
| HyFormer | 1 epoch | 0.797713 | Stronger than tabular baselines from 1 epoch. |
| EulerNet(1 layer) | 1 epoch | 0.714584 | Weak in submission setting. |
| SCV | 1 epoch | 0.723080 | Weak in submission setting. |
| HyFormer | 10 epochs | 0.813109 | Strong base model. |
| HyFormer-WuKong simple fusion | 10 epochs | 0.800326 | Local gain did not transfer. |
| HyFormer without meanpool(seq) | 10 epochs | 0.804653 | Sequence mean summary was important. |
| HyFormer tref | 10 epochs | 0.817137 | Stable high-performing variant. |
| HyFormer tref+NS | 10 epochs | 0.810007 | Additional NS conditioning hurt. |
| HyFormer LightGCN | 10 epochs | 0.627606 | Graph/label mismatch failure. |
| HyFormer LightGCN v2 | 10 epochs | 0.796200 | Label-free graph still underperformed. |
| HyFormer tref CNN | 10 epochs | 0.809216 | CNN sequence summary underperformed. |
| HyFormer tref-lite | 10 epochs | 0.812782 | Removing the refiner lost the gain. |
| HyFormer u init | 10 epochs | 0.619317 | Click/user-history initialization failed. |
| HyFormer seq init | 10 epochs | 0.809185 | Sequence prior was insufficient. |
| HyFormer-WuKong bilinear fusion | 10 epochs | 0.799282 | Bilinear fusion did not solve auxiliary noise. |
| HyFormer-WuKong-DCNv2 bilinear fusion | 10 epochs | 0.798890 | Three-way fusion failed. |
| HyFormer UniMixer | 10 epochs | 0.817867 | Best recorded AUC. |
| HyFormer UniMixer token only | 10 epochs | 0.817618 | Tokenization was highly effective. |
| HyFormer UniMixer auxloss | 10 epochs | 0.812109 | Auxiliary loss hurt submission. |
| HyFormer UniMixer NLIR | 10 epochs | 0.812992 | NLIR did not transfer well. |
| HyFormer UniMixer RankUp split | 10 epochs | 0.815673 | Useful but not best. |
| HyFormer UniMixer match token | 10 epochs | 0.806234 | Match token introduced noise. |
| HyFormer UniMixer SWA | 10 epochs | 0.748470 | Failed in this setting. |

The strongest results were `HyFormer UniMixer` (0.817867), `HyFormer UniMixer token only` (0.817618), and `HyFormer tref` (0.817137). Internal tokenization and query/global design were more effective than large N-tower fusion.

## 6. Model-Family Analysis

### 6.1 RF, MLP, and LSTM

These models served as pipeline and feature-processing sanity checks. MLP was competitive on Criteo tiny and better than RF/LSTM on Toss tiny. LSTM did not provide a strong standalone sequence inductive bias. In submission, however, HyFormer variants clearly dominated.

### 6.2 DCNv2

DCNv2 was tested as a strong sparse tabular interaction baseline. It achieved 0.790978 submission AUC after one epoch and was stable on Toss and Criteo. However, HyFormer-DCN fusion did not exceed the best HyFormer variants. DCNv2 remains valuable as an early baseline and as a possible lightweight residual auxiliary, but large fusion should be avoided.

### 6.3 WuKong

WuKong was motivated by strong feature-interaction modeling. It achieved 0.791163 submission AUC after one epoch and was strong on Toss tiny. Yet HyFormer-WuKong fusion underperformed the HyFormer base in submission. WuKong is useful as an independent baseline but can become an overfitting or noise source when attached to HyFormer.

### 6.4 HyFormer Base

HyFormer was the primary sequence-aware backbone. It processes user/item non-sequential tokens and multi-domain sequences through query-based sequence decoding and token mixing. It reached 0.797713 after one epoch and 0.813109 after ten epochs, outperforming the simple tabular baselines in submission.

### 6.5 HyFormer tref

The `tref` direction improved query generation via stable user, item, and sequence summaries.

```text
U' = Refiner(U_NS)
I' = Refiner(I_NS)
F_u = MeanPool(U')
F_i = MeanPool(I')
S_i = MeanPool(Seq_i)
global_i = [F_u ; F_i ; S_i]
q_i = FFN_i(global_i)
```

`HyFormer tref` improved submission AUC from 0.813109 to 0.817137. Removing the refiner reduced AUC to 0.812782, and adding stronger NS conditioning reduced AUC to 0.810007. This indicates that stable summary construction was more useful than stronger feature mixing.

### 6.6 Residual Variants

Residual and gated-residual variants were intended to reduce representation drift and improve gradient flow. They produced only minor local changes and did not become a major source of improvement. Residual design was useful for stability but not sufficient as a performance driver.

### 6.7 LightGCN and User Initialization

Graph and user-prior variants attempted to inject user-item interaction information into `F_u`. They failed sharply: LightGCN reached 0.627606, and user initialization reached 0.619317 submission AUC. The likely cause was train/test feature-availability mismatch or label-derived shortcut learning. Future user priors should only use information available identically at inference time.

### 6.8 N-tower and Fusion

N-tower variants were motivated by the ensemble restriction: the model could not ensemble multiple submissions, but it could contain multiple internal views. Empirically, large fusion was harmful.

| Variant | Submission AUC | Interpretation |
| --- | ---: | --- |
| HyFormer-WuKong simple fusion | 0.800326 | Large concat/product fusion failed. |
| HyFormer-WuKong bilinear fusion | 0.799282 | Constrained fusion still failed. |
| HyFormer-WuKong-DCNv2 bilinear fusion | 0.798890 | Additional tower introduced noise. |

Future N-tower designs should preserve the main path, e.g., `logit = logit_main + alpha * gate * logit_aux`, with small initialized auxiliary contribution.

### 6.9 UniMixer Variants

UniMixer variants replaced fixed RankMixer behavior with learnable tokenization or token mixing.

| Variant | Local Toss AUC | Submission AUC | Interpretation |
| --- | ---: | ---: | --- |
| HyFormer-UniMixer token | 0.600537 | 0.817618 | Tokenizer modification transferred well. |
| HyFormer-UniMixer block | 0.593689 | - | Block-only change degraded locally. |
| HyFormer-UniMixer | 0.602282 | 0.817867 | Best submission result. |
| HyFormer-UniMixer RankUp split | 0.603180 | 0.815673 | Useful but not best. |
| HyFormer-UniMixer NLIR | 0.603881 | 0.812992 | Local gain did not transfer. |
| HyFormer-UniMixer auxloss | 0.605133 | 0.812109 | Auxiliary loss hurt submission. |
| HyFormer-UniMixer match token | - | 0.806234 | Match token was noisy. |
| HyFormer-UniMixer SWA | - | 0.748470 | SWA failed. |

Tokenization changes were more reliable than heavier block or loss modifications.

## 7. HyFormer Ablation

Assets:

- Figure: [ablation_auc.svg](../assets/summary_kdd_submission/ablation_auc.svg)
- CSV: [ablation_results.csv](../assets/summary_kdd_submission/ablation_results.csv)

![HyFormer ablation AUC](../assets/summary_kdd_submission/ablation_auc.svg)

Run: `hyformer_ablation_hun_gpu6_260515_150309`, 10 epochs on `toss`.

| Variant | Best AUC | Best LogLoss | Description |
| --- | ---: | ---: | --- |
| `query_seq_only` | 0.618288 | 0.096820 | Query context uses only `MeanPool(Seq_i)`. |
| `rank_none` | 0.614427 | 0.096313 | Removes RankMixer token mixing. |
| `no_time` | 0.613943 | 0.096351 | Removes time-bucket embedding. |
| `num_queries_1` | 0.613752 | 0.096543 | Uses one query per sequence. |
| `baseline` | 0.612267 | 0.096431 | Default ablation baseline. |
| `seq_longer` | 0.610896 | 0.096613 | Uses longer/top-k sequence encoder. |
| `rank_ffn_only` | 0.610277 | 0.096411 | Keeps FFN but removes token mixing. |
| `seq_swiglu` | 0.609848 | 0.096918 | Replaces sequence self-attention with SwiGLU. |
| `query_ns_only` | 0.608567 | 0.096468 | Query context uses only user/item NS summary. |
| `rope` | 0.608465 | 0.096520 | Adds RoPE. |
| `query_zero_seq` | 0.604653 | 0.096487 | Removes sequence summary from query context. |
| `no_sparse_reinit` | 0.585525 | 0.097284 | Disables sparse embedding re-initialization. |

The best local variant was `query_seq_only`, suggesting that sequence summaries were the most stable query-generation signal. The large drop from `no_sparse_reinit` confirms the importance of high-cardinality sparse regularization. However, this ablation was local-only; final decisions were made by submission AUC.

## 8. Relation to Paper Reviews

### 8.1 RankUp

RankUp argues that deep rankers can suffer from low-rank representation collapse and proposes randomized splitting, global tokens, and cross-pretrained embedding interactions. The `hyf_unimix_rankup_split` experiment reflected this direction and achieved 0.815673 submission AUC. It supported the conclusion that tokenization-level changes are safer than adding large auxiliary towers.

### 8.2 TokenFormer

TokenFormer emphasizes Sequential Collapse Propagation, where static tokens contaminate sequence representations. The `hyf_unimix_nlir` variant tested a related gating idea. It improved local Toss AUC slightly but reached only 0.812992 in submission. Thus, small gating was stable but not a primary source of gain in this setting.

### 8.3 LoopCTR

LoopCTR scales CTR models through repeated shared layers and process supervision rather than more inference-time parameters. The `hyf_unimix_auxloss` experiment followed this motivation. It improved local Toss AUC but reduced submission AUC to 0.812109. Auxiliary supervision should therefore be weighted carefully and validated early by submission score.

## 9. Log Inventory

`KDD_submission/log` contained 105 log files, and 62 runs included held-out local test metrics.

| Run | Purpose | Local/Test Result |
| --- | --- | --- |
| `hyformer_ablation_hun_gpu6_260515_150309` | HyFormer ablation | Best variant `query_seq_only`; local test AUC 0.594585 |
| `hyf_tref_dump_hun_gpu6_260511_191959` | tref sample/local run | Local test AUC 0.743371 |
| `hyf_unimixer_hun_gpu6_260518_195502` | UniMixer tokenizer+block | Local test AUC 0.602282 |
| `hyf_unimixer_token_hun_gpu6_260518_193039` | UniMixer tokenizer only | Local test AUC 0.600537 |
| `hyf_unimix_nlir_hun_gpu6_260519_175221` | TokenFormer NLIR idea | Local test AUC 0.603881 |
| `hyf_unimix_auxloss_hun_gpu6_260519_182404` | LoopCTR/process supervision idea | Local test AUC 0.605133 |
| `hyf_wk_bifusion_hun_gpu6_260518_142637` | HyFormer-WuKong bilinear fusion | Local test AUC 0.664605 |
| `hyf_wk_dcn_bifusion_hun_gpu6_260518_144604` | HyFormer-WuKong-DCNv2 fusion | Local test AUC 0.661140 |
| `dcnv2_hun_gpu6_260430_201239` | DCNv2 sample run | Local test AUC 0.694129 |

Some early logs contain NaN or infinite LogLoss. These failures are valuable records and should be preserved.

## 10. Directions to Stop

| Direction | Evidence | Recommendation |
| --- | --- | --- |
| Click-label graph / LightGCN | 0.627606 submission AUC; v2 only 0.796200 | Stop by default. |
| User click initialization | 0.619317 submission AUC | Avoid train/test mismatch. |
| Large concat/product fusion | HyFormer-WuKong simple fusion 0.800326 | Do not disturb the main tower. |
| WuKong/DCNv2 three-way fusion | 0.798890 | Added towers produced noise. |
| CNN sequence summary | tref CNN 0.809216 | Prefer mean sequence summary. |
| SWA | UniMixer SWA 0.748470 | Avoid in this setting. |
| Over-trusting local tiny scores | Fusion looked strong locally but failed in submission | Use local scores only as filters. |

## 11. Reusable Checklist

Day 1 setup:

- Build local/submission split scripts first.
- Run DCNv2, MLP, WuKong, and HyFormer for one epoch in parallel.
- Add NaN checks, empty-sequence attention tests, and label-distribution checks.
- Ablate high-cardinality sparse re-initialization early.

Model exploration:

- If sequence signals are strong, start from HyFormer/tref-like models.
- If the data is mostly tabular, prioritize DCNv2, WuKong, and MLP.
- Use N-tower models only with residual or gated logit fusion.
- Prefer small changes to tokenization, global summaries, and query generation.

Submission policy:

- Treat local Toss/Frappe/Criteo scores as sanity checks.
- Do not trust local AUC ranking without a distribution-matched validation set.
- Stop models that improve local scores but reduce submission AUC.

Record keeping:

- Keep `log`, `checkpoint`, and `local_runs` aligned by run tag.
- Separate local and submission results in tables.
- Preserve failures, especially NaN, OOM, infinite LogLoss, and leakage-prone results.

## 12. Final Takeaways

The main contribution of `KDD_submission` was not only the best-scoring model but also a repeatable experimental framework and a clear failure map.

Reusable conclusions:

- Small improvements inside the main HyFormer path were most effective.
- Sequence mean summaries were critical.
- High-cardinality sparse re-initialization was important.
- Label-derived graph and user initialization were risky.
- Large N-tower fusion often failed despite promising local scores.
- Future competitions should run fast tabular baselines and the main sequence model in parallel, then validate local-submission mismatch as early as possible.

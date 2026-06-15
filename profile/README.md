# TAAC KDD Cup 2026 

![KDD26_Logo](../assets/KDD26-Logo4-black.png)

This repository presents our research and engineering efforts for the TAAC KDD Cup 2026 CTR Prediction Track, including scalable CTR modeling architectures, experimental pipelines, and post-competition analyses.

[![KDD Cup](https://img.shields.io/badge/KDD%20Cup-2026-0052cc?style=flat)](https://algo.qq.com/)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/TAAC2026)
[![FuxiCTR](https://img.shields.io/badge/FuxiCTR-GitHub-black?style=flat&logo=github&logoColor=white)](https://github.com/reczoo/FuxiCTR)
[![Experiment Repo](https://img.shields.io/badge/Experiment-Repository-blue?style=flat&logo=github&logoColor=white)](https://github.com/KDDcup26DIL/KDD_submission)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Documentation-00B8D9?style=flat)](https://deepwiki.com/KDDcup26DIL/KDD_submission)
[![UNIST DI Lab](https://img.shields.io/badge/UNIST-DI%20Lab-purple?style=flat)](https://sites.google.com/view/unist-dilab/)

## Team

<table>
<tr>

<td align="center">
<a href="https://github.com/hun9008">
<img src="https://github.com/hun9008.png" width="120px;" alt="정용훈"/>
<br />
<b>Yonghun Jeong</b>
</a>
<br />
MS/Ph.D. Student
</td>

<td align="center">
<a href="https://github.com/archivehee">
<img src="https://github.com/archivehee.png" width="120px;" alt="강대희"/>
<br />
<b>Daehee Kang </b>
</a>
<br />
MS/Ph.D. Student
</td>

<td align="center">
<a href="https://github.com/minssay">
<img src="https://github.com/minssay.png" width="120px;" alt="서민성"/>
<br />
<b>Minseong Seo </b>
</a>
<br />
MS/Ph.D. Student
</td>

</tr>
</table>

## Repositories

| Repository | Description |
| --- | --- |
| [`KDD_submission`](https://github.com/KDDcup26DIL/KDD_submission) | Main experiment repository used for the competition. The codebase was reorganized and adapted to comply with the official submission format of the TAAC KDD Cup 2026. |
| [`FuxiCTR`](https://github.com/KDDcup26DIL/FuxiCTR) | Repository for CTR prediction experiments based on the FuxiCTR framework, including model implementation, training, and evaluation pipelines. |
## Motivation
Recommendation research has progressed along two major branches:
1. Feature Interaction Models - focus on modeling high-dimensional multi-field categorical and contextual features.
2. Sequential Models - capture the temporal dynamcis of user behavior through embedding-based retrieval systems and Transformer-style ranking models.

Separated model progress limits → shallow corss-paradigm interaction, inconsistent optimization obejctives, limited scalability, and increasing hardware and engineering compelxity.

To address above limitations, need to further accelerate "Towards Unifying Sequence Modeling and Feature Interaction for Large-scale Recommendaiton"

## Progress

### CTR Prediction Baselines
| Model Name | Description |
| --- | --- |
| [`HyFormer`](https://arxiv.org/pdf/2601.12681) | Query Decoding module that expands non-sequential features into global tokens to decode long behavioral sequences layer-by-layer, then alternates this with a Query Boosting module to explicitly mix and enrich these tokens through lightweight feature interaction across stacked layers. |

### Previous Competition Analysis

...

### Dataset Analysis

...

### research (try)
[![HyFormer](https://img.shields.io/badge/HyFormer-arXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2601.12681)

대회에서 제공한 3개의 최근 CTR prediction 모델을 리뷰하고 그 중 코드가 제공된 HyFormer를 backbone으로 개선할 수 있는 부분을 고민했다.


## Reviews


## Conclusion


## Reference

| Paper | Conference | Topic |
| --- | --- | --- |
| [`Distribution-Aware End-to-End Embedding for Streaming Numerical Features in Click-Through Rate Prediction`](https://doi.org/10.1145/3746252.3761565) | preprint | Numerical feature embedding, Streaming training, Sample-based distribution estimation, Quantile space interpolation, Field-wise modulation |
| [`EulerNet: Adaptive Feature Interaction Learning via Euler's Formula for CTR Prediction`](https://doi.org/10.1145/3539618.3591681) | SIGIR(2023) | Euler's formula, Complex vector space, Phase and amplitude modulation, Adaptive high-order combination, Unified architecture |
| [`FINAL: Factorized Interaction Layer for CTR Prediction`](https://doi.org/10.1145/3539618.3591988) | SIGIR(2023) | Factorized interaction layer, Fast exponentiation algorithm, Hierarchical interaction expansion, Multi-block knowledge exchange, Knowledge distillation |
| [`HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction`](https://arxiv.org/pdf/2601.12681) | preprint | Global token interface, Behavior sequence decoding, Token mixture boosting, Multi-sequence independent modeling, Bidirectional information flow |
| [`OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender`](https://arxiv.org/abs/2510.26104) | preprint | Unified tokenizer, Causal attention mask, Cross-request KV caching, Single Transformer backbone, Mixed-parameterization setting |
| [`Towards Universal Sequence Representation Learning for Recommender Systems`](https://doi.org/10.1145/3534678.3539381) | KDD(2022) | UniSRec, Text-based encoding, Parameter whitening transform, Mixture-of-Experts adapter, Multi-domain contrastive learning |
| [`Streamlining Feature Interactions via Selectively Crossing Vectors for Click-Through Rate Prediction`](https://doi.org/10.1145/3746252.3761193) | CIKM(2025) | Selectively crossing vectors, Global sparse interaction graph, Multi-stage expert learning, Inter-intra stream fusion, Label bias distillation |
| [`Towards Unifying Feature Interaction Models for Click-Through Rate Prediction`](https://arxiv.org/abs/2411.12441) | preprint | IPA framework, Interaction function typification, Layer pooling method, Layer-wise weight combiner, Dimension collapse mitigation |
| [`Wukong: Towards a Scaling Law for Large-Scale Recommendation`](https://arxiv.org/abs/2403.02545) | preprint | Power-law based FM stacking, Linear compression block, Dense scaling law expansion, Low-rank computation optimization, Pyramidal structure |



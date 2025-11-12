# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Reference Coverage Analysis

**Date**: November 12, 2025  
**Purpose**: Audit all citations in final report (sections 01-12) and identify which have documentation

---

## Executive Summary

### Coverage Status

| Category                    | Papers Found in Archive | Papers Mentioned                                | Coverage |
| --------------------------- | ----------------------- | ----------------------------------------------- | -------- |
| **Core GNN/Traffic Models** | 6/6                     | STGCN, DCRNN, GraphWaveNet, ASTGCN, GCN, GAT    | ✅ 100%  |
| **Transformer & Attention** | 2/3                     | Transformer, GAT, (missing Temporal Fusion)     | 🟡 67%   |
| **Probabilistic Models**    | 1/2                     | Mixture models discussed, (missing Bishop 1994) | 🟡 50%   |
| **Classical Methods**       | 0/4                     | ARIMA, Kalman, VAR, LSTM                        | ❌ 0%    |
| **Evaluation Metrics**      | 0/2                     | CRPS, Statistical tests                         | ❌ 0%    |

**Total**: 9/17 major citations have supporting documentation (53%)

---

## Detailed Findings

### ✅ WELL-DOCUMENTED (Archive Coverage)

#### 1. Spatio-Temporal Graph Models

**Found in**: `docs/archive/architecture/`, `docs/archive/audits/UNIFIED_IO.md`

- **STGCN** (Yu et al., 2018)

  - ✅ Multiple mentions in STMGT_ARCHITECTURE_ANALYSIS.md
  - ✅ Comparison in MODEL_VALUE_AND_LIMITATIONS.md
  - 📝 Note: "Yu et al., IJCAI 2018" mentioned in 02_literature_review.md

- **GraphWaveNet** (Wu et al., 2019)

  - ✅ Extensive analysis in UNIFIED_IO.md
  - ✅ Adaptive adjacency mechanism documented
  - ✅ Comparison showing STMGT beats GraphWaveNet by 28%
  - 📝 Note: "Wu et al., IJCAI 2019" in literature review

- **ASTGCN** (Guo et al., 2019)

  - ✅ Implementation notes in archive/experimental/astgcn_v0/
  - ✅ Node-based vs edge-based comparison in UNIFIED_IO.md
  - ✅ Failed experiment documented (MAE 4.29)
  - 📝 Note: "Guo et al., AAAI 2019" in literature review

- **DCRNN** (Li et al., 2018)

  - 🟡 Mentioned indirectly through METR-LA dataset
  - 📝 Note: Needs explicit citation in literature review

- **MTGNN** (Wu et al., 2020)
  - ✅ Reference: "Wu et al., KDD 2020" in literature review
  - 🟡 Limited technical details in archive

#### 2. Graph Neural Networks

**Found in**: `docs/archive/architecture/STMGT_MODEL_ANALYSIS.md`

- **GCN** (Kipf & Welling, 2017)

  - ✅ GCN baseline implemented and tested (MAE 3.91)
  - ✅ Comparison in model_comparison_table.md
  - 🟡 Original paper not explicitly cited yet

- **GAT / GATv2Conv** (Veličković et al., 2018)
  - ✅ Extensively documented in STMGT architecture
  - ✅ Used as core spatial encoder: `GATv2Conv(96, 96, heads=4)`
  - ✅ "Dynamic message aggregation" benefits listed
  - 📝 Note: "Veličković et al. (2018)" in literature review

#### 3. Transformer Architecture

**Found in**: `docs/archive/architecture/STMGT_MODEL_ANALYSIS.md`

- **Transformer** (Vaswani et al., 2017)
  - ✅ Temporal attention module documented
  - ✅ Multi-head attention (4 heads) implementation
  - 📝 Note: "Vaswani et al. (2017)" in literature review
  - 🟡 Original "Attention Is All You Need" paper needs full citation

#### 4. Probabilistic Forecasting

**Found in**: `docs/archive/architecture/STMGT_MODEL_ANALYSIS.md`

- **Gaussian Mixture Models**

  - ✅ GMM decoder with 3 components documented
  - ✅ Negative log-likelihood loss explained
  - ✅ Code example provided
  - 🟡 Needs citation to foundational GMM work

- **Mixture Density Networks** (Bishop, 1994)

  - ❌ Not explicitly referenced in archive
  - 📝 Note: Listed in CITATIONS_NEEDED.md

- **CRPS** (Gneiting & Raftery, 2007)
  - ✅ CRPS=2.23 reported in test results
  - ❌ Metric definition not documented in archive
  - 📝 Note: Critical citation needed

---

### ⚠️ PARTIALLY DOCUMENTED

#### 5. LSTM & Sequential Models

**Found in**: Multiple training scripts and results

- **LSTM** (Hochreiter & Schmidhuber, 1997)

  - ✅ LSTM baseline implemented (MAE 4.35-4.85)
  - ✅ Results in outputs/lstm_baseline_production/
  - ❌ Original 1997 paper not cited
  - 📝 Note: "Hochreiter & Schmidhuber (1997)" needed

- **LSTM for Traffic** (Ma et al., 2015; Duan et al., 2016)
  - 📝 Note: Mentioned in literature review but no details
  - ❌ Papers not found in archive

#### 6. Advanced Transformers

- **Temporal Fusion Transformer** (Lim et al., 2021)

  - ❌ Not in archive
  - 🟡 Listed as "IMPORTANT" in CITATIONS_NEEDED.md

- **Informer** (Zhou et al., 2021)
  - ❌ Not in archive
  - 🟢 Listed as "NICE TO HAVE"

---

### ❌ NOT DOCUMENTED (Need Research)

#### 7. Classical Statistical Methods

**Status**: No documentation in archive

- **ARIMA** (Box & Jenkins)

  - ❌ Foundational work not found
  - 📝 Mentioned in 02_literature_review.md line 21-25

- **Kalman Filters** (Kalman, 1960)

  - ❌ Original paper not found
  - 📝 Mentioned in 02_literature_review.md line 27-30

- **Vector Autoregression (VAR)**
  - ❌ No documentation
  - 📝 Mentioned in 02_literature_review.md line 32-35

#### 8. Evaluation & Statistics

**Status**: No documentation in archive

- **MAPE** (Armstrong & Collopy, 1992)

  - ❌ Not documented
  - ✅ Used extensively: STMGT MAPE=19.68%

- **Diebold-Mariano Test** (1995)
  - ❌ Not documented
  - 🟡 Listed as "IMPORTANT" for forecast comparison

#### 9. Data Augmentation

**Status**: Minimal documentation

- **Time Series Augmentation** (Wen et al., 2020)
  - 🟡 Data augmentation implemented (extreme weather)
  - ❌ No theoretical paper cited
  - 📝 Code in data augmentation pipeline

#### 10. Other Supporting Work

- **ChebNet** (Defferrard et al., 2016)

  - 📝 Mentioned in 02_literature_review.md line 95
  - ❌ Not detailed in archive

- **GRU** (Cho et al., 2014)

  - 📝 Mentioned in literature review
  - ❌ Not used or detailed

- **PyTorch Geometric** (Fey & Lenssen, 2019)
  - ✅ Used throughout project
  - ❌ Not cited

---

## Section-by-Section Audit

### Section 02: Literature Review

**Status**: 📝 Has inline references but missing BibTeX

**References mentioned**:

- Yu et al., IJCAI 2018 (STGCN) ✅
- Wu et al., IJCAI 2019 (GraphWaveNet) ✅
- Wu et al., KDD 2020 (MTGNN) ✅
- Guo et al., AAAI 2019 (ASTGCN) ✅
- Zheng et al., AAAI 2020 ✅
- Li et al., AAAI 2022 ✅
- Defferrard et al. (2016) - ChebNet ❌
- Veličković et al. (2018) - GAT ✅
- Brody et al. (2022) - GATv2 ❌
- Vaswani et al. (2017) - Transformer ✅
- Perez et al. (2018) - FiLM ❌
- Zhang et al. (2019) - Weather impact ❌

**Findings**:

- 6/12 papers have archive documentation
- All major SOTA traffic models covered
- Missing: Foundational GNN papers (Kipf & Welling 2017)

### Section 03-05: Data & Preprocessing

**References mentioned**:

- OpenWeatherMap API ❌ (needs citation)
- Google Maps API ❌ (needs citation)
- Z-score normalization ❌ (textbook reference needed)

### Section 06-07: Methodology & Model

**References mentioned**:

- GAT architecture ✅ (well documented)
- Transformer attention ✅ (well documented)
- GMM probabilistic output ✅ (well documented)
- Negative log-likelihood ❌ (needs ML textbook)

### Section 08-09: Evaluation & Results

**References mentioned**:

- MAE, RMSE, R² ❌ (needs statistical textbook)
- CRPS metric ❌ (needs Gneiting & Raftery 2007)
- Model comparison ✅ (fully documented)

### Section 10-11: Conclusion & References

**Status**: 11_references.md exists but empty
**Action needed**: Fill with BibTeX entries

---

## Priority Actions for Phase 4

### 🔴 CRITICAL (Must Find)

1. **Kipf & Welling (2017)** - "Semi-Supervised Classification with Graph Convolutional Networks"

   - ICLR 2017
   - Foundation for all GNN work
   - Used in GCN baseline

2. **Vaswani et al. (2017)** - "Attention Is All You Need"

   - NeurIPS 2017
   - Foundation for Transformer
   - Used in temporal module

3. **Li et al. (2018)** - "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting"

   - ICLR 2018
   - Introduced METR-LA dataset
   - DCRNN model reference

4. **Yu et al. (2018)** - "Spatio-Temporal Graph Convolutional Networks"

   - IJCAI 2018
   - STGCN model
   - Already mentioned, need full citation

5. **Wu et al. (2019)** - "Graph WaveNet for Deep Spatial-Temporal Graph Modeling"

   - IJCAI 2019
   - Current SOTA baseline
   - Already mentioned, need full citation

6. **Hochreiter & Schmidhuber (1997)** - "Long Short-Term Memory"

   - Neural Computation
   - LSTM foundation
   - Used in baseline

7. **Gneiting & Raftery (2007)** - "Strictly Proper Scoring Rules, Prediction, and Estimation"
   - JASA
   - CRPS metric definition
   - Used in evaluation

### 🟡 IMPORTANT (Should Find)

8. **Guo et al. (2019)** - "Attention Based Spatial-Temporal Graph Convolutional Networks"

   - AAAI 2019
   - ASTGCN model
   - Already mentioned, need full citation

9. **Veličković et al. (2018)** - "Graph Attention Networks"

   - ICLR 2018
   - GAT foundation
   - Used extensively in STMGT

10. **Bishop (1994)** - "Mixture Density Networks"

    - NCRG Report
    - MDN foundation
    - Used in GMM decoder

11. **Defferrard et al. (2016)** - "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering"

    - NeurIPS 2016
    - ChebNet (precursor to GCN)

12. **Wu et al. (2020)** - "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks"
    - KDD 2020
    - MTGNN model
    - Already mentioned, need full citation

### 🟢 NICE TO HAVE (Supporting)

13. **Brody et al. (2022)** - "How Attentive are Graph Attention Networks?"

    - ICLR 2022
    - GATv2 improvements

14. Statistical textbooks for:

    - ARIMA (Box & Jenkins)
    - Kalman Filters (Kalman 1960)
    - MAE/RMSE/R² definitions

15. **Lim et al. (2021)** - "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
    - International Journal of Forecasting
    - Advanced transformer for time series

---

## Research Queries for Phase 4

### Query 1: Core GNN Papers

```
Find the following foundational GNN papers with full citations and DOI:
1. Kipf & Welling (2017) "Semi-Supervised Classification with Graph Convolutional Networks" ICLR
2. Veličković et al. (2018) "Graph Attention Networks" ICLR
3. Defferrard et al. (2016) "Convolutional Neural Networks on Graphs" NeurIPS
```

### Query 2: Traffic Forecasting SOTA

```
Find the following traffic forecasting papers with full citations:
1. Li et al. (2018) "Diffusion Convolutional Recurrent Neural Network" ICLR
2. Yu et al. (2018) "Spatio-Temporal Graph Convolutional Networks" IJCAI
3. Wu et al. (2019) "Graph WaveNet" IJCAI
4. Guo et al. (2019) "ASTGCN" AAAI
5. Wu et al. (2020) "MTGNN" KDD
```

### Query 3: Transformer & Attention

```
Find these attention mechanism papers:
1. Vaswani et al. (2017) "Attention Is All You Need" NeurIPS
2. Lim et al. (2021) "Temporal Fusion Transformers" IJF
3. Brody et al. (2022) "How Attentive are Graph Attention Networks?" ICLR
```

### Query 4: Probabilistic Forecasting

```
Find papers on probabilistic models and evaluation:
1. Bishop (1994) "Mixture Density Networks"
2. Gneiting & Raftery (2007) "Strictly Proper Scoring Rules" JASA
3. Hochreiter & Schmidhuber (1997) "Long Short-Term Memory" Neural Computation
```

### Query 5: Classical Methods

```
Find foundational papers for classical methods:
1. Box & Jenkins - ARIMA methodology
2. Kalman (1960) - Kalman Filter
3. Standard textbook references for MAE, RMSE, R²
```

---

## Output Format Needed

For each paper, generate BibTeX entry:

```bibtex
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017},
  url={https://arxiv.org/abs/1609.02907}
}
```

---

## Summary

**Strengths**:

- ✅ All major traffic forecasting models well-documented in archive
- ✅ STMGT architecture thoroughly analyzed
- ✅ Model comparison complete with actual results
- ✅ Implementation details preserved

**Gaps**:

- ❌ Missing foundational paper full citations (Kipf, Vaswani, etc.)
- ❌ No classical method references
- ❌ Evaluation metric references incomplete
- ❌ 11_references.md file empty

**Next Steps**:

1. Use Perplexity/Consensus to find 15 critical papers
2. Generate BibTeX entries for each
3. Fill 11_references.md
4. Replace inline references with proper [1], [2] citations
5. Cross-check all sections for citation completeness

---

**Estimated Time for Phase 4**: 3-4 hours

- Research: 2 hours (5 queries × 20-25 min each)
- BibTeX formatting: 1 hour
- Verification: 30 min
- Integration: 30 min

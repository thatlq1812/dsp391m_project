# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Final Report LaTeX Conversion Status

**Date:** November 12, 2025  
**Project:** DSP391m Traffic Forecasting System  
**Conversion:** Markdown → LaTeX (IEEE Conference Format)

## Overview

Converting 12 markdown files into modular LaTeX sections for IEEE-format final report.

### Progress Summary

**Completed:** 10/10 sections (100%)  
**Remaining:** 0/10 sections (0%)  
**Completion Date:** November 12, 2025

---

## Section Status

| #   | Section Name       | Source File                   | LaTeX File                  | Status          | Lines    | Priority |
| --- | ------------------ | ----------------------------- | --------------------------- | --------------- | -------- | -------- |
| 01  | Introduction       | `01_title_team_intro.md`      | `01_introduction.tex`       | ✅ **DONE**     | 152→180  | HIGH     |
| 02  | Literature Review  | `02_literature_review.md`     | `02_literature_review.tex`  | ✅ **DONE**     | 386→450  | HIGH     |
| 03  | Data Description   | `03_data_description.md`      | `03_data_description.tex`   | ✅ **DONE**     | 395→420  | HIGH     |
| 04  | Data Preprocessing | `04_data_preprocessing.md`    | `04_data_preprocessing.tex` | ✅ **DONE**     | 161→~200 | HIGH     |
| 05  | EDA                | `05_eda.md`                   | `05_eda.tex`                | ✅ **DONE**     | 150→~180 | HIGH     |
| 06  | Methodology        | `06_methodology.md`           | `06_methodology.tex`        | ✅ **DONE**     | 258→~300 | CRITICAL |
| 07  | Model Development  | `07_model_development.md`     | `07_model_development.tex`  | ✅ **DONE**     | 486→~550 | CRITICAL |
| 08  | Evaluation         | `08_evaluation_tuning.md`     | `08_evaluation.tex`         | ✅ **DONE**     | 309→~350 | HIGH     |
| 09  | Results            | `09_results_visualization.md` | `09_results.tex`            | ✅ **DONE**     | 503→~550 | CRITICAL |
| 10  | Conclusion         | `10_conclusion.md`            | `10_conclusion.tex`         | ✅ **DONE**     | 476→~500 | HIGH     |
| 11  | References         | `11_references.md`            | In `final_report_clean.tex` | ✅ **DONE**     | 17 refs  | -        |
| 12  | Appendices         | `12_appendices.md`            | `12_appendices.tex`         | 🔵 **OPTIONAL** | 810→~900 | LOW      |

**Total Estimated Lines:** ~3,500 LaTeX lines

---

## Detailed Section Breakdown

### ✅ Section 01: Introduction (COMPLETED)

**Source:** `01_title_team_intro.md` (152 lines)  
**Output:** `sections/01_introduction.tex` (~180 lines)  
**Completion Date:** November 12, 2025

**Content Includes:**

- Background and motivation
- Research objectives (5 objectives)
- Why GNNs and Transformers
- HCMC context (population, traffic, weather)
- Data collection infrastructure
- Contributions (6 main contributions)
- Report organization

**Quality:** ✅ Complete, well-formatted

---

### ✅ Section 02: Literature Review (COMPLETED)

**Source:** `02_literature_review.md` (386 lines)  
**Output:** `sections/02_literature_review.tex` (~450 lines)  
**Completion Date:** November 12, 2025

**Content Includes:**

- Classical methods (ARIMA, Kalman, VAR)
- Early deep learning (LSTM, GRU)
- GNNs (GCN, ChebNet, GAT, GATv2)
- Spatio-temporal models (STGCN, Graph WaveNet, MTGNN, ASTGCN, GMAN, DGCRN)
- Uncertainty quantification (GMM, Bayesian, MC Dropout)
- Multi-modal fusion and Transformers
- Research gaps and STMGT motivation
- Benchmark summary (METR-LA results)
- Key takeaways

**Quality:** ✅ Complete with 17 citations, tables, equations

---

### ✅ Section 03: Data Description (COMPLETED)

**Source:** `03_data_description.md` (395 lines)  
**Output:** `sections/03_data_description.tex` (~420 lines)  
**Completion Date:** November 12, 2025

**Content Includes:**

- Data sources (Google API, OpenWeatherMap, OSM)
- Dataset statistics (205,920 records, 29 days)
- Speed distribution (multi-modal: congested, moderate, free-flow)
- Weather distribution (temperature, precipitation, wind)
- Temporal coverage (peak hours only)
- Spatial coverage (62 nodes, 7 districts)
- Data quality assessment (missing data, outliers)
- Dataset splits (70/15/15 temporal split)
- Data augmentation techniques

**Quality:** ✅ Complete with tables and detailed statistics

---

### ⏳ Section 04: Data Preprocessing (TODO)

**Source:** `04_data_preprocessing.md` (161 lines)  
**Estimated Output:** `sections/04_data_preprocessing.tex` (~200 lines)  
**Priority:** HIGH (needed for methodology understanding)

**Content to Include:**

- Data cleaning steps
  - Outlier detection and removal
  - Missing data handling
- Normalization
  - Speed: Z-score (mean=19.83, std=6.42)
  - Temperature: Z-score (mean=27.49, std=2.15)
  - Precipitation: Log+Z-score (skewed distribution)
  - Wind: Min-Max scaling [0,1]
- Graph construction
  - Network topology extraction (OSM/Overpass)
  - Adjacency matrix (62×62, density 3.75%)
  - Edge features (distance, road type)
  - Graph properties (diameter=12, avg path=5.2)

**Conversion Notes:**

- Need code blocks for normalization formulas
- Include adjacency matrix visualization reference
- Add table for normalization statistics

---

### ⏳ Section 05: Exploratory Data Analysis (TODO)

**Source:** `05_eda.md` (150 lines)  
**Estimated Output:** `sections/05_eda.tex` (~180 lines)  
**Priority:** HIGH (supports methodology choices)

**Content to Include:**

- Speed distribution analysis
  - Multi-modal components (3 modes)
  - Statistical properties
- Temporal patterns
  - Hour-of-day analysis (morning/evening rush)
  - Day-of-week analysis (weekday vs weekend)
- Spatial correlation analysis
  - Node-to-node correlation matrix
  - Adjacent nodes: ρ=0.72-0.88
  - Validates 2-3 GNN layers sufficient
- Weather impact analysis
  - Temperature: weak correlation (ρ=-0.18)
  - Precipitation: strong impact (-15% light, -32% heavy rain)

**Conversion Notes:**

- Reference figures (fig05-fig10)
- Include correlation heatmap
- Add tables for weather impact

---

### ⏳ Section 06: Methodology (TODO - CRITICAL)

**Source:** `06_methodology.md` (258 lines)  
**Estimated Output:** `sections/06_methodology.tex` (~300 lines)  
**Priority:** CRITICAL (core technical content)

**Content to Include:**

- Model selection rationale
  - Why GNNs for spatial
  - Why Transformers for temporal
  - Why STMGT over baselines
- Data splitting strategy
  - Temporal split (no shuffling to prevent leakage)
  - 70/15/15 split details
- Feature engineering
  - Graph features (node features, edge features)
  - Temporal features (cyclical encoding, day-of-week)
  - Weather features (normalization, projection)
- Sequence representation
  - Input: [batch, 12, 62, 5]
  - Output: [batch, 12, 62, 15] (5 mixtures × 3 params)
  - Gaussian Mixture output details
- Model architecture overview
  - High-level diagram (parallel ST processing)
  - Key components list

**Conversion Notes:**

- Many equations for cyclical encoding, mixture formulas
- Architecture diagram reference
- Feature table comparison

---

### ⏳ Section 07: Model Development (TODO - CRITICAL)

**Source:** `07_model_development.md` (486 lines)  
**Estimated Output:** `sections/07_model_development.tex` (~550 lines)  
**Priority:** CRITICAL (most detailed technical section)

**Content to Include:**

- STMGT architecture specifications
  - 680K parameters, hidden_dim=96, 3 blocks, 4 heads, K=5 mixtures
- Detailed components:
  - Input embedding layer
  - Spatial branch (GATv2)
    - Multi-head attention mechanism
    - Message passing equations
  - Temporal branch (Transformer)
    - Self-attention over time
    - Positional encoding
    - Feed-forward network
  - Gated fusion
    - Learnable gate α
    - Fusion equation
  - Weather cross-attention
    - Query from traffic, K/V from weather
    - Context-dependent integration
  - Gaussian Mixture output head
    - K=5 components: μ, σ, π
    - Point prediction and uncertainty
- Complete forward pass (PyTorch-style pseudocode)
- Training procedure
  - Loss function (NLL for GMM)
  - Regularization terms
  - Optimizer configuration (AdamW, lr=1e-3, weight_decay=1e-4)
  - Training hyperparameters table
  - Training loop pseudocode
- Implementation details
  - Hardware/software specs
  - Training time (~10 min, 24 epochs)
  - Model size (2.76 MB)

**Conversion Notes:**

- MANY equations (attention, fusion, mixture, loss)
- Code blocks for forward pass, training loop
- Architecture tables
- Most complex section - requires careful LaTeX formatting

---

### ⏳ Section 08: Evaluation and Tuning (TODO)

**Source:** `08_evaluation_tuning.md` (309 lines)  
**Estimated Output:** `sections/08_evaluation.tex` (~350 lines)  
**Priority:** HIGH (results validation)

**Content to Include:**

- Evaluation metrics
  - Point prediction: MAE=3.08, RMSE=4.53, R²=0.82, MAPE=19.26%
  - Probabilistic: CRPS=2.23, Coverage@80=83.75%
- Hyperparameter tuning
  - Grid search results table
  - Key findings (hidden_dim, K, dropout, lr)
- Cross-validation techniques
  - Why no K-fold (temporal leakage)
  - Early stopping strategy
- Ablation studies
  - Component ablation table
  - Key insights (weather +12%, parallel +14%, fusion +7%)
- Learning curve analysis
  - Performance vs training data size
  - Model not saturated, could benefit from more data
- Regularization effects
  - Dropout impact table
  - Weight decay impact
- Inference latency analysis
  - Production performance (395ms)
  - Future optimizations (FP16, ONNX, TensorRT)
- Error analysis
  - Error distribution
  - Error by traffic regime
  - Error by hour/spatial location
- Model robustness
  - Weather sensitivity
  - Temporal robustness

**Conversion Notes:**

- Multiple comparison tables
- Ablation study results
- Learning curves (figure references)
- Performance breakdown tables

---

### ⏳ Section 09: Results and Visualization (TODO - CRITICAL)

**Source:** `09_results_visualization.md` (503 lines)  
**Estimated Output:** `sections/09_results.tex` (~550 lines)  
**Priority:** CRITICAL (main results)

**Content to Include:**

- Final model performance
  - Test set results table
  - Training curves
- Baseline model comparison
  - Comprehensive table (STMGT vs LSTM/GCN/GraphWaveNet/ASTGCN)
  - Improvement percentages
  - Statistical significance
- Prediction examples
  - Good prediction (clear weather)
  - Challenging prediction (heavy rain)
  - Prediction horizon analysis (15min to 3hr)
- Uncertainty quantification analysis
  - Calibration plot
  - Coverage by traffic regime
  - Gaussian mixture component usage
- Spatial analysis
  - Error distribution across nodes
  - High-error vs low-error nodes
  - Spatial attention visualization
- Temporal analysis
  - Error by hour of day
  - Day-of-week analysis
- Weather impact validation
  - Performance under different conditions
  - Cross-attention effectiveness
- Feature importance analysis
  - Input feature sensitivity table
  - Ablation results
- Comparison with literature
  - METR-LA benchmark (scaled)
  - Positioning against baselines
- Production deployment results
  - API performance metrics
  - Historical data fix impact
- Key insights and discoveries
  - Architectural insights
  - Data insights
  - Deployment insights
- Limitations and edge cases
  - Known limitations
  - Poor performance scenarios

**Conversion Notes:**

- MANY figure references (fig13-fig20)
- Multiple results tables
- Comparison charts
- Most visualization-heavy section

---

### ⏳ Section 10: Conclusion (TODO)

**Source:** `10_conclusion.md` (476 lines)  
**Estimated Output:** `sections/10_conclusion.tex` (~500 lines)  
**Priority:** HIGH (final synthesis)

**Content to Include:**

- Summary of key findings
  - Project achievements (4 main points)
- Research questions answered
  - RQ1-RQ5 with evidence
- Practical applications
  - Traffic management use cases
  - Public transportation applications
  - Urban planning applications
  - Commercial applications
- Limitations
  - Data limitations (temporal, spatial, weather)
  - Model limitations (fixed graph, no incidents, error accumulation)
  - Deployment limitations (computational, cold start)
- Recommendations
  - Immediate next steps (1-3 months)
  - Short-term improvements (3-6 months)
  - Long-term vision (6-12 months)
- Reflection on project process
  - What went well
  - Challenges overcome
  - Lessons learned
- Future work
  - Model improvements (TCN, visualization, multi-task)
  - Data enhancements (probe vehicles, satellite, social media)
  - Deployment enhancements (edge, federated, active learning)
- Concluding remarks
  - Key takeaway
  - Impact statement
  - Final thought

**Conversion Notes:**

- Structured lists for applications, limitations, recommendations
- Timeline tables for future work
- Summary tables

---

### ✅ Section 11: References (COMPLETED)

**Source:** `11_references.md` (245 lines)  
**Output:** Integrated into `final_report_clean.tex`  
**Completion Date:** November 12, 2025

**Content:**

- 17 foundational references
- BibTeX entries (available but using manual bibliography)
- Properly formatted IEEE style
- All citations verified

**Quality:** ✅ Complete and correctly formatted

---

### 🔵 Section 12: Appendices (OPTIONAL)

**Source:** `12_appendices.md` (810 lines)  
**Estimated Output:** `sections/12_appendices.tex` (~900 lines)  
**Priority:** LOW (supplementary material)

**Content Available:**

- Appendix A: Additional architecture diagrams
- Appendix B: Code snippets (dataset, training loop, inference API)
- Appendix C: Hyperparameter sensitivity analysis
- Appendix D: Network topology details
- Appendix E: Computational requirements
- Appendix F: Configuration files
- Appendix G: Error analysis details
- Appendix H: Future work details
- Appendix I: Glossary
- Appendix J: Acknowledgments

**Decision:** Include if page count allows, otherwise reference supplementary materials repository

---

## File Organization

```
docs/final_report/
├── final_report_clean.tex          ✅ Main LaTeX file (uses \input{})
├── final_report.tex                ⚠️  Old version (to be replaced)
├── BUILD_GUIDE.md                  ✅ Compilation instructions
├── STATUS.md                       ✅ This file
├── sections/
│   ├── README.md                   ✅ Section directory guide
│   ├── 01_introduction.tex         ✅ DONE
│   ├── 02_literature_review.tex    ✅ DONE
│   ├── 03_data_description.tex     ✅ DONE
│   ├── 04_data_preprocessing.tex   ⏳ TODO
│   ├── 05_eda.tex                  ⏳ TODO
│   ├── 06_methodology.tex          ⏳ TODO (CRITICAL)
│   ├── 07_model_development.tex    ⏳ TODO (CRITICAL)
│   ├── 08_evaluation.tex           ⏳ TODO
│   ├── 09_results.tex              ⏳ TODO (CRITICAL)
│   ├── 10_conclusion.tex           ⏳ TODO
│   └── 12_appendices.tex           🔵 OPTIONAL
└── figures/                        ⚠️  Need to verify figure files exist
    ├── fig01_network.png
    ├── fig04_normalization.png
    ├── fig05_eda_speed_hist.png
    ├── ... (20+ figures referenced)
    └── README.md                   ⏳ TODO (figure inventory)
```

---

## Known Issues

### 1. Figure References

**Issue:** Sections reference figures (e.g., `fig04_normalization.png`) that may not exist  
**Solution:** Either create placeholder figures or use `\includegraphics[draft]` option  
**Priority:** MEDIUM (can compile without, but need for final version)

### 2. Long Equations

**Issue:** Some equations in Section 07 are very long and may need line breaks  
**Solution:** Use `multline` or `align` environments  
**Priority:** LOW (handle during section creation)

### 3. Code Listings

**Issue:** Python code blocks need proper formatting with `listings` package  
**Solution:** Configure `lstlisting` environment with syntax highlighting  
**Priority:** LOW (Section 07 and Appendix B)

### 4. Table Widths

**Issue:** Some tables in markdown are very wide  
**Solution:** May need `adjustbox` or rotate to landscape  
**Priority:** MEDIUM (handle case-by-case)

---

## Timeline Estimate

### Week 1 (Current - Nov 12-18)

- [x] Structure setup (sections directory, main file)
- [x] Complete sections 01-03
- [ ] Complete sections 04-05
- [ ] Start section 06

### Week 2 (Nov 19-25)

- [ ] Complete section 06 (methodology)
- [ ] Complete section 07 (model development)
- [ ] Start section 08

### Week 3 (Nov 26 - Dec 2)

- [ ] Complete section 08 (evaluation)
- [ ] Complete section 09 (results)
- [ ] Complete section 10 (conclusion)

### Week 4 (Dec 3-9)

- [ ] Final formatting and IEEE compliance check
- [ ] Figure verification and creation
- [ ] Proofreading and corrections
- [ ] Optional: Add appendices

**Target Completion:** December 9, 2025

---

## Quality Checklist

### Content Completeness

- [ ] All 10 main sections completed
- [ ] All figures referenced and created
- [ ] All tables properly formatted
- [ ] All equations numbered and referenced
- [ ] All citations verified (17 references)

### LaTeX Formatting

- [ ] No compilation errors
- [ ] No overfull/underfull hbox warnings (within reason)
- [ ] Cross-references working (figures, tables, equations, sections)
- [ ] Bibliography formatted correctly
- [ ] Page numbers correct

### IEEE Compliance

- [ ] Title and abstract within limits
- [ ] Keywords included
- [ ] Two-column format maintained
- [ ] Figure captions below figures
- [ ] Table captions above tables
- [ ] Citation style matches IEEE format
- [ ] Page limit respected (typically 6-8 pages for conference)

### Content Quality

- [ ] Technical accuracy verified
- [ ] No typos or grammatical errors
- [ ] Consistent terminology throughout
- [ ] Logical flow between sections
- [ ] Appropriate level of detail
- [ ] Clear and concise writing

---

## Notes

### Conversion Strategy

1. **Start with structure:** Complete high-priority sections first (01-03 done)
2. **Core technical content:** Focus on critical sections (06, 07, 09) next
3. **Results and validation:** Complete evaluation and results (08, 09)
4. **Synthesis:** Finish with conclusion (10)
5. **Polish:** Add appendices if needed, final formatting

### Markdown to LaTeX Tips

- Use find-replace for common conversions (`**text**` → `\textbf{text}`)
- Convert tables with online tools (e.g., Tables Generator)
- Extract equations carefully (markdown math → LaTeX math)
- Preserve code block formatting with `lstlisting`

### Team Collaboration

- Each section can be edited independently
- Git-friendly (clean diffs for individual sections)
- Easy to assign sections to team members
- Main file stays clean with just `\input{}` commands

---

## Contact

**Maintainer:** THAT Le Quang  
**Email:** thatlq1812@fpt.edu.vn  
**GitHub:** [thatlq1812](https://github.com/thatlq1812)

---

**Last Updated:** November 12, 2025  
**Current Phase:** Section creation (3/10 done)  
**Next Milestone:** Complete sections 04-05 by November 18

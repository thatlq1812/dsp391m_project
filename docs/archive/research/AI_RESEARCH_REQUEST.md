# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# AI Research Request - Academic Citations for Traffic Forecasting Project

**Date:** November 12, 2025  
**Purpose:** Find full bibliographic information and BibTeX entries for academic papers  
**Usage:** Final report for DS Capstone project on traffic speed forecasting

---

## Project Context

### Our Project: STMGT (Spatio-Temporal Multi-Modal Graph Transformer)

**What we built:**

- A novel deep learning model for traffic speed forecasting in Ho Chi Minh City, Vietnam
- Combines Graph Attention Networks (GAT) + Transformer attention + Gaussian Mixture Model probabilistic output
- Trained on 29 days of real GPS trajectory data (205,920 samples)
- 62 road network nodes, 144 directed edges

**Performance achieved:**

- **STMGT V3:** MAE = 3.08 km/h, R² = 0.817 (our model)
- **GCN baseline:** MAE = 3.91 km/h
- **LSTM baseline:** MAE = 4.35 km/h
- **GraphWaveNet (SOTA 2019):** MAE = 11.04 km/h

**Key innovations:**

1. Edge-based prediction (not node-based like ASTGCN)
2. Multi-modal fusion (spatial + temporal + weather)
3. Probabilistic output with uncertainty quantification (CRPS = 2.23)
4. Beats current SOTA by 28% on our dataset

**What we need:**

- Full academic citations for papers we reference in literature review
- BibTeX format ready for LaTeX compilation
- Papers range from classical methods (ARIMA, Kalman) to latest GNN/Transformer work

---

## Research Instructions

### For Each Paper, Please Provide:

1. **Full BibTeX entry** in this format:

```bibtex
@inproceedings{authorYear,
  title={Full Paper Title},
  author={Author1 and Author2 and Author3},
  booktitle={Conference/Journal Name},
  year={YYYY},
  pages={start--end},
  organization={Publisher if applicable},
  doi={10.xxxx/xxxxx},
  url={https://arxiv.org/abs/xxxx.xxxxx or DOI URL}
}
```

2. **Verification info:**

   - Confirm conference/journal (e.g., ICLR 2017, NeurIPS 2020)
   - Confirm first author and year
   - Provide DOI or arXiv link

3. **Brief note** (1-2 sentences):
   - Why this paper is important to traffic forecasting / our work
   - Example: "Introduced GCN which became foundation for all spatial-temporal models"

---

## Papers to Find (Grouped by Priority)

---

## 🔴 CRITICAL PRIORITY (7 Papers) - Must Have for Literature Review

### 1. Graph Convolutional Networks (GCN)

**Known Information:**

- Authors: Thomas N. Kipf, Max Welling
- Title: "Semi-Supervised Classification with Graph Convolutional Networks"
- Conference: ICLR 2017
- arXiv: 1609.02907

**Why we need it:**

- Foundation of all graph neural networks
- We use GCN-based methods throughout our architecture
- Our GCN baseline model uses this approach
- Cited 10,000+ times

**What to find:**

- Full BibTeX with page numbers, DOI
- Official ICLR proceedings link if available

---

### 2. Transformer - Attention Is All You Need

**Known Information:**

- Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
- Title: "Attention Is All You Need"
- Conference: NeurIPS 2017 (originally NIPS 2017)
- arXiv: 1706.03762

**Why we need it:**

- Foundation of all attention mechanisms
- Our temporal module uses Transformer multi-head attention
- Revolutionary paper in deep learning

**What to find:**

- Full BibTeX with all 8 authors
- NeurIPS official proceedings citation

---

### 3. DCRNN - Diffusion Convolutional Recurrent Neural Network

**Known Information:**

- Authors: Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu
- Title: "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting"
- Conference: ICLR 2018
- Introduced METR-LA dataset (widely used benchmark)

**Why we need it:**

- First major application of GNN to traffic forecasting
- Introduced the METR-LA dataset (standard benchmark)
- We compare against their approach conceptually

**What to find:**

- Full BibTeX
- ICLR 2018 proceedings link
- arXiv link if available

---

### 4. STGCN - Spatio-Temporal Graph Convolutional Networks

**Known Information:**

- Authors: Bing Yu, Haoteng Yin, Zhanxing Zhu
- Title: "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting"
- Conference: IJCAI 2018
- Currently cited as "Yu et al., IJCAI 2018" in our report

**Why we need it:**

- Major baseline model for traffic forecasting
- Combines ChebNet graph convolutions with temporal convolutions
- Widely used architecture pattern

**What to find:**

- Full BibTeX with all authors
- IJCAI 2018 proceedings page numbers
- DOI or official URL

---

### 5. GraphWaveNet - Graph WaveNet for Deep Spatial-Temporal Graph Modeling

**Known Information:**

- Authors: Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Chengqi Zhang
- Title: "Graph WaveNet for Deep Spatial-Temporal Graph Modeling"
- Conference: IJCAI 2019
- Currently cited as "Wu et al., IJCAI 2019" in our report

**Why we need it:**

- State-of-the-art baseline we compare against
- Adaptive adjacency matrix learning
- We beat this model by 28% on our dataset (major result)

**What to find:**

- Full BibTeX
- IJCAI 2019 official proceedings
- arXiv link if available

---

### 6. LSTM - Long Short-Term Memory

**Known Information:**

- Authors: Sepp Hochreiter, Jürgen Schmidhuber
- Title: "Long Short-Term Memory"
- Journal: Neural Computation, Vol 9, Issue 8, 1997
- Pages: 1735-1780

**Why we need it:**

- Foundation of all recurrent neural networks
- Our LSTM baseline uses this architecture
- Classic paper (cited 50,000+ times)

**What to find:**

- Full journal citation with volume, issue, pages
- DOI for Neural Computation journal

---

### 7. CRPS - Strictly Proper Scoring Rules

**Known Information:**

- Authors: Tilmann Gneiting, Adrian E. Raftery
- Title: "Strictly Proper Scoring Rules, Prediction, and Estimation"
- Journal: Journal of the American Statistical Association (JASA), 2007
- Vol 102, Issue 477

**Why we need it:**

- Defines the CRPS (Continuous Ranked Probability Score) metric
- We use CRPS = 2.23 to evaluate probabilistic forecasting
- Standard reference for probabilistic evaluation

**What to find:**

- Full journal citation
- DOI for JASA paper
- Page numbers

---

## 🟡 IMPORTANT PRIORITY (5 Papers) - Should Have

### 8. ASTGCN - Attention Based Spatial-Temporal Graph Convolutional Networks

**Known Information:**

- Authors: Shengnan Guo, Youfang Lin, Ning Feng, Chao Song, Huaiyu Wan
- Title: "Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting"
- Conference: AAAI 2019
- Currently cited as "Guo et al., AAAI 2019" in our report

**Why we need it:**

- Major baseline model we compare against
- Combines attention mechanisms with GCN
- We tested this model (MAE = 4.29 on our data)

---

### 9. GAT - Graph Attention Networks

**Known Information:**

- Authors: Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
- Title: "Graph Attention Networks"
- Conference: ICLR 2018
- arXiv: 1710.10903

**Why we need it:**

- Core of our spatial encoder
- We use GATv2Conv (improved version) in STMGT
- Attention weights for neighboring road segments

---

### 10. Mixture Density Networks

**Known Information:**

- Author: Christopher M. Bishop
- Title: "Mixture Density Networks"
- Type: NCRG Technical Report (Neural Computing Research Group), 1994
- Aston University

**Why we need it:**

- Foundation of our GMM probabilistic output
- We predict 3 Gaussian components for uncertainty
- Classic work on probabilistic neural networks

**Alternative:** If hard to find, cite Bishop's 1995 Neural Networks journal paper on MDN

---

### 11. ChebNet - Convolutional Neural Networks on Graphs

**Known Information:**

- Authors: Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst
- Title: "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering"
- Conference: NeurIPS 2016
- arXiv: 1606.09375

**Why we need it:**

- Precursor to GCN
- Introduces Chebyshev polynomial approximation for graph filters
- Used in STGCN architecture

---

### 12. MTGNN - Multivariate Time Series Forecasting with Graph Neural Networks

**Known Information:**

- Authors: Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Xiaojun Chang, Chengqi Zhang
- Title: "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks"
- Conference: KDD 2020
- Currently cited as "Wu et al., KDD 2020" in our report

**Why we need it:**

- Recent advancement in graph-based time series forecasting
- Learns graph structure from data
- Shows latest trends in the field

---

## 🟢 SUPPORTING PAPERS (3 Papers) - Nice to Have

### 13. GATv2 - How Attentive are Graph Attention Networks?

**Known Information:**

- Authors: Shaked Brody, Uri Alon, Eran Yahav
- Title: "How Attentive are Graph Attention Networks?"
- Conference: ICLR 2022
- arXiv: 2105.14491

**Why we need it:**

- We use GATv2Conv (improved GAT) in our model
- Fixes dynamic attention problem in original GAT
- Recent improvement to core component

---

### 14. Temporal Fusion Transformers

**Known Information:**

- Authors: Bryan Lim, Sercan Ö. Arık, Nicolas Loeff, Tomas Pfister
- Title: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
- Journal: International Journal of Forecasting, 2021
- arXiv: 1912.09363

**Why we need it:**

- Advanced Transformer for time series
- Shows latest developments in Transformer-based forecasting
- Interpretable attention weights

---

### 15. ARIMA Foundational Work

**Known Information:**

- Authors: George E. P. Box, Gwilym M. Jenkins
- Title: "Time Series Analysis: Forecasting and Control"
- Type: Book, multiple editions
- First published: 1970, revised 1976, 2008, 2015

**Why we need it:**

- Classical baseline method we mention
- Foundation of statistical time series forecasting
- Standard reference for ARIMA methodology

**What to find:**

- Book citation (prefer recent edition like 5th ed. 2015)
- Publisher: Wiley
- ISBN if available

---

## Additional Supporting References (If Easy to Find)

### 16. Statistical Metrics (MAE, RMSE, R²)

**Option A:** Standard ML textbook:

- "The Elements of Statistical Learning" by Hastie, Tibshirani, Friedman (2009)
- Springer
- Free PDF available

**Option B:** Specific metric papers:

- Willmott & Matsuura (2005) for MAE advantages
- Hyndman & Koehler (2006) for forecast accuracy measures

---

### 17. PyTorch Geometric (Tool we use)

**Known Information:**

- Authors: Matthias Fey, Jan Eric Lenssen
- Title: "Fast Graph Representation Learning with PyTorch Geometric"
- Workshop: ICLR 2019 Workshop on Representation Learning on Graphs and Manifolds
- arXiv: 1903.02428

**Why we need it:**

- Core library we use for GNN implementation
- Should acknowledge in technical implementation section

---

## Output Format Request

### Please organize as:

````markdown
# Critical Papers (1-7)

## 1. Kipf & Welling 2017 - GCN

**BibTeX:**

```bibtex
[full entry here]
```
````

**Verification:**

- Conference: ICLR 2017 ✓
- DOI: [link]
- arXiv: 1609.02907 ✓

**Note:** Foundation paper for graph convolutional networks, cited in spatial encoder design.

---

[Repeat for all papers]

```

---

## Notes for AI Research Tool

### Search Tips:

1. **For conference papers:** Search "DBLP [paper title]" or "Semantic Scholar [title]"
2. **For arXiv papers:** Use arXiv ID if known (e.g., 1706.03762)
3. **For journal papers:** Search journal name + year + authors
4. **For books:** Include edition number and ISBN

### Verification Checklist:

- ✓ Author names match (first author at minimum)
- ✓ Year is correct
- ✓ Conference/journal name is official (not workshop/preprint)
- ✓ BibTeX uses correct entry type (@inproceedings, @article, @book)
- ✓ URL/DOI is accessible

### Common Pitfalls to Avoid:

- ❌ Using arXiv citation when official conference version exists
- ❌ Missing page numbers for conference proceedings
- ❌ Incomplete author lists (only first author)
- ❌ Wrong conference name (e.g., "NIPS" vs "NeurIPS")
- ❌ Using workshop papers instead of main conference

---

## Timeline & Deliverable

**Estimated Research Time:** 2-3 hours total
- Critical papers (7): ~1.5 hours (12-15 min each)
- Important papers (5): ~45 min (9 min each)
- Supporting papers (3): ~30 min (10 min each)

**Deliverable Format:**
- Single markdown file with all BibTeX entries
- Organized by priority (Critical → Important → Supporting)
- Each entry verified with DOI/arXiv link
- Ready to copy-paste into `11_references.md`

---

## Questions to Answer (Optional Context)

If the AI research tool can answer these, it helps our writing:

1. **For STGCN (Yu et al. 2018):** What temporal convolution kernel size did they use?
2. **For GraphWaveNet (Wu et al. 2019):** What datasets did they test on? (We know METR-LA, PeMS-BAY)
3. **For DCRNN (Li et al. 2018):** How many nodes/edges in METR-LA dataset?
4. **For CRPS (Gneiting 2007):** What page range defines CRPS formula?

These help us cite specific details accurately.

---

## Contact & Clarifications

**If any paper is ambiguous:**
- Prioritize official conference/journal version over arXiv
- Prefer most cited version (check Google Scholar)
- If multiple papers by same authors, choose the one matching our description

**If cannot find a paper:**
- Note which one is missing
- Suggest alternative if available (e.g., different Bishop MDN paper)
- We can work with 80% coverage (12/15 papers)

---

## Final Checklist

Before submitting results, verify:

- [ ] All 7 critical papers found with BibTeX
- [ ] At least 4/5 important papers found
- [ ] BibTeX entries compile (no syntax errors)
- [ ] DOI/arXiv links are clickable and work
- [ ] Author names are complete (all authors listed)
- [ ] Conference/journal names are official
- [ ] Year matches our citations in report

---

**Thank you for helping complete our academic citation list!**

This will enable us to:
1. Properly credit foundational work
2. Pass academic integrity review
3. Compile professional LaTeX report with bibliography
4. Submit for DS Capstone evaluation

**Project Goal:** Traffic speed forecasting system for Ho Chi Minh City using state-of-the-art GNN + Transformer hybrid architecture.

**Report Status:** 15/17 figures done, 4 models trained, analysis complete. Just need proper citations to finalize!
```

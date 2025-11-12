# Quick Reference Guide for AI Citation Research

**Purpose:** Guide for using AI tools (Perplexity, Claude, ChatGPT, Consensus.app) to find academic citations

---

## Files Overview

1. **`AI_RESEARCH_REQUEST.md`** (Main file)

   - Complete project context
   - 15 papers to find with known information
   - Output format requirements
   - 📋 **Give this to AI**

2. **`BIBTEX_RESULTS_TEMPLATE.md`** (Template)

   - Pre-formatted template for results
   - Checkboxes for verification
   - 📝 **AI fills this out**

3. **`REFERENCE_COVERAGE_ANALYSIS.md`** (Background)
   - What we already have in archive
   - Why each paper is needed
   - 📚 **For your reference**

---

## How to Use with Different AI Tools

### Option 1: Perplexity.ai (Recommended)

**Why:** Direct access to academic databases, provides citations

**Prompt Template:**

```
I need to find academic citations for a traffic forecasting research project.
I have a detailed list of 15 papers with known authors and conferences.

Please read this file: [paste AI_RESEARCH_REQUEST.md]

Then find BibTeX entries for papers 1-7 (Critical Priority) first.
For each paper provide:
1. Full BibTeX entry
2. DOI or arXiv link
3. Verification that conference/year matches

Start with paper #1 (Kipf & Welling 2017 GCN).
```

**Follow-up:**

- Ask for 2-3 papers at a time to avoid overwhelming
- Verify each batch before continuing
- Use "Pro" mode for better accuracy

---

### Option 2: Claude/ChatGPT with Scholar Tools

**Why:** Can browse web and access Google Scholar

**Prompt Template:**

```
I'm doing academic citation research for a deep learning traffic forecasting project.

Task: Find full bibliographic information and BibTeX entries for 15 papers.

Context file: [paste relevant sections from AI_RESEARCH_REQUEST.md]

Process:
1. Search Google Scholar for each paper
2. Find official conference/journal version (not arXiv preprint)
3. Generate BibTeX using Google Scholar's "Cite" feature
4. Verify DOI link works
5. Fill out the template: [paste relevant section from BIBTEX_RESULTS_TEMPLATE.md]

Start with Critical Papers (1-7). Do paper #1 first as an example.
```

---

### Option 3: Consensus.app

**Why:** Specialized AI for academic papers, high accuracy

**Process:**

1. Search for each paper by title
2. Click "Cite" button
3. Copy BibTeX format
4. Manually fill template

**Example search:**

```
"Semi-Supervised Classification with Graph Convolutional Networks" Kipf Welling
```

---

### Option 4: Manual (Semantic Scholar + DBLP)

**For verification or if AI struggles:**

1. **Semantic Scholar** (semanticscholar.org)

   - Search by title
   - Click "Cite" → "BibTeX"
   - Copy entry

2. **DBLP** (dblp.org)

   - Best for conference papers
   - Search by author + year
   - Click "export record" → "BibTeX"

3. **arXiv** (arxiv.org)
   - Use arXiv ID (e.g., 1706.03762)
   - Click "Export citation" → "BibTeX"
   - **Note:** Only use if official version not found

---

## Step-by-Step Workflow

### Phase 1: Critical Papers (1-7) - ~90 minutes

**Goal:** Get the 7 most important citations

1. Start with paper #1 (Kipf GCN)

   - Should be easiest (famous paper)
   - Use as test to verify AI understands format

2. Continue with papers 2-7

   - Batch in groups of 2-3
   - Verify each before continuing

3. Check quality:
   - [ ] All authors listed
   - [ ] DOI links work
   - [ ] Conference/journal name correct
   - [ ] No syntax errors in BibTeX

**Checkpoint:** If 7/7 found, continue. If <5/7, try different AI tool.

---

### Phase 2: Important Papers (8-12) - ~45 minutes

**Goal:** Get 4/5 of these (80% ok)

1. Papers 8-10 (ASTGCN, GAT, Bishop MDN)

   - MDN might be hardest (1994 tech report)
   - If Bishop not found, note as "missing"

2. Papers 11-12 (ChebNet, MTGNN)
   - Should be straightforward

**Checkpoint:** If ≥4/5 found, good enough. Move to Phase 3.

---

### Phase 3: Supporting Papers (13-15) - ~30 minutes

**Goal:** Nice to have, not critical

1. Paper 13 (GATv2 - 2022)
2. Paper 14 (Temporal Fusion - 2021)
3. Paper 15 (ARIMA book)

**Checkpoint:** Even 0/3 is ok if Phases 1-2 complete.

---

## Quality Checklist

Before submitting results, verify:

### BibTeX Syntax

```bibtex
@inproceedings{authorYEAR,  ← No spaces in key
  title={Full Title},        ← Capital letters preserved
  author={A and B and C},    ← Use "and" not commas
  booktitle={Conference},    ← Official name
  year={2023},               ← Number, no quotes
  pages={123--456},          ← Use -- for ranges
  doi={10.1234/xyz}          ← No https:// prefix
}
```

### Common Errors to Avoid

- ❌ `author={A, B, C}` → ✅ `author={A and B and C}`
- ❌ `booktitle="ICLR"` → ✅ `booktitle={International Conference on Learning Representations}`
- ❌ `pages={123-456}` → ✅ `pages={123--456}`
- ❌ Using arXiv when conference version exists
- ❌ Missing DOI when available

---

## Example Perfect Entry

```bibtex
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017},
  url={https://openreview.net/forum?id=SJU4ayYgl}
}
```

**What makes it perfect:**

- ✅ Key is `authorYEAR` format
- ✅ Full title with capitals preserved
- ✅ Both authors listed with "and"
- ✅ Full conference name with acronym
- ✅ Official ICLR URL (OpenReview)
- ✅ Compiles without errors

---

## Troubleshooting

### "AI can't find a paper"

**Solution:**

1. Try exact title in quotes on Google Scholar
2. Search by first author + year
3. Check if conference/year is wrong in our request
4. Look for arXiv version as last resort

### "Multiple versions exist"

**Priority:**

1. Official conference/journal (best)
2. arXiv with same title (ok)
3. Workshop version (avoid)
4. Preprint only (last resort)

**Example:** For Transformer paper:

- ✅ Use NeurIPS 2017 version
- ❌ Don't use arXiv-only version

### "BibTeX won't compile"

**Common fixes:**

1. Check for unescaped special characters: `&` → `\&`
2. Balance braces: `{Title}` not `{Title`
3. Remove line breaks inside fields
4. Check comma placement

**Test with:** [bibcheck.org](http://bibcheck.org/) or LaTeX compiler

---

## Time Estimates

| Phase              | Papers | Time       | Cumulative |
| ------------------ | ------ | ---------- | ---------- |
| Critical (1-7)     | 7      | 90 min     | 1.5 hrs    |
| Important (8-12)   | 5      | 45 min     | 2.25 hrs   |
| Supporting (13-15) | 3      | 30 min     | 2.75 hrs   |
| Verification       | -      | 30 min     | 3 hrs      |
| **Total**          | **15** | **~3 hrs** |            |

**Realistic:** 2.5-4 hours depending on:

- AI tool speed
- Paper availability
- Your BibTeX experience

---

## Success Criteria

**Minimum acceptable:**

- ✅ 7/7 Critical papers (100%)
- ✅ 3/5 Important papers (60%)
- ✅ 0/3 Supporting papers (ok to skip)
- **Total: 10/15 papers (67%)**

**Target:**

- ✅ 7/7 Critical (100%)
- ✅ 4/5 Important (80%)
- ✅ 2/3 Supporting (67%)
- **Total: 13/15 papers (87%)**

**Perfect:**

- ✅ 15/15 papers (100%)

---

## After Research Complete

1. **Save results** in `BIBTEX_RESULTS_TEMPLATE.md`
2. **Copy BibTeX entries** to `docs/final_report/11_references.md`
3. **Update report** with citation numbers [1], [2], etc.
4. **Test compilation** with LaTeX (Phase 6)

---

## Contact Points

**If stuck:**

1. Check `REFERENCE_COVERAGE_ANALYSIS.md` for alternatives
2. Mark paper as "NOT FOUND" in template
3. We can proceed with 10/15 papers minimum

**Questions about our project:**

- See `AI_RESEARCH_REQUEST.md` "Project Context" section
- We're comparing STMGT vs baselines (GCN, LSTM, GraphWaveNet)
- Need citations to support literature review

---

## Quick Start Command

**For Perplexity/Claude:**

```
I need help finding academic citations for my traffic forecasting project.

I'll paste two files:
1. AI_RESEARCH_REQUEST.md (what to find)
2. BIBTEX_RESULTS_TEMPLATE.md (where to put results)

Please start with Critical Papers #1-3, find their BibTeX entries,
and fill out the template. Let me know if any paper can't be found.

[Paste AI_RESEARCH_REQUEST.md]
[Paste BIBTEX_RESULTS_TEMPLATE.md sections for papers 1-3]
```

---

**Good luck with the research!**

Expected outcome: 10-15 high-quality BibTeX entries ready for LaTeX compilation.

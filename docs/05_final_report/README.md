# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Final Report - STMGT Traffic Forecasting System

This directory contains the final project report in LaTeX format (IEEE conference style) for the DSP391m Traffic Forecasting project.

## Quick Start

```bash
# Navigate to directory
cd docs/final_report

# Compile main document
pdflatex final_report_clean.tex
pdflatex final_report_clean.tex  # Run twice for cross-references

# Output: final_report_clean.pdf
```

## Project Structure

```
final_report/
├── final_report_clean.tex          # Main LaTeX document ✅
├── BUILD_GUIDE.md                  # Detailed compilation guide ✅
├── STATUS.md                       # Conversion progress tracker ✅
├── README.md                       # This file ✅
│
├── sections/                       # Modular LaTeX sections
│   ├── README.md                   # Section guide ✅
│   ├── 01_introduction.tex         # ✅ DONE
│   ├── 02_literature_review.tex    # ✅ DONE
│   ├── 03_data_description.tex     # ✅ DONE
│   ├── 04_data_preprocessing.tex   # ⏳ TODO
│   ├── 05_eda.tex                  # ⏳ TODO
│   ├── 06_methodology.tex          # ⏳ TODO (CRITICAL)
│   ├── 07_model_development.tex    # ⏳ TODO (CRITICAL)
│   ├── 08_evaluation.tex           # ⏳ TODO
│   ├── 09_results.tex              # ⏳ TODO (CRITICAL)
│   └── 10_conclusion.tex           # ⏳ TODO
│
├── figures/                        # Figure files (.png, .pdf)
│   └── ... (to be populated)
│
├── backup_old_tex/                 # Old LaTeX versions (backup)
│   └── ...
│
└── [Markdown sources]              # Original markdown files
    ├── 01_title_team_intro.md
    ├── 02_literature_review.md
    ├── 03_data_description.md
    ├── ... (12 files total)
```

## Demo Workflow Update

- Demo script now supports optional static map (Figure 3) via `--include-map` flag; map is not required for main results or presentation.
- Google baseline metrics use real data if available (`duration_in_traffic`, `distance_km`, or `google_speed_kmh`); otherwise, synthetic baseline is used for comparison figures.
- CLI usage and output files updated; see demo README for details.

## Current Status

**Progress:** 3/10 sections completed (30%)

| Section               | Status  | Priority     |
| --------------------- | ------- | ------------ |
| 01 Introduction       | ✅ Done | HIGH         |
| 02 Literature Review  | ✅ Done | HIGH         |
| 03 Data Description   | ✅ Done | HIGH         |
| 04 Data Preprocessing | ⏳ TODO | HIGH         |
| 05 EDA                | ⏳ TODO | HIGH         |
| 06 Methodology        | ⏳ TODO | **CRITICAL** |
| 07 Model Development  | ⏳ TODO | **CRITICAL** |
| 08 Evaluation         | ⏳ TODO | HIGH         |
| 09 Results            | ⏳ TODO | **CRITICAL** |
| 10 Conclusion         | ⏳ TODO | HIGH         |

See [STATUS.md](STATUS.md) for detailed progress tracking.

## Key Files

### 📄 final_report_clean.tex

**Main LaTeX document** - Uses `\input{}` to include modular sections. This is the file you compile.

**Features:**

- IEEE conference format
- Modular architecture (sections loaded from `sections/` directory)
- Complete abstract and introduction
- 17 references included
- Ready to compile (with placeholder content for incomplete sections)

### 📚 sections/

**Individual section files** - Each section is a separate `.tex` file for easy editing and version control.

**Benefits:**

- Easy to find and edit specific content
- Clean git diffs (track changes per section)
- Can work on sections in parallel
- Reusable in presentations

### 📖 BUILD_GUIDE.md

**Comprehensive build instructions** - How to compile, troubleshoot, and create new sections.

**Covers:**

- Prerequisites and LaTeX installation
- Compilation methods (pdflatex, latexmk, VS Code)
- Creating new sections
- Common LaTeX commands
- Troubleshooting guide

### 📊 STATUS.md

**Detailed progress tracker** - Section-by-section conversion status with content summaries.

**Includes:**

- Completion status for each section
- Content outlines
- Estimated line counts
- Conversion notes and special considerations
- Timeline and quality checklist

## Documentation Files

### Source Markdown Files (Original Content)

| File                        | Lines | Description                       |
| --------------------------- | ----- | --------------------------------- |
| 01_title_team_intro.md      | 152   | Title, team, introduction         |
| 02_literature_review.md     | 386   | Literature review (60+ papers)    |
| 03_data_description.md      | 395   | Data sources, statistics          |
| 04_data_preprocessing.md    | 161   | Cleaning, normalization, graph    |
| 05_eda.md                   | 150   | EDA, patterns, correlations       |
| 06_methodology.md           | 258   | Model selection, features         |
| 07_model_development.md     | 486   | STMGT architecture details        |
| 08_evaluation_tuning.md     | 309   | Metrics, tuning, ablation         |
| 09_results_visualization.md | 503   | Results, comparisons, analysis    |
| 10_conclusion.md            | 476   | Summary, limitations, future work |
| 11_references.md            | 245   | Bibliography and citations        |
| 12_appendices.md            | 810   | Additional details, code          |

**Total:** ~3,700+ lines of markdown content

## Architecture Overview

### Modular Design Philosophy

**Problem:** Single large LaTeX file is hard to:

- Navigate and edit
- Track changes in version control
- Collaborate on with multiple people
- Maintain and update

**Solution:** Modular architecture with `\input{}`

```latex
% In final_report_clean.tex
\input{sections/01_introduction}
\input{sections/02_literature_review}
% ...etc
```

**Benefits:**

1. Each section is self-contained
2. Easy to locate and edit specific content
3. Clean git diffs (changes show up per section)
4. Can work on multiple sections in parallel
5. Sections are reusable in other documents

### Compilation Process

```
[01_introduction.tex]      ┐
[02_literature_review.tex] │
[03_data_description.tex]  │
[04_data_preprocessing.tex]├──> final_report_clean.tex ──> pdflatex ──> final_report_clean.pdf
[...]                      │
[10_conclusion.tex]        │
[References in main file]  ┘
```

## How to Contribute

### Adding a New Section

1. **Create section file** in `sections/` directory:

   ```bash
   touch sections/04_data_preprocessing.tex
   ```

2. **Add header comment:**

   ```latex
   % Section 4: Data Preprocessing
   % Maintainer: THAT Le Quang (thatlq1812)
   % Source: 04_data_preprocessing.md
   ```

3. **Convert markdown to LaTeX:**

   - Use templates from existing sections
   - Follow IEEE format guidelines
   - See [BUILD_GUIDE.md](BUILD_GUIDE.md) for conversion tips

4. **Update main file:**

   ```latex
   % In final_report_clean.tex
   \input{sections/04_data_preprocessing}
   ```

5. **Test compilation:**

   ```bash
   pdflatex final_report_clean.tex
   ```

6. **Update STATUS.md:**
   - Mark section as complete
   - Add any notes or special considerations

### Editing an Existing Section

1. **Open section file** (e.g., `sections/01_introduction.tex`)
2. **Make changes**
3. **Test compilation** to ensure no errors
4. **Commit changes** with descriptive message

### LaTeX Best Practices

- Use `\cite{}` for citations
- Use `\ref{}` for cross-references (figures, tables, equations)
- Label everything: `\label{fig:name}`, `\label{tab:name}`, `\label{eq:name}`
- Use IEEE-style formatting
- Comment complex LaTeX code
- Keep lines under 100 characters for readability

## Format Specification

**Conference:** IEEE Conference Format (IEEEtran class)  
**Style:** Two-column, 10pt font  
**Page Limit:** Typically 6-8 pages (adjustable)  
**Citation:** IEEE numerical style `[1], [2], [3]`

### Document Structure

1. **Title and Authors**
2. **Abstract** (150-250 words)
3. **Keywords** (5-7 terms)
4. **Main Sections:**
   - Introduction
   - Literature Review
   - Data Description
   - Data Preprocessing
   - Exploratory Data Analysis
   - Methodology
   - Model Development
   - Evaluation and Tuning
   - Results and Visualization
   - Conclusion
5. **Acknowledgment**
6. **References** (17 citations)
7. **Appendices** (optional)

## Requirements

### Software

**LaTeX Distribution:**

- Windows: MiKTeX or TeX Live
- macOS: MacTeX
- Linux: TeX Live

**Editor (Optional but Recommended):**

- VS Code with LaTeX Workshop extension
- TeXstudio
- Overleaf (online)

### LaTeX Packages

All required packages are included in `final_report_clean.tex`:

- `IEEEtran` (document class)
- `cite`, `amsmath`, `graphicx`, `booktabs`, `hyperref`
- `listings` (code formatting)
- `algorithm`, `algorithmic` (algorithms)
- `subcaption`, `float` (figures)

## Common Tasks

### Compile the Document

```bash
cd docs/final_report
pdflatex final_report_clean.tex
pdflatex final_report_clean.tex  # Run twice
```

### View the PDF

```bash
# Windows
start final_report_clean.pdf

# macOS
open final_report_clean.pdf

# Linux
xdg-open final_report_clean.pdf
```

### Clean Build Artifacts

```bash
rm *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz
```

### Check Status

```bash
cat STATUS.md  # View detailed progress
cat BUILD_GUIDE.md  # View compilation guide
```

## Troubleshooting

### "File not found" error

**Problem:** Section file doesn't exist  
**Solution:** Comment out `\input{}` line or create the file

```latex
% \input{sections/04_data_preprocessing}  % TODO: Create this
```

### Missing references

**Problem:** Citations show as `[?]`  
**Solution:** Run pdflatex twice

```bash
pdflatex final_report_clean.tex
pdflatex final_report_clean.tex
```

### Figures not displaying

**Problem:** Image files missing or wrong path  
**Solution:**

- Create `figures/` directory
- Add placeholder images
- Or use `draft` option: `\includegraphics[draft,width=0.8\columnwidth]{fig.png}`

### Package not found

**Problem:** LaTeX package missing  
**Solution:** Install package via package manager

```bash
# MiKTeX
mpm --install=<package-name>

# TeX Live
tlmgr install <package-name>
```

## Resources

### LaTeX References

- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX)
- [Overleaf Documentation](https://www.overleaf.com/learn)
- [IEEE Template](https://www.ieee.org/conferences/publishing/templates.html)
- [Tables Generator](https://www.tablesgenerator.com/)

### Project Documentation

- [BUILD_GUIDE.md](BUILD_GUIDE.md) - Compilation instructions
- [STATUS.md](STATUS.md) - Progress tracker
- [sections/README.md](sections/README.md) - Section guide
- [CHANGELOG.md](../../CHANGELOG.md) - Project changelog

### External Links

- [STMGT Project Repository](https://github.com/thatlq1812/dsp391m_project)
- [IEEE Format Guidelines](https://www.ieee.org/publications/)
- [LaTeX Stack Exchange](https://tex.stackexchange.com/)

## Timeline

**Started:** November 12, 2025  
**Current Phase:** Section creation (3/10 done)  
**Target Completion:** December 9, 2025 (4 weeks)

### Milestones

- [x] Week 1: Structure and sections 01-03 (DONE)
- [ ] Week 2: Sections 04-06 (methodology critical)
- [ ] Week 3: Sections 07-09 (model development and results)
- [ ] Week 4: Section 10, formatting, final review

## Team

**Maintainer:** THAT Le Quang  
**Role:** AI & DS Major Student  
**Email:** thatlq1812@fpt.edu.vn  
**GitHub:** [@thatlq1812](https://github.com/thatlq1812)

## License

This documentation is part of the DSP391m Traffic Forecasting System project.

---

**Last Updated:** November 12, 2025  
**Version:** 1.0  
**Status:** In Progress (30% complete)

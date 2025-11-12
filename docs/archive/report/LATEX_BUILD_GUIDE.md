# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# LaTeX Report Build Guide

This directory contains the LaTeX source for the STMGT Traffic Forecasting final report.

## 📁 Directory Structure

```
docs/final_report/
├── final_report.tex          # Main LaTeX document
├── references.bib            # Bibliography database (17 references)
├── build.sh                  # Build script (Linux/Mac)
├── build.bat                 # Build script (Windows)
├── sections/                 # Section files (to be created)
│   ├── 01_introduction.tex
│   ├── 02_literature_review.tex
│   ├── 03_data_description.tex
│   ├── 04_data_preprocessing.tex
│   ├── 05_exploratory_data_analysis.tex
│   ├── 06_methodology.tex
│   ├── 07_model_development.tex
│   ├── 08_evaluation_tuning.tex
│   ├── 09_results_visualization.tex
│   ├── 10_conclusion.tex
│   └── 12_appendices.tex
├── figures/                  # 15 PNG figures (already exist)
│   ├── fig01_speed_distribution.png
│   ├── fig02_network_topology.png
│   └── ... (13 more figures)
└── 0X_*.md                   # Source markdown files (12 files)
```

## 🔧 Prerequisites

### Required Software

1. **LaTeX Distribution** (choose one):

   - **Windows:** [MiKTeX](https://miktex.org/download) (recommended)
   - **Linux:** TeX Live (`sudo apt install texlive-full`)
   - **Mac:** MacTeX (`brew install mactex`)

2. **Biber** (bibliography processor):
   - Usually included with TeX Live/MiKTeX
   - If missing: `tlmgr install biber` (TeX Live) or MiKTeX Package Manager

### Required LaTeX Packages

These will be auto-installed by MiKTeX or included in TeX Live full:

- `biblatex` - Modern bibliography
- `graphicx` - Figure inclusion
- `hyperref` - PDF hyperlinks
- `booktabs` - Professional tables
- `listings` - Code formatting
- `amsmath`, `amssymb` - Math symbols
- `geometry` - Page layout
- `fancyhdr` - Headers/footers

## 🚀 Quick Start

### Option 1: Automated Build (Recommended)

**Windows:**

```cmd
build.bat
```

**Linux/Mac:**

```bash
chmod +x build.sh
./build.sh
```

### Option 2: Manual Build

```bash
pdflatex final_report.tex
biber final_report
pdflatex final_report.tex
pdflatex final_report.tex
```

**Why 3 passes?**

1. First pass: Generate `.aux` files with references
2. Biber: Process bibliography
3. Second pass: Resolve citations and cross-references
4. Third pass: Finalize all references

## 📝 Next Steps

### 1. Convert Markdown to LaTeX

You have 12 markdown files that need to be converted to LaTeX format. Two approaches:

**Approach A: Automated with Pandoc**

```bash
# Install pandoc
# Windows: choco install pandoc
# Linux: sudo apt install pandoc

# Convert each section
for i in {01..12}; do
    pandoc "0${i}_*.md" -o "sections/0${i}_*.tex" --wrap=preserve
done
```

**Approach B: Manual Conversion** (More control)

```bash
# Create each section file in sections/
# Copy content from markdown and adjust:
# - Change ## to \section{}, ### to \subsection{}
# - Convert ![](figures/figXX.png) to \begin{figure}...\end{figure}
# - Convert tables to LaTeX tabular format
# - Add \cite{} for references
```

### 2. Figure Path Verification

Figures are already in `figures/` directory with correct names. LaTeX will reference them as:

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/fig01_speed_distribution.png}
    \caption{Traffic Speed Distribution}
    \label{fig:speed_distribution}
\end{figure}
```

### 3. Citation Format

Replace markdown citations `[1]` with LaTeX `\cite{hochreiter1997long}`:

- `[1]` → `\cite{hochreiter1997long}`
- `[2]` → `\cite{bishop1994mixture}`
- `[5]` → `\cite{kipf2017semi}`
- `[1,2,3]` → `\cite{hochreiter1997long,bishop1994mixture,gneiting2007strictly}`

## 📊 Current Status

✅ **Completed:**

- Main LaTeX template (`final_report.tex`)
- Bibliography database (`references.bib` with 17 entries)
- Build scripts (Windows + Linux/Mac)
- Figure files (15 PNGs)
- Markdown source (12 sections, 81% complete)

⏳ **TODO:**

- Convert markdown sections to LaTeX (sections/0X\_\*.tex)
- Adjust figure references to LaTeX format
- Replace markdown citations with \cite{}
- Test build and fix any LaTeX errors

## 🎨 Customization

### Change Page Margins

Edit `final_report.tex`:

```latex
\geometry{
    left=2.5cm,   % Change this
    right=2.5cm,  % Change this
    top=3cm,
    bottom=3cm
}
```

### Change Font Size

```latex
\documentclass[11pt,a4paper]{article}  % Change 12pt to 11pt or 10pt
```

### Change Title/Author

Edit the `\title{}` and `\author{}` sections in `final_report.tex`.

## 🐛 Troubleshooting

### Error: "File not found"

- Check that `sections/` directory exists
- Verify all `.tex` files are created
- Check figure paths in `figures/` directory

### Bibliography not appearing

- Ensure `biber` is installed
- Run full build sequence (3 passes)
- Check `final_report.blg` for biber errors

### Missing packages

- **MiKTeX:** Will auto-install on first build
- **TeX Live:** Install with `tlmgr install <package>`

### Figures not showing

- Verify PNG files exist in `figures/` directory
- Check file names match exactly (case-sensitive on Linux)
- Ensure `\graphicspath{{figures/}}` is set (already in template)

## 📚 Resources

- [LaTeX Documentation](https://www.latex-project.org/help/documentation/)
- [Overleaf Guides](https://www.overleaf.com/learn)
- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX)
- [BibLaTeX Documentation](https://ctan.org/pkg/biblatex)

## 📈 Expected Output

**Final PDF specs:**

- Format: A4, single-column
- Pages: ~30-40 pages (estimated)
- Sections: 12 (including appendices)
- Figures: 15 PNG images
- References: 17 academic citations
- File size: ~3-5 MB (with high-res figures)

---

**Status:** LaTeX template ready. Proceed with markdown → LaTeX conversion.

**Last Updated:** November 12, 2025

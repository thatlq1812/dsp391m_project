# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# LaTeX Sections Directory

This directory contains modular LaTeX sections for the final report. Each section is a separate `.tex` file that is included in the main `final_report.tex` using `\input{}` commands.

## Structure

```
sections/
├── 01_introduction.tex          ✅ Created
├── 02_literature_review.tex     ✅ Created
├── 03_data_description.tex      ✅ Created
├── 04_data_preprocessing.tex    ⏳ To be created
├── 05_eda.tex                   ⏳ To be created
├── 06_methodology.tex           ⏳ To be created
├── 07_model_development.tex     ⏳ To be created
├── 08_evaluation.tex            ⏳ To be created
├── 09_results.tex               ⏳ To be created
├── 10_conclusion.tex            ⏳ To be created
└── README.md                    ✅ This file
```

## Mapping from Markdown Sources

| LaTeX Section             | Source Markdown             | Status  |
| ------------------------- | --------------------------- | ------- |
| 01_introduction.tex       | 01_title_team_intro.md      | ✅ Done |
| 02_literature_review.tex  | 02_literature_review.md     | ✅ Done |
| 03_data_description.tex   | 03_data_description.md      | ✅ Done |
| 04_data_preprocessing.tex | 04_data_preprocessing.md    | ⏳ TODO |
| 05_eda.tex                | 05_eda.md                   | ⏳ TODO |
| 06_methodology.tex        | 06_methodology.md           | ⏳ TODO |
| 07_model_development.tex  | 07_model_development.md     | ⏳ TODO |
| 08_evaluation.tex         | 08_evaluation_tuning.md     | ⏳ TODO |
| 09_results.tex            | 09_results_visualization.md | ⏳ TODO |
| 10_conclusion.tex         | 10_conclusion.md            | ⏳ TODO |

## Usage

### In main `final_report.tex`:

```latex
% Include a section
\input{sections/01_introduction}
\input{sections/02_literature_review}
% ... etc
```

### Compiling the Report

```bash
cd docs/final_report
pdflatex final_report.tex
pdflatex final_report.tex  # Run twice for cross-references
```

## Benefits of This Structure

1. **Modularity:** Each section can be edited independently
2. **Maintainability:** Easy to find and update specific content
3. **Collaboration:** Multiple people can work on different sections
4. **Version Control:** Git diffs are cleaner for individual sections
5. **Reusability:** Sections can be reused in presentations or other documents

## Creating New Sections

To create a new section file:

1. Copy the header comment from existing sections
2. Start with `\section{Title}` (don't use `\documentclass`, `\begin{document}`, etc.)
3. Use `\subsection{}` and `\subsubsection{}` for hierarchy
4. Add to main `final_report.tex` with `\input{sections/XX_name}`

## LaTeX Style Guidelines

- Use IEEE format commands: `\cite{}`, `\ref{}`, `\label{}`
- Table and figure labels: `tab:name`, `fig:name`
- Equations: Use `\begin{equation}...\end{equation}` for numbered equations
- Lists: Use `\begin{itemize}` or `\begin{enumerate}`
- Bold: `\textbf{text}`, Italic: `\textit{text}`
- Code: `\texttt{code}` or `\begin{lstlisting}...\end{lstlisting}`

## References Management

All references are managed in the main `final_report.tex` file using:

```latex
\begin{thebibliography}{99}
\bibitem{ref_key}
Author et al., "Title", Journal, Year.
\end{thebibliography}
```

Cite in sections using: `\cite{ref_key}`

## Notes

- Figures should reference `figures/figXX_name.png`
- All paths are relative to the `final_report.tex` location
- Compile from the `docs/final_report/` directory

## TODO List

- [ ] Create sections 04-10 from markdown sources
- [ ] Add figure placeholders with proper captions
- [ ] Complete references section with full citations
- [ ] Create appendices section
- [ ] Verify all cross-references work
- [ ] Final formatting and IEEE compliance check

---

## Demo Figure Notes

- Figure 3 (static map) is now optional and generated only if `--include-map` is set in the demo script. It is not required for main results or PowerPoint.
- Google baseline metrics use real data columns if present; otherwise, synthetic values are used for comparison figures.

Last Updated: November 12, 2025

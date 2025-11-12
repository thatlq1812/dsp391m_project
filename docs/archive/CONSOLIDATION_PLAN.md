# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Documentation Consolidation Plan

**Date:** November 12, 2025
**Status:** Planning Phase

## Problem Statement

Current documentation structure is **fragmented and overwhelming**:

- 50+ documentation files across multiple directories
- Duplicate information (multiple setup guides, training guides)
- Unclear hierarchy (what to read first?)
- Mix of active docs and historical artifacts
- Hard to find essential information

## Goals

1. **Reduce to ~15 essential docs** (70% reduction)
2. **Clear hierarchy** - beginner → advanced
3. **Single source of truth** for each topic
4. **Easy navigation** via master README
5. **Preserve history** in archive

---

## Current Structure Analysis

### ✅ Keep (Essential - 14 files)

**Root Level:**

- `README.md` - Master entry point (to be rewritten)
- `TRAINING_WORKFLOW.md` - Complete training pipeline
- `CHANGELOG.md` (in docs/) - Project history

**Core Documentation (`docs/`):**

1. `API_DOCUMENTATION.md` - API reference
2. `MODEL_VALUE_AND_LIMITATIONS.md` - Model assessment
3. `V3_DESIGN_RATIONALE.md` - Design decisions

**User Guides (`docs/guides/`):** 4. `CLI_USER_GUIDE.md` - CLI usage 5. `TRAINING_GUIDE.md` - Training instructions 6. `DEPLOYMENT.md` - Production deployment 7. `weather_data_explained.md` - Data concepts

**Development (`docs/`):** 8. `AUGMENTATION_MIGRATION_GUIDE.md` - Aug system 9. `data_leakage_fix.md` (in fix/) - Critical fix doc

**Architecture (`docs/architecture/`):** 10. `STMGT_ARCHITECTURE.md` - Model design 11. `STMGT_DATA_IO.md` - Data pipeline

**Research:** 12. `STMGT_RESEARCH_CONSOLIDATED.md` - Research summary

**Reports:** 13. Final report (when ready)

### 📦 Archive (Redundant/Outdated - ~35 files)

**Setup Guides (Consolidate → README):**

- `docs/guides/README_SETUP.md` → Merge to root README
- `docs/guides/CLI_GITBASH_GUIDE.md` → Merge to CLI_USER_GUIDE

**Training Guides (Consolidate → TRAINING_WORKFLOW.md):**

- `docs/guides/TRAINING_GUIDE.md` → Keep but link to TRAINING_WORKFLOW
- `docs/guides/CONFIG_TUNING_GUIDE.md` → Merge to TRAINING_GUIDE
- `docs/guides/DATA_LOADING_OPTIMIZATION.md` → Merge to TRAINING_GUIDE

**Architecture Analysis (Merge → STMGT_ARCHITECTURE.md):**

- `docs/architecture/STMGT_ARCHITECTURE_ANALYSIS.md` → Merge
- `docs/architecture/STMGT_MODEL_ANALYSIS.md` → Merge

**Reports (Keep only final):**

- `docs/report/RP3.md`, `RP3_ReCheck.md` → Archive (drafts)
- `docs/report/ROADMAP_REPORT3_TO_FINAL.md` → Archive (completed)
- `docs/report/REPORT_COMPLETION_TODO.md` → Archive (completed)

**Instructions (Archive - phases completed):**

- `docs/instructions/PHASE1_WEB_MVP.md` → Archive
- `docs/instructions/PHASE2_MODEL_IMPROVEMENTS.md` → Archive
- `docs/instructions/PHASE3_PRODUCTION.md` → Keep for deployment
- `docs/instructions/PHASE4_EXCELLENCE.md` → Archive

**Misc (Archive):**

- `docs/INDEX.md` → Replace with new README structure
- `docs/CLEANUP_SUMMARY.md` → Already in CHANGELOG
- `docs/research/AUTO_EXPANSION_SYSTEM.md` → Archive (not implemented)
- `docs/guides/API_TESTING_GUIDE.md` → Merge to API_DOCUMENTATION
- `docs/guides/VM_CONFIG_INTEGRATION.md` → Merge to DEPLOYMENT

### 🗑️ Delete (Duplicate/Obsolete)

- All `desktop.ini` files
- `docs/report/figures/` → Move to final report only

---

## Proposed New Structure

```
project/
├── README.md                          # 🌟 Master guide (NEW - comprehensive)
├── TRAINING_WORKFLOW.md               # Complete training pipeline
├── docs/
│   ├── CHANGELOG.md                   # Project history
│   ├── API.md                         # API reference (renamed)
│   ├── MODEL.md                       # Model overview (consolidated)
│   ├── ARCHITECTURE.md                # Technical design (consolidated)
│   ├── RESEARCH.md                    # Research summary
│   ├── DEPLOYMENT.md                  # Production guide (consolidated)
│   ├── TRAINING.md                    # Training guide (consolidated)
│   ├── DATA.md                        # Data concepts (consolidated)
│   ├── CLI.md                         # CLI reference (consolidated)
│   ├── AUGMENTATION.md                # Augmentation guide
│   ├── FIXES.md                       # Critical fixes (data leakage)
│   ├── final_report/                  # Final report only
│   └── archive/                       # Historical docs
│       ├── guides/                    # Old guides
│       ├── architecture/              # Old architecture
│       ├── research/                  # Old research
│       ├── report/                    # Draft reports
│       ├── instructions/              # Phase instructions
│       └── sessions/                  # Session logs
```

**Total: 12 active docs + 1 master README + final report = ~14 files**

---

## New Master README Structure

```markdown
# STMGT Traffic Forecasting System

## Quick Start (5 minutes)

- Installation
- First run
- Basic usage

## For Users

- [CLI Guide](docs/CLI.md) - Command-line interface
- [API Reference](docs/API.md) - REST API documentation
- [Deployment](docs/DEPLOYMENT.md) - Production setup

## For Developers

- [Training Workflow](TRAINING_WORKFLOW.md) - Complete pipeline
- [Training Guide](docs/TRAINING.md) - Advanced training
- [Architecture](docs/ARCHITECTURE.md) - System design
- [Data Guide](docs/DATA.md) - Data concepts
- [Augmentation](docs/AUGMENTATION.md) - Data augmentation

## Research & Documentation

- [Model Overview](docs/MODEL.md) - Model value and limitations
- [Research Summary](docs/RESEARCH.md) - Academic contribution
- [Final Report](docs/final_report/) - Complete project report

## Project Information

- [Changelog](docs/CHANGELOG.md) - Version history
- [Critical Fixes](docs/FIXES.md) - Important bug fixes
- [License](LICENSE) - MIT License
```

---

## Consolidation Actions

### Phase 1: Archive Old Docs (Safe)

Move to `docs/archive/`:

- `docs/guides/README_SETUP.md`
- `docs/guides/CLI_GITBASH_GUIDE.md`
- `docs/guides/CONFIG_TUNING_GUIDE.md`
- `docs/guides/DATA_LOADING_OPTIMIZATION.md`
- `docs/guides/API_TESTING_GUIDE.md`
- `docs/guides/VM_CONFIG_INTEGRATION.md`
- `docs/architecture/STMGT_ARCHITECTURE_ANALYSIS.md`
- `docs/architecture/STMGT_MODEL_ANALYSIS.md`
- `docs/research/AUTO_EXPANSION_SYSTEM.md`
- `docs/report/` (except final_report/)
- `docs/instructions/PHASE1_WEB_MVP.md`
- `docs/instructions/PHASE2_MODEL_IMPROVEMENTS.md`
- `docs/instructions/PHASE4_EXCELLENCE.md`
- `docs/INDEX.md` (old index)
- `docs/CLEANUP_SUMMARY.md`

### Phase 2: Create Consolidated Docs

**New files to create:**

1. **docs/CLI.md** - Consolidate:

   - `docs/guides/CLI_USER_GUIDE.md`
   - `docs/guides/CLI_GITBASH_GUIDE.md`

2. **docs/API.md** - Consolidate:

   - `docs/API_DOCUMENTATION.md`
   - `docs/guides/API_TESTING_GUIDE.md`

3. **docs/MODEL.md** - Consolidate:

   - `docs/MODEL_VALUE_AND_LIMITATIONS.md`
   - Key points from `V3_DESIGN_RATIONALE.md`

4. **docs/ARCHITECTURE.md** - Consolidate:

   - `docs/architecture/STMGT_ARCHITECTURE.md`
   - `docs/architecture/STMGT_ARCHITECTURE_ANALYSIS.md`
   - `docs/architecture/STMGT_MODEL_ANALYSIS.md`

5. **docs/TRAINING.md** - Consolidate:

   - `docs/guides/TRAINING_GUIDE.md`
   - `docs/guides/CONFIG_TUNING_GUIDE.md`
   - `docs/guides/DATA_LOADING_OPTIMIZATION.md`
   - Reference to `TRAINING_WORKFLOW.md`

6. **docs/DATA.md** - Consolidate:

   - `docs/guides/weather_data_explained.md`
   - `docs/architecture/STMGT_DATA_IO.md`
   - Data concepts and schemas

7. **docs/DEPLOYMENT.md** - Consolidate:

   - `docs/guides/DEPLOYMENT.md`
   - `docs/guides/VM_CONFIG_INTEGRATION.md`
   - `docs/instructions/PHASE3_PRODUCTION.md`

8. **docs/AUGMENTATION.md** - Consolidate:

   - `docs/guides/AUGMENTATION_MIGRATION_GUIDE.md`
   - `docs/guides/safe_augmentation_guide.md`

9. **docs/RESEARCH.md** - Rename:

   - `docs/research/STMGT_RESEARCH_CONSOLIDATED.md`

10. **docs/FIXES.md** - Consolidate:
    - `docs/fix/data_leakage_fix.md`
    - Other critical fixes

### Phase 3: Rewrite Master README

Create comprehensive root `README.md`:

- Quick start (installation, first run)
- Clear navigation to all docs
- Separate sections for users/developers/researchers
- Link to TRAINING_WORKFLOW.md prominently
- Include recent training results

### Phase 4: Update References

- Update all internal links
- Update CHANGELOG references
- Update training scripts comments

---

## Migration Safety

**Before consolidation:**

1. ✓ Already archived: sessions/, audits/, upgrade/
2. ✓ Git commit current state
3. Create backup: `docs_backup_20251112/`

**During consolidation:**

1. Move files (don't delete)
2. Create new consolidated docs
3. Test all links

**After consolidation:**

1. Verify no broken links
2. Update CHANGELOG with consolidation entry
3. Team review

---

## Timeline

- **Phase 1:** Archive (30 min) - Move old docs to archive
- **Phase 2:** Consolidate (2 hours) - Create new consolidated docs
- **Phase 3:** Master README (1 hour) - Rewrite root README
- **Phase 4:** Verify (30 min) - Check links, test navigation

**Total: ~4 hours**

---

## Expected Outcome

**Before:**

- 50+ docs, unclear structure
- Hard to find information
- Duplicate content

**After:**

- 14 essential docs, clear hierarchy
- Easy navigation from README
- Single source of truth
- All history preserved in archive

**User experience:**

```
User: "How do I train the model?"
→ README → TRAINING_WORKFLOW.md (complete step-by-step)
   or
→ README → docs/TRAINING.md (advanced options)
```

---

## Next Steps

1. Get approval for consolidation plan
2. Create backup and git commit
3. Execute Phase 1 (archive old docs)
4. Execute Phase 2 (create consolidated docs)
5. Execute Phase 3 (rewrite README)
6. Execute Phase 4 (verify links)
7. Update CHANGELOG

**Ready to proceed?**

# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Instructions Index

**Purpose:** Structured roadmap from current state (8.5/10) to excellence (10/10)

**Current Status:** Report 3/4 completion, production-ready MVP  
**Target:** Perfect score with publication-quality deliverables  
**Timeline:** 2-3 weeks to Final Delivery

---

## Execution Order

Execute phases sequentially. Each phase has clear success criteria.

### Phase 1: Web MVP Completion (3-4 days) CURRENT PRIORITY

**File:** `PHASE1_WEB_MVP.md`

**Goal:** Deploy functional Google Maps interface with real-time predictions

**Success Criteria:**

- [ ] Frontend displays 78 nodes on HCMC map
- [ ] Click node → see 3-hour forecast chart
- [ ] Color-coded markers (green/yellow/red by speed)
- [ ] <100ms API response time
- [ ] Working demo for Report 3 presentation

**Blockers:** None - backend ready, just need frontend

---

### Phase 2: Model Quality Assurance (1 week)

**File:** `PHASE2_MODEL_IMPROVEMENTS.md`

**Goal:** Eliminate overfitting risk, verify test metrics, achieve stable R²≥0.50

**Success Criteria:**

- [ ] Validation/test MAE difference <10%
- [ ] Cross-validation confirms R²=0.45-0.55
- [ ] Training curves show no overfitting
- [ ] Ablation study completed
- [ ] Model card documentation

**Blockers:** Need Phase 1 complete for baseline comparison

---

### Phase 3: Production Hardening (4-5 days)

**File:** `PHASE3_PRODUCTION.md`

**Goal:** Enterprise-grade API with caching, auth, monitoring

**Success Criteria:**

- [ ] Redis caching (15min TTL)
- [ ] API key authentication
- [ ] Rate limiting (100 req/min)
- [ ] Prometheus metrics + Grafana dashboard
- [ ] Load tested 1000 concurrent users
- [ ] Docker containerized

**Blockers:** Need Phase 2 model finalized

---

### Phase 4: Excellence Features (3-4 days)

**File:** `PHASE4_EXCELLENCE.md`

**Goal:** Publication-quality features for 10/10 score

**Success Criteria:**

- [ ] Automated retraining pipeline
- [ ] Model explainability (SHAP/attention viz)
- [ ] Confidence calibration plots
- [ ] Multi-horizon evaluation (15min, 1hr, 3hr)
- [ ] Comparison with commercial APIs (Google Traffic)
- [ ] Academic paper draft ready

**Blockers:** All previous phases complete

---

## Quick Reference

| Phase   | Duration | Priority  | Dependencies |
| ------- | -------- | --------- | ------------ |
| Phase 1 | 3-4 days | 🔴 HIGH   | None         |
| Phase 2 | 1 week   | 🟠 MEDIUM | Phase 1      |
| Phase 3 | 4-5 days | 🟡 MEDIUM | Phase 2      |
| Phase 4 | 3-4 days | 🟢 LOW    | Phase 1-3    |

**Total Estimated Time:** 2.5-3 weeks

---

## Current Issues to Fix Immediately

Before starting Phase 1, fix these quick wins:

1. **Fix duplicate header** in `docs/STMGT_RESEARCH_CONSOLIDATED.md` line 1-10
2. **Create `.env` file** with conda environment name
3. **Add requirements.txt content** to git
4. **Verify test/validation split** logic in data loaders

See each phase file for detailed tasks and acceptance criteria.

---

## Success Metrics (8.5 → 10/10)

| Category              | Current (8.5)         | Target (10.0)           | Gap            |
| --------------------- | --------------------- | ----------------------- | -------------- |
| **Model Performance** | R²=0.79 (suspicious)  | R²=0.50-0.55 (verified) | Verify metrics |
| **Code Quality**      | No errors, good docs  | +Test coverage 80%+     | Add tests      |
| **Production Ready**  | FastAPI basic         | +Auth, cache, monitor   | Phase 3        |
| **Research Quality**  | Consolidated research | +Ablation, comparison   | Phase 2        |
| **Demo Quality**      | Streamlit dashboard   | +Web MVP, video demo    | Phase 1        |
| **Documentation**     | Excellent structure   | +Model card, API docs   | Phase 4        |

---

## How to Use These Instructions

1. **Read this index first** - understand overall flow
2. **Start with Phase 1** - don't skip ahead
3. **Check off tasks** in each phase file as you complete them
4. **Update CHANGELOG.md** after each major milestone
5. **Test thoroughly** before moving to next phase
6. **Ask for help** if blocked >1 day on any task

Each phase file contains:

- Detailed task breakdown
- Code templates/snippets
- Testing strategies
- Common pitfalls
- Acceptance criteria

Let's build something excellent!

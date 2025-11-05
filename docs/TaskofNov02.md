# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Phase Plan – November 02, 2025

## Phase 0 – Codebase Modularization (Week 0-1)

- [x] Extracted STMGT logic into `traffic_forecast/models/stmgt/` with separate `model.py`, `train.py`, `evaluate.py`, `inference.py`, and `losses.py` modules for clearer ownership.
- [x] Introduced shared utilities in `traffic_forecast/core/` (`config_loader`, `artifacts`, `reporting`) and updated CLI/dashboard tooling to consume them.
- [x] Refreshed `scripts/training/train_stmgt.py`, dashboard training flows, and regression scaffolds to read/write the new configs and artifacts without breaking compatibility.
- [x] Documented architecture changes in `docs/STMGT_ARCHITECTURE.md` and added quick-start guidance for registering future models.

## Phase 1 – Model Hardening & Repository Cleanup (Week 1-2)

- [x] Finalized synthetic-to-production data bridge by restoring CLI parity, adding dataset validation to collection/train scripts, and shipping the refreshed `analyze_augmentation_strategy.py` report.
- [x] Recovered STMGT utility tests (`tests/test_stmgt_utils.py`) and ran targeted pytest; broader regression coverage for collectors remains scheduled.
- [x] Standardized configuration and registry layers via the rewritten `traffic_forecast/core/config_loader.py` and new Pydantic registry schema consumed across dashboard and training flows.
- [x] Audited dashboard training control, removing stale helpers and wiring it to the validated registry/data root helpers for consistent job launches.
- [ ] Next sprint: extend regression tests to ingestion pipelines and document validator usage inside `docs/WORKFLOW.md` once scripts stabilize.

## Phase 2 – Hosted Inference API (Week 2-3)

- Design FastAPI endpoints: `/predict`, `/model/runs`, `/health`, `/report` with API-key auth, logging, and rate limiting.
- Implement registry-driven inference loader (select model, restore checkpoint, run prediction pipeline).
- Containerize the service (Docker) and deploy to Google Cloud (Cloud Run or VM) together with managed model artifacts.
- Add pytest/contract tests and a lightweight CI pipeline for build & deploy.

## Phase 3 – Flutter Web Client (Week 3-4+)

- Bootstrap Flutter Web project that consumes the hosted API while internal control remains on the Streamlit dashboard.
- Build key views: logistics overview (ETA, congestion alerts), route-level forecast visualization, and route recommendation module scaffolding.
- Implement authentication, error handling, and caching for production use; deploy the web client to Firebase Hosting or GCP static hosting.

## Continuing Operations

- Maintain the existing Streamlit dashboard for internal model control, monitoring, and configuration.
- After Phase 0 completion, iterate on API deployment (Phase 1) and then move into the Flutter client build (Phase 2) for pilot evaluations with logistics partners.

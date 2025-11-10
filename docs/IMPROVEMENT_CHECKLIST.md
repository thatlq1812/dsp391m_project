# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Project Improvement Checklist

**Date:** November 10, 2025  
**Context:** Post-review action items and enhancement roadmap

---

## Completed (November 10, 2025)

### Code Cleanup

- [x] Removed duplicate `stmgt.py` (556 lines) - kept `stmgt/model.py` as canonical
- [x] Removed old `.bat` file (`run_v2_training.bat`)
- [x] Cleaned all `__pycache__` directories and `.pyc` files
- [x] Moved `setup_cli.py` to `scripts/`
- [x] Moved `start_api_simple.py` to `scripts/deployment/`

### Web Inference Enhancement

- [x] Implemented actual edge-specific prediction in `predictor.py`
- [x] Removed TODO placeholder at line 558
- [x] Added proper node-to-edge mapping for predictions
- [x] Added confidence intervals (80% CI) for edge predictions
- [x] Added current speed tracking for edges

---

## High Priority (Before Production Deployment)

### 1. Testing & Quality Assurance

#### Add Coverage Reporting

```bash
pip install pytest-cov
pytest --cov=traffic_forecast --cov-report=html --cov-report=term
```

**Target:** 80% coverage minimum

**Action Items:**

- [ ] Add pytest-cov to requirements.txt
- [ ] Configure coverage in pytest.ini
- [ ] Add coverage badge to README.md
- [ ] Document coverage targets in CONTRIBUTING.md

#### Expand Test Suite

- [ ] Add performance tests for API endpoints (response time <200ms)
- [ ] Add load testing (100 concurrent requests)
- [ ] Add edge case tests (invalid inputs, missing data)
- [ ] Add integration tests for full prediction pipeline
- [ ] Test edge prediction with real data

### 2. Configuration Management

#### Centralize Configuration

- [ ] Create unified `ProjectConfig` schema with Pydantic
- [ ] Validate all configs on startup
- [ ] Document all config fields in `docs/CONFIG_REFERENCE.md`
- [ ] Add config validation script: `scripts/maintenance/validate_configs.py`

**Files to consolidate:**

- `configs/project_config.yaml`
- `configs/train_normalized_v*.json`
- `configs/vm_config.json`
- `.env`

#### Environment Variables

- [ ] Document all env vars in `.env.example`
- [ ] Add validation for required env vars
- [ ] Add type hints for env var usage

### 3. API Production Readiness

#### Security

- [ ] Add API key authentication
- [ ] Implement rate limiting (100 req/min per IP)
- [ ] Add CORS whitelist configuration
- [ ] Add request logging middleware
- [ ] Add input sanitization

#### Performance

- [ ] Add response caching (Redis)
- [ ] Implement batch prediction endpoint
- [ ] Add model warmup on startup
- [ ] Optimize graph loading (lazy loading)

#### Monitoring

- [ ] Add Prometheus metrics endpoint
- [ ] Track prediction latency histogram
- [ ] Track error rates by endpoint
- [ ] Add health check with dependencies
- [ ] Log all predictions to file

### 4. Web Interface Enhancements

#### Real-time Features

- [ ] Add WebSocket support for live updates
- [ ] Implement auto-refresh every 15 minutes
- [ ] Add "Live Mode" toggle
- [ ] Show timestamp of last update

#### Visualization

- [ ] Add historical playback (timeline slider)
- [ ] Show prediction confidence as heatmap
- [ ] Add traffic flow animation
- [ ] Visualize attention weights on map
- [ ] Add chart for speed trends

#### User Experience

- [ ] Add loading states for all actions
- [ ] Add error messages with retry
- [ ] Add tooltips for all features
- [ ] Add keyboard shortcuts
- [ ] Add responsive mobile layout

---

## Medium Priority (Next 2-4 Weeks)

### 5. Model Interpretability

#### Visualization Tools

- [ ] Create attention weight visualizer
- [ ] Add feature importance analysis
- [ ] Show GNN propagation paths
- [ ] Add SHAP-like explanations
- [ ] Visualize uncertainty sources

**Script:** `tools/visualize_model_internals.py`

#### Debugging Tools

- [ ] Add prediction debugging mode
- [ ] Show which nodes influenced prediction
- [ ] Display temporal attention patterns
- [ ] Track prediction accuracy over time

### 6. Data Quality Monitoring

#### Real-time Monitoring

- [ ] Track data completeness (% expected records)
- [ ] Detect anomalies (IQR-based outlier detection)
- [ ] Check temporal consistency (gaps, duplicates)
- [ ] Monitor API costs per day
- [ ] Alert on quality drops (>10% missing data)

**Script:** `scripts/monitoring/data_quality_monitor.py`

#### Validation

- [ ] Add data validation on collection
- [ ] Implement schema versioning
- [ ] Add data lineage tracking
- [ ] Create data quality dashboard

### 7. Documentation Updates

#### API Documentation

- [ ] Add OpenAPI/Swagger UI
- [ ] Document all endpoints with examples
- [ ] Add authentication guide
- [ ] Create API client library (Python)

#### User Guides

- [ ] Create video tutorial for web interface
- [ ] Add troubleshooting FAQ
- [ ] Document common error codes
- [ ] Create admin guide

#### Code Documentation

- [ ] Add architecture diagrams
- [ ] Document all major classes
- [ ] Add type hints everywhere
- [ ] Generate API docs with Sphinx

---

## Low Priority (Future Enhancements)

### 8. Scaling for City-Wide Deployment

#### Architecture Changes

- [ ] Implement hierarchical GNN (district → city)
- [ ] Add global pooling layer for long-range propagation
- [ ] Increase GNN depth to 10-15 blocks
- [ ] Implement mini-batch training for large graphs

**Target:** 2,000+ nodes, 5,000+ edges

#### Infrastructure

- [ ] Add horizontal API scaling (load balancer)
- [ ] Implement model serving with TorchServe
- [ ] Add database for historical predictions
- [ ] Implement distributed training

### 9. Advanced Features

#### Machine Learning

- [ ] Implement online learning (model updates)
- [ ] Add A/B testing framework
- [ ] Implement model versioning
- [ ] Add automated retraining pipeline
- [ ] Track model drift

#### Traffic Intelligence

- [ ] Add incident detection
- [ ] Implement traffic pattern discovery
- [ ] Add anomaly alerting
- [ ] Predict congestion likelihood
- [ ] Suggest optimal departure times

#### User Features

- [ ] Add user accounts and saved routes
- [ ] Implement route notifications
- [ ] Add commute time tracking
- [ ] Create mobile app
- [ ] Add traffic report crowdsourcing

### 10. Research Extensions

#### Model Improvements

- [ ] Experiment with graph transformers
- [ ] Try diffusion models for uncertainty
- [ ] Implement attention visualization
- [ ] Add external factors (events, construction)
- [ ] Multi-task learning (speed + flow + density)

#### Evaluation

- [ ] Compare with commercial APIs (Google/Waze)
- [ ] Conduct user studies
- [ ] Measure real-world impact
- [ ] Publish research paper

---

## Technical Debt

### Code Organization

- [ ] Consolidate model registry implementations
- [ ] Clean up archive/ directory (document or remove)
- [ ] Standardize error handling patterns
- [ ] Remove deprecated functions
- [ ] Refactor long functions (>100 lines)

### Dependencies

- [ ] Audit and minimize dependencies
- [ ] Update to latest stable versions
- [ ] Remove TensorFlow (only use PyTorch)
- [ ] Pin all versions in requirements.txt

### Performance

- [ ] Profile slow functions
- [ ] Optimize data loading pipeline
- [ ] Reduce memory usage
- [ ] Cache expensive operations

---

## Metrics & Success Criteria

### Code Quality

- **Coverage:** 80% minimum
- **Lint Score:** 9.0/10 (pylint)
- **Type Coverage:** 90% (mypy)
- **Documentation:** All public APIs documented

### Performance

- **API Latency:** <200ms p95
- **Prediction Accuracy:** MAE <3.1 km/h
- **Model Size:** <100MB
- **Memory Usage:** <2GB per API instance

### Reliability

- **Uptime:** 99.5% (SLA)
- **Error Rate:** <0.1%
- **Data Quality:** >95% completeness
- **Model Accuracy:** No degradation >5% over 30 days

---

## Review Schedule

- **Weekly:** Test coverage, API performance, data quality
- **Bi-weekly:** Code review, documentation updates
- **Monthly:** Model performance, user feedback, security audit
- **Quarterly:** Architecture review, scaling plan, research roadmap

---

## Priority Matrix

```
High Impact + Quick Win:
- Add API authentication
- Implement response caching
- Add test coverage reporting

High Impact + Long Term:
- City-wide scaling
- Model interpretability
- Real-time monitoring

Low Impact + Quick Win:
- Clean technical debt
- Update documentation
- Fix minor bugs

Low Impact + Long Term:
- Mobile app
- Advanced research features
```

---

## Next Steps (This Week)

1. **Tuesday:** Add pytest-cov and reach 70% coverage
2. **Wednesday:** Implement API authentication and rate limiting
3. **Thursday:** Add Prometheus metrics
4. **Friday:** Create CONFIG_REFERENCE.md
5. **Weekend:** Test web interface with real users

---

## Notes

- Keep this checklist updated after each sprint
- Mark items complete with date: `[x] 2025-11-XX: Task description`
- Add new items as they emerge from reviews
- Prioritize based on production readiness
- Document decisions in CHANGELOG.md

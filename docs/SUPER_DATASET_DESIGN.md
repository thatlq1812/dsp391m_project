# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Super Dataset Design: 1-Year Traffic Simulation

**Purpose:** Create challenging evaluation dataset that prevents autocorrelation exploitation and tests true spatio-temporal learning.

**Target Size:** 1 year (52 weeks × 7 days × 144 timestamps/day = ~52,000 timestamps)

---

## Design Principles

### 1. Prevent Autocorrelation Exploitation

- Include non-stationary patterns (holidays, seasons)
- Add sudden disruptions (incidents, construction)
- Vary temporal patterns across weeks
- Force models to learn causality, not just correlation

### 2. Test Real-World Scenarios

- Rush hour vs off-peak dynamics
- Weekday vs weekend patterns
- Weather impact on traffic
- Special events (concerts, sports, festivals)
- Construction zones with changing patterns

### 3. Multi-Scale Temporal Dynamics

- Short-term: 10-minute fluctuations
- Medium-term: hourly patterns
- Long-term: weekly cycles
- Seasonal: monthly/quarterly trends

---

## Dataset Components

### A. Base Traffic Patterns (Realistic Foundation)

#### 1. Weekday Rush Hour Pattern

```
06:00-09:00: Morning rush (30-50 km/h → 15-25 km/h)
09:00-16:00: Midday stable (35-45 km/h)
16:00-19:00: Evening rush (30-50 km/h → 10-20 km/h)
19:00-23:00: Evening normal (40-50 km/h)
23:00-06:00: Night time (45-52 km/h)
```

#### 2. Weekend Leisure Pattern

```
06:00-09:00: Slow morning (40-50 km/h)
09:00-12:00: Shopping hours (30-40 km/h)
12:00-18:00: Leisure activities (25-35 km/h)
18:00-22:00: Dinner/entertainment (30-40 km/h)
22:00-06:00: Night time (45-52 km/h)
```

#### 3. Spatial Variations

- **District 1 (CBD):** Heavy morning/evening rush
- **District 7 (Residential):** Moderate all day
- **District 10 (Mixed):** Variable patterns
- **Highway edges:** Higher speeds, less congestion

### B. Disruption Events (Challenge Models)

#### 1. Traffic Incidents (Random, 2-5 per week)

```python
Types:
- Minor accident: Speed drop to 10-20 km/h for 30-60 minutes
- Major accident: Speed drop to 5-15 km/h for 1-3 hours
- Vehicle breakdown: Partial lane block, 20-30% speed reduction
- Recovery period: Gradual return to normal over 30-60 minutes

Spatial impact:
- Affected edge: 70-90% speed reduction
- Adjacent edges: 30-50% speed reduction
- Upstream edges (within 3 hops): 10-20% congestion
```

#### 2. Construction Zones (Long-term, 5-10 zones/year)

```python
Patterns:
- Duration: 2-8 weeks
- Active hours: 09:00-17:00 (weekdays only)
- Speed reduction: 40-60% during active hours
- Spillover: Adjacent edges affected 20-30%
- Location rotation: Different zones each quarter
```

#### 3. Weather Events (Seasonal, realistic)

```python
Rainy days (20% of year):
- Light rain: 10-15% speed reduction
- Heavy rain: 30-40% speed reduction
- Duration: 1-4 hours, mostly afternoon/evening

Foggy mornings (5% of year):
- 20-30% speed reduction
- Duration: 06:00-09:00
- Visibility-dependent recovery

Extreme heat (10% of year):
- Minimal traffic impact (<5%)
- Mainly behavioral (people stay home)
```

#### 4. Special Events (Monthly, 1-3 events)

```python
Large events (concerts, sports):
- Pre-event: 50-80% congestion increase near venue
- During event: Normal traffic (people already there)
- Post-event: 100-150% congestion spike for 1-2 hours

Festivals/holidays:
- City-wide 30-50% traffic reduction
- Tourist areas: 50-80% increase
- Duration: 1-3 days
```

#### 5. Public Holidays (Vietnamese Calendar)

```python
Major holidays (Tet, National Day, etc.):
- 3-7 days: 60-80% overall traffic reduction
- Pre-holiday: 20-30% increase (shopping, travel prep)
- Post-holiday: 20-30% increase (return travel)

Long weekends:
- Friday evening: 40-60% increase (travel start)
- Sunday evening: 40-60% increase (return)
- Saturday/Sunday: 30-40% CBD reduction
```

### C. Seasonal Patterns (Long-term Trends)

#### 1. School Calendar Impact

```python
School year (Sep-May):
- Morning rush 20-30% more intense
- Afternoon pickup time (15:00-16:00) congestion
- School zones heavily affected

Summer break (Jun-Aug):
- 15-25% overall traffic reduction
- Tourist areas increase
- Different temporal patterns
```

#### 2. Economic Cycles

```python
Month-end/start:
- Increased traffic Tuesday-Thursday (shopping, services)
- 10-15% more congestion

Quarter-end:
- Business district 15-20% busier
- Extended rush hours
```

#### 3. Cultural Events (Vietnamese Context)

```python
Tet preparation (2 weeks before):
- Gradual traffic increase (shopping)
- Peak 2-3 days before Tet (50-80% increase)

Tet holiday (1 week):
- 70-90% traffic reduction
- Tourist sites increase

Post-Tet (1 week):
- Gradual recovery to normal
```

---

## Implementation Strategy

### Phase 1: Base Pattern Generation (Week 1)

**Tools:** Synthetic generation with noise

```python
Components:
1. Generate 52 weeks of base patterns
   - Weekday template (Mon-Fri)
   - Weekend template (Sat-Sun)
   - Node-specific multipliers (spatial variation)
   - Time-of-day functions (temporal variation)

2. Add realistic noise
   - Gaussian noise: σ = 2-5 km/h
   - Temporal smoothing (moving average)
   - Spatial correlation (neighboring edges)
```

### Phase 2: Seasonal Overlay (Week 1)

**Approach:** Multiplicative factors

```python
Seasonal components:
1. School calendar (binary mask × intensity)
2. Weather patterns (probabilistic, region-specific)
3. Economic cycles (periodic functions)
4. Holiday calendar (predefined dates)
```

### Phase 3: Event Injection (Week 2)

**Strategy:** Controlled randomness

```python
Event placement:
1. Random sampling (time, location, severity)
2. Constraints:
   - No overlapping major events
   - Realistic frequency (not every day)
   - Spatial distribution (different areas)
3. Realistic progression:
   - Build-up period
   - Peak impact
   - Recovery/decay

Implementation:
- Incidents: Poisson process (λ = 3/week)
- Construction: Fixed schedule with rotation
- Weather: Conditional probability (seasonal)
- Special events: Calendar-based with randomness
```

### Phase 4: Spatial Propagation (Week 2)

**Method:** Graph-based diffusion

```python
Propagation rules:
1. Direct impact: Affected edge (70-90% reduction)
2. 1-hop neighbors: 30-50% of impact
3. 2-hop neighbors: 10-20% of impact
4. 3+ hops: Minimal (<5%)

Factors:
- Edge direction (upstream vs downstream)
- Edge capacity (highway vs local road)
- Time of day (rush hour = more propagation)
```

### Phase 5: Validation & Quality Control (Week 3)

**Checks:**

```python
Statistical validation:
1. Speed distribution realistic (3-52 km/h)
2. No negative speeds
3. Temporal smoothness (no sudden jumps without events)
4. Spatial consistency (neighbors correlated)

Pattern validation:
1. Rush hour patterns present
2. Weekend vs weekday difference
3. Holiday effects visible
4. Event impacts realistic

Autocorrelation validation:
1. Lag-12 correlation < 0.95 (not too predictable)
2. Lag-144 (1 day) shows weekly pattern
3. Lag-1008 (1 week) correlation moderate
```

---

## Technical Specifications

### Data Schema

```python
DataFrame columns:
- timestamp: datetime64 (10-min intervals, 52,000 rows)
- node_a_id, node_b_id: str (edge identifier)
- speed_kmh: float (3-52 range)
- is_incident: bool (incident flag)
- is_construction: bool (construction flag)
- weather_condition: str (clear/rain/fog)
- is_holiday: bool (holiday flag)
- event_type: str (null/concert/sports/festival)
- temperature_c: float (25-35 range)
- precipitation_mm: float (0-50 range)

Metadata:
- edges: 144 edges (same topology)
- nodes: 62 nodes (HCMC road network)
- total_samples: ~7.5M rows (52,000 timestamps × 144 edges)
```

### File Structure

```
data/processed/
├── super_dataset_1year.parquet (main file, ~500MB)
├── super_dataset_metadata.json (events, holidays, construction)
├── super_dataset_statistics.json (validation metrics)
└── super_dataset_splits.json (train/val/test boundaries)
```

### Train/Val/Test Split

```python
Strategy: Temporal split with gap

Train: Months 1-8 (35 weeks, 67%)
  - Include full seasonal cycle
  - All event types represented
  - Construction zone variety

Gap: Week 36-37 (2 weeks)
  - Prevent information leakage
  - Simulate deployment delay

Val: Months 9-10 (9 weeks, 17%)
  - Different seasonal phase
  - New event instances
  - Overlapping construction zones

Test: Months 11-12 (8 weeks, 16%)
  - Year-end patterns
  - Holiday season (challenging)
  - Fresh event combinations
```

---

## Challenging Scenarios

### Scenario 1: Cascading Failures

```python
Trigger: Major incident during rush hour
Effect:
- Primary edge: 80% speed drop
- 3 adjacent edges: 50% drop (spillover)
- 8 upstream edges: 20-30% drop (congestion wave)
- Duration: 2 hours (incident) + 1 hour (recovery)

Challenge: Model must learn spatial propagation
```

### Scenario 2: Weather + Holiday Interaction

```python
Situation: Heavy rain on holiday travel day
Effect:
- Base holiday: -60% traffic (people stay home)
- Rain impact: -30% speed (those who travel)
- Combined: Non-linear interaction

Challenge: Model must learn feature interactions
```

### Scenario 3: Construction Zone Adaptation

```python
Pattern:
- Week 1-4: Edge A blocked (traffic reroutes to B)
- Week 5-8: Edge B blocked (traffic back to A)
- Adaptation: Drivers learn optimal routes over time

Challenge: Model must learn temporal adaptation
```

### Scenario 4: Multi-Event Overlap

```python
Situation: Concert + Rain + Friday evening
Effect:
- Concert alone: +80% near venue
- Rain alone: -30% speed
- Friday rush: +40% baseline
- Combined: Complex non-additive effect

Challenge: Model must handle event combinations
```

---

## Expected Performance Impact

### GraphWaveNet (Naive Baseline)

```
Current (1 week, stable):
  MAE: 0.25 km/h (autocorrelation exploitation)

Expected (1 year, challenging):
  MAE: 4-6 km/h (forced to learn patterns)
  - Incidents: Cannot predict from autocorrelation
  - Seasonal: Must capture long-term trends
  - Events: Need spatial reasoning
```

### LSTM Baseline

```
Current (1 week):
  MAE: 4.42 km/h

Expected (1 year):
  MAE: 5-7 km/h
  - Temporal learning helps
  - But no spatial awareness for incidents
```

### STMGT (Our Model)

```
Current (1 week):
  MAE: 1.88 km/h

Expected (1 year):
  MAE: 3-4 km/h
  - Spatial-temporal graphs help incidents
  - Multi-head attention for event combinations
  - Should maintain relative advantage
```

---

## Implementation Plan

### Week 1: Foundation

- [ ] Generate base traffic patterns (52 weeks)
- [ ] Add spatial variations (node-specific)
- [ ] Implement seasonal overlays
- [ ] Create holiday calendar (Vietnamese)

### Week 2: Events

- [ ] Implement incident generator (Poisson)
- [ ] Add construction zone scheduler
- [ ] Create weather simulator (seasonal)
- [ ] Design special event calendar

### Week 3: Integration

- [ ] Spatial propagation algorithm
- [ ] Event interaction logic
- [ ] Quality validation suite
- [ ] Generate final dataset

### Week 4: Validation & Documentation

- [ ] Statistical validation
- [ ] Visualization (sample weeks)
- [ ] Documentation (data dictionary)
- [ ] Split creation (train/val/test)

---

## Success Metrics

### Dataset Quality

1. ✅ Speed range realistic (3-52 km/h)
2. ✅ Temporal smoothness (except events)
3. ✅ Spatial consistency (correlated neighbors)
4. ✅ Event frequency realistic (not too many/few)
5. ✅ Seasonal patterns visible

### Challenge Level

1. ✅ Autocorrelation lag-12 < 0.95 (harder than current)
2. ✅ Incident unpredictability (random timing)
3. ✅ Event variety (10+ event types)
4. ✅ Construction zones (5+ zones/year)
5. ✅ Weather variability (seasonal patterns)

### Model Discrimination

1. ✅ GraphWaveNet MAE > 4 km/h (no shortcuts)
2. ✅ STMGT maintains advantage (spatial reasoning)
3. ✅ Clear performance separation
4. ✅ Realistic evaluation

---

## Tools & Scripts

### Primary Script

```bash
scripts/data/generate_super_dataset.py
  --duration 365  # days
  --interval 10   # minutes
  --edges 144
  --output data/processed/super_dataset_1year.parquet
  --seed 42
  --config configs/super_dataset_config.yaml
```

### Configuration File

```yaml
# configs/super_dataset_config.yaml
base_patterns:
  weekday_rush_hours: [7-9, 17-19]
  weekend_leisure_hours: [10-18]
  night_speed_boost: 1.15

incidents:
  rate_per_week: 3
  duration_range: [30, 180] # minutes
  severity_range: [0.1, 0.9] # speed reduction

construction:
  zones_per_year: 8
  duration_range: [14, 56] # days
  active_hours: [9, 17]

weather:
  rainy_days_percent: 20
  fog_days_percent: 5
  rain_duration_range: [60, 240] # minutes

events:
  per_month: 2
  types: [concert, sports, festival, parade]
  venue_edges: [...specific edges...]
```

### Validation Script

```bash
scripts/data/validate_super_dataset.py
  --dataset data/processed/super_dataset_1year.parquet
  --output data/processed/super_dataset_statistics.json
  --visualize
```

---

## Timeline

**Total Duration:** 3-4 weeks

- Week 1: Foundation + Seasonal patterns
- Week 2: Event injection + Spatial propagation
- Week 3: Integration + Quality control
- Week 4: Validation + Documentation

**Start Date:** November 15, 2025  
**Target Completion:** December 13, 2025

---

## Next Steps

1. Review and approve design document
2. Create configuration file template
3. Implement base pattern generator
4. Build event injection framework
5. Run first prototype (1 month sample)
6. Validate and iterate
7. Generate full 1-year dataset

---

**Status:** 📋 DESIGN COMPLETE - Ready for implementation  
**Approval Required:** Yes (review challenging scenarios)  
**Estimated Dataset Size:** ~500MB parquet file  
**Expected Training Impact:** 3-5x longer training time (worth it!)

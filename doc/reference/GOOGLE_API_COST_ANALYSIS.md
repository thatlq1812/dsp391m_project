# Google Directions API - Cost Analysis & Optimization
## Overview
Phân tích chi phí sử dụng Google Directions API cho traffic data collection.
## Pricing (2025)
**Google Directions API**:
- Standard: **$5.00 per 1,000 requests**
- No free tier for Directions API
- [Official Pricing](https://developers.google.com/maps/billing/gmp-billing)
## Cost Breakdown
### Collection Parameters
```yaml
Frequency: Every 15 minutes (96 collections/day)
Duration: 7 days / 30 days
Edges per collection: nodes * k_neighbors
```
### Cost Scenarios
#### **Option 1: 128 nodes, k_neighbors=5** (Maximum coverage)
```
Edges per collection: 128 × 5 = 640
Requests per day: 640 × 96 = 61,440
```
**Cost**:
- 7 days: 430,080 requests = **$2,150.40**
- 30 days: 1,843,200 requests = **$9,216.00/month**
**Pros**: Most complete network data 
**Cons**: Very expensive
---
#### **Option 2: 128 nodes, k_neighbors=3** (Recommended [DONE])
```
Edges per collection: 128 × 3 = 384
Requests per day: 384 × 96 = 36,864
```
**Cost**:
- 7 days: 258,048 requests = **$1,290.24**
- 30 days: 1,105,920 requests = **$5,529.60/month**
**Pros**: Good coverage with reasonable cost 
**Cons**: Slightly less network detail
---
#### **Option 3: 50 nodes, k_neighbors=5** (Current/Testing)
```
Edges per collection: 50 × 5 = 250
Requests per day: 250 × 96 = 24,000
```
**Cost**:
- 7 days: 168,000 requests = **$840.00**
- 30 days: 720,000 requests = **$3,600.00/month**
**Pros**: Lower cost 
**Cons**: Limited coverage (only 50 nodes)
---
#### **Option 4: Mock API** (Development - FREE [DONE][DONE][DONE])
```
Cost: $0
```
**Pros**:
- Zero cost for development and testing
- Unlimited requests
- Realistic simulated traffic patterns
**Cons**:
- Not real traffic data
- Need real API for production
## Cost Comparison Table
| Configuration | Nodes | k | Edges | 7 Days | 30 Days |
| --------------- | ------- | ----- | ------- | ---------- | ---------- |
| Option 1 | 128 | 5 | 640 | $2,150 | $9,216 |
| **Option 2** [DONE] | **128** | **3** | **384** | **$1,290** | **$5,530** |
| Option 3 | 50 | 5 | 250 | $840 | $3,600 |
| Option 4 (Mock) | 128 | 3 | 384 | **$0** | **$0** |
## Current Configuration
**File**: `configs/project_config.yaml`
```yaml
google_directions:
enabled: true
k_neighbors: 3 # Optimized for cost
limit_nodes: 128 # Good coverage
use_mock_api: true # FREE for development!
```
**Node Selection**:
```yaml
node_selection:
max_nodes: 128 # Limit to highest quality nodes
min_degree: 4
min_importance_score: 20.0
```
## Cost Optimization Strategies
### 1. Use Mock API for Development (CURRENT [DONE])
```yaml
google_directions:
use_mock_api: true # Set to false only for production
```
**Savings**: 100% (from $5,530/month → $0)
### 2. Reduce k_neighbors
```yaml
k_neighbors: 3 # Instead of 5
```
**Savings**: 40% (from $9,216 → $5,530 per month)
**Impact**: Still good network coverage (3 neighbors per node)
### 3. Reduce Collection Frequency
Instead of every 15 minutes:
| Frequency | Collections/day | 30-day Cost (128 nodes, k=3) |
| --------- | --------------- | ---------------------------- |
| 15 min | 96 | $5,530 |
| 30 min | 48 | $2,765 |
| 1 hour | 24 | $1,382 |
**Trade-off**: Less temporal resolution
### 4. Limit Nodes to High-Traffic Areas
```yaml
node_selection:
max_nodes: 128 # Select 128 best nodes only
min_importance_score: 20.0 # Only major intersections
```
**Benefit**: Focus on most important locations
### 5. Use Caching
Cache Google API responses for:
- Same origin-destination pairs
- Similar time periods
- Historical data lookup
**Potential savings**: 20-30%
## Recommended Setup
### Development/Testing (FREE)
```yaml
google_directions:
use_mock_api: true
limit_nodes: 128
k_neighbors: 3
```
**Cost**: $0
### Production (Optimized Cost)
```yaml
google_directions:
use_mock_api: false # Use real API
limit_nodes: 128
k_neighbors: 3
rate_limit_requests_per_minute: 2800
```
**Cost**: ~$5,530/month
**Strategy**:
- Run real API during peak hours only (7-9 AM, 5-7 PM) → Save 70%
- Use mock/cached data for off-peak hours
- **Estimated cost**: ~$1,650/month
## Monthly Budget Planning
### Conservative Budget ($2,000/month)
- 128 nodes, k=3, every 30 minutes
- Real API during rush hours only (4 hours/day)
- Mock API rest of the time
- **Estimated**: ~$1,800/month
### Standard Budget ($5,000/month)
- 128 nodes, k=3, every 15 minutes
- Real API all day
- **Estimated**: ~$5,530/month
### Premium Budget ($10,000/month)
- 128 nodes, k=5, every 15 minutes
- Real API all day
- **Estimated**: ~$9,216/month
## API Request Breakdown
### Per Collection Cycle
```python
# 128 nodes, k_neighbors = 3
nodes = 128
k = 3
edges = nodes * k = 384 requests
# Time: ~2-3 minutes (with rate limiting)
# Cost: 384 × $5/1000 = $1.92
```
### Daily
```python
collections_per_day = 96 # Every 15 min
daily_requests = 384 × 96 = 36,864
daily_cost = $184.32
```
### Weekly
```python
weekly_requests = 36,864 × 7 = 258,048
weekly_cost = $1,290.24
```
### Monthly
```python
monthly_requests = 36,864 × 30 = 1,105,920
monthly_cost = $5,529.60
```
## Switching Between Mock and Real API
### Enable Mock API (Development)
```bash
# In configs/project_config.yaml
google_directions:
use_mock_api: true
```
### Enable Real API (Production)
```bash
# In configs/project_config.yaml
google_directions:
use_mock_api: false
```
**Note**: Ensure `GOOGLE_MAPS_API_KEY` environment variable is set!
## Monitoring Costs
### Track API Usage
```python
# Check logs
grep "Google Directions API" logs/collector.log | wc -l
# Estimated cost
requests=$(grep "Google Directions API" logs/collector.log | wc -l)
cost=$(echo "scale=2; $requests * 5 / 1000" | bc)
echo "Estimated cost: \$$cost USD"
```
### Google Cloud Console
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Navigate to "APIs & Services" → "Dashboard"
3. View "Directions API" usage
4. Check billing reports
### Set Budget Alerts
```bash
# In Google Cloud Console
1. Go to Billing → Budgets & Alerts
2. Create budget: $6,000/month
3. Set alerts at: 50%, 75%, 90%, 100%
```
## Cost Reduction Checklist
- [x] [DONE] Use mock API for development
- [x] [DONE] Reduce k_neighbors from 5 to 3
- [x] [DONE] Limit nodes to 128 highest quality
- [ ] [PENDING] Implement caching for duplicate requests
- [ ] [PENDING] Run real API only during peak hours
- [ ] [PENDING] Use historical data for predictions during off-peak
## Summary
**Current Configuration** (Optimized):
- 128 nodes (high quality only)
- k_neighbors = 3 (good coverage)
- Mock API for development (FREE)
- Real API option for production ($5,530/month)
**Cost Savings Achieved**:
- Mock API: 100% savings during development
- Reduced k: 40% savings vs k=5
- Limited nodes: Focused on quality over quantity
**Recommendation**:
1. [DONE] Use mock API for development and testing (current setup)
2. When ready for production, enable real API during peak hours only
3. Monitor costs closely and adjust collection frequency as needed
---
**Last Updated**: 2025-10-25 
**Configuration**: v3.1 (optimized for cost)

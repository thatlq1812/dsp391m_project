# Performance Monitoring Guide

This guide explains how to monitor and optimize performance of the Traffic Forecast system.

## Benchmarking

### Run Performance Tests

```bash
# Install benchmark plugin
pip install pytest-benchmark

# Run all performance tests
pytest tests/test_performance.py -v --benchmark-only

# Run with detailed output
pytest tests/test_performance.py -v --benchmark-only --benchmark-verbose

# Save benchmark results
pytest tests/test_performance.py --benchmark-only --benchmark-save=baseline

# Compare with baseline
pytest tests/test_performance.py --benchmark-only --benchmark-compare=baseline
```

### Key Metrics

Performance tests measure:

- **Feature engineering speed** - Time to create temporal/spatial features
- **Distance calculations** - Haversine distance computation
- **Data validation** - Quality checks on large datasets
- **Storage operations** - Database write/read performance
- **Model predictions** - Inference speed

### Performance Targets

| Operation                      | Target  | Current |
| ------------------------------ | ------- | ------- |
| Feature engineering (10k rows) | < 1s    | TBD     |
| Haversine distance             | < 1ms   | TBD     |
| Data validation (1k rows)      | < 500ms | TBD     |
| Storage write (100 rows)       | < 100ms | TBD     |
| Model prediction               | < 10ms  | TBD     |

## Profiling

### CPU Profiling

```bash
# Install profiling tools
pip install py-spy cProfile-tools

# Profile specific script
python -m cProfile -o profile.stats scripts/collect_and_render.py --once

# Analyze results
python -m pstats profile.stats
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler scripts/collect_and_render.py --once

# Line-by-line profiling (add @profile decorator)
mprof run scripts/collect_and_render.py --once
mprof plot
```

### Live Profiling

```bash
# Install py-spy
pip install py-spy

# Live profiling of running process
py-spy top --pid <PID>

# Generate flame graph
py-spy record -o profile.svg --pid <PID>
```

## Monitoring Production

### Metrics Collection

Key metrics to monitor:

- **Collection time** - Time per data collection cycle
- **API response time** - Google/OpenMeteo API latency
- **Database performance** - Query execution time
- **Memory usage** - RAM consumption
- **CPU usage** - Processing load

### Logging Performance

Enable performance logging:

```python
import logging
import time

logger = logging.getLogger(__name__)

def timed_operation(func):
    """Decorator to log operation timing."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} completed in {duration:.2f}s")
        return result
    return wrapper
```

### Dashboard Monitoring

Use the live dashboard to monitor:

```bash
python scripts/live_dashboard.py
```

Metrics available:

- Real-time collection status
- Data quality trends
- System resource usage
- API call statistics

## Optimization Tips

### Database Optimization

1. **Use indexes:**

   ```sql
   CREATE INDEX idx_timestamp ON traffic_data(timestamp);
   CREATE INDEX idx_nodes ON traffic_data(node_a_id, node_b_id);
   ```

2. **Batch inserts:**

   ```python
   # Instead of single inserts
   history.save_batch(timestamps, data_list)
   ```

3. **Connection pooling:**
   ```python
   from sqlalchemy.pool import QueuePool
   engine = create_engine(url, poolclass=QueuePool, pool_size=10)
   ```

### Feature Engineering Optimization

1. **Vectorize operations:**

   ```python
   # Use pandas vectorized operations
   df['speed_kmh'] = df['distance_km'] / (df['duration_sec'] / 3600)
   ```

2. **Use NumPy:**

   ```python
   import numpy as np
   # NumPy is faster than Python loops
   speeds = np.array(speeds)
   avg_speed = np.mean(speeds)
   ```

3. **Avoid DataFrame iteration:**

   ```python
   # Bad
   for _, row in df.iterrows():
       process(row)

   # Good
   df.apply(process, axis=1)
   ```

### API Call Optimization

1. **Parallel requests:**

   ```python
   with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
       futures = [executor.submit(api_call, data) for data in batch]
   ```

2. **Rate limiting:**

   ```python
   from traffic_forecast.collectors.google.collector import RateLimiter
   limiter = RateLimiter(requests_per_minute=2500)
   limiter.wait_if_needed()
   ```

3. **Caching:**

   ```python
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def expensive_operation(param):
       return result
   ```

### Memory Optimization

1. **Use generators:**

   ```python
   # Instead of loading all data
   def data_generator():
       for chunk in pd.read_csv('large_file.csv', chunksize=1000):
           yield chunk
   ```

2. **Delete large objects:**

   ```python
   import gc
   del large_dataframe
   gc.collect()
   ```

3. **Use appropriate data types:**
   ```python
   df['node_id'] = df['node_id'].astype('category')
   df['speed'] = df['speed'].astype('float32')  # vs float64
   ```

## Bottleneck Detection

### Find Performance Bottlenecks

1. **Time each component:**

   ```python
   import time

   start = time.time()
   collect_data()
   print(f"Collection: {time.time() - start:.2f}s")

   start = time.time()
   process_data()
   print(f"Processing: {time.time() - start:.2f}s")
   ```

2. **Use context managers:**

   ```python
   from contextlib import contextmanager

   @contextmanager
   def timer(name):
       start = time.time()
       yield
       print(f"{name}: {time.time() - start:.2f}s")

   with timer("Data collection"):
       collect_data()
   ```

3. **Profile specific functions:**

   ```python
   import cProfile
   import pstats

   profiler = cProfile.Profile()
   profiler.enable()

   # Code to profile
   expensive_function()

   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(10)
   ```

## Performance Testing in CI/CD

Add performance regression tests to CI:

```yaml
# .github/workflows/performance.yml
name: Performance Tests

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -r requirements.txt pytest-benchmark
      - run: pytest tests/test_performance.py --benchmark-only
      - run: pytest tests/test_performance.py --benchmark-compare=baseline || true
```

## Resources

- [Python Profiling Tools](https://docs.python.org/3/library/profile.html)
- [Pandas Performance](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)
- [NumPy Performance](https://numpy.org/doc/stable/user/performance.html)
- [SQLAlchemy Performance](https://docs.sqlalchemy.org/en/20/faq/performance.html)

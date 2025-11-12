# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Weather Data vs Data Leakage: A Clear Explanation

## The Confusion

Many people think: "If we use weather data from the dataset that includes future timestamps, isn't that data leakage?"

**Short Answer:** NO - Weather data is NOT leakage. Let me explain why.

---

## Understanding the Difference

### What is Data Leakage?

**Data leakage** occurs when information from the test set influences training in a way that wouldn't be available during real deployment.

### Two Types of Variables

#### 1. Endogenous Variables (What We Predict)

- **Traffic Speed** ← This is what we're trying to predict
- Not available for future timestamps
- Must be forecasted by our model

**Example:**

```
Time: 2025-11-12 15:00
Traffic speed at 18:00: ??? (unknown - must predict)
```

#### 2. Exogenous Variables (External Inputs)

- **Weather**, **Time of Day**, **Day of Week**, **Holidays**
- Available or forecastable at prediction time
- External to the system we're modeling

**Example:**

```
Time: 2025-11-12 15:00
Weather forecast at 18:00: 28°C, 5mm rain (available from weather API)
Time at 18:00: Hour=18, DayOfWeek=Tuesday (known)
```

---

## The Key Insight

### Scenario 1: Real-World Deployment

```
Current Time: 15:00
Want to predict: Traffic speed at 18:00

Available Information:
✓ Weather forecast for 18:00 (from OpenWeatherMap API)
✓ Time features (hour=18, dow=Tuesday)
✓ Historical traffic patterns (from past data)
✗ Actual traffic speed at 18:00 (doesn't exist yet)

Model Input:
- Past traffic speeds: [45, 42, 48, ...] (history)
- Weather forecast: 28°C, 5mm rain (for 18:00)
- Time features: hour=18, dow=2

Model Output:
- Predicted speed: 35 km/h
```

### Scenario 2: Training on Historical Data

```
Training Example from 2025-10-15:
Timestamp: 15:00 (in the past)

Input Features:
- Traffic history: [40, 38, 45, ...] (before 15:00)
- Weather at 15:00: 29°C, 2mm rain
- Time features: hour=15, dow=Tuesday

Target:
- Actual traffic speed at 15:00: 42 km/h

This is VALID because:
- In deployment, we'd have weather forecasts
- Weather at prediction time is observable/forecastable
- We're NOT using future traffic speeds
```

---

## What IS Data Leakage

### Example 1: Using Test Set Traffic Patterns (LEAKAGE)

```python
# WRONG - This is data leakage
all_data = pd.concat([train, test])  # Includes future traffic
hourly_pattern = all_data.groupby('hour')['speed'].mean()  # ← Uses test traffic

# When training
predicted_speed = base_speed * hourly_pattern[hour]  # ← Influenced by future
```

**Why this is leakage:**

- Uses actual traffic speeds from test period
- These speeds don't exist at prediction time
- Creates artificially good performance

### Example 2: Using Weather Data (NOT LEAKAGE)

```python
# CORRECT - This is NOT leakage
# Weather at each timestamp
weather_data = df[['timestamp', 'temperature', 'rain']]

# When training on 2025-10-15 15:00:
features = {
    'weather_temp': 29°C,      # Observable at 15:00
    'weather_rain': 2mm,       # Observable at 15:00
    'hour': 15,                # Known
    'dow': 2                   # Known
}
```

**Why this is NOT leakage:**

- Weather is observable/forecastable at prediction time
- Same information available in deployment
- Weather API provides forecasts 3-7 days ahead

---

## The Weather Data Flow

### In Training (Historical Data)

```
Dataset Structure:
timestamp | speed_kmh | temperature | wind | rain
----------|-----------|-------------|------|-----
10:00     | 45.2      | 28.5        | 12.0 | 0.0
10:15     | 42.8      | 28.7        | 13.2 | 0.5

Model learns correlation:
"When temp=28°C and rain=0.5mm → speed ≈ 43 km/h"
```

### In Deployment (Real-time)

```
Current Time: 10:00
Predict for: 13:00 (3 hours ahead)

Inputs:
- Traffic history: [45.2, 42.8, ...] (past measurements)
- Weather forecast: temp=30°C, rain=1mm (from API)
- Time features: hour=13, dow=Monday

Model applies learned correlation:
"Given forecast temp=30°C, rain=1mm → predict speed ≈ 40 km/h"
```

---

## What We're Actually Fixing

Our data leakage fix addresses:

### ✓ FIXING (Real Leakage)

1. **Traffic pattern statistics from test set**

   ```python
   # Using test set hourly patterns
   test_hourly = test_df.groupby('hour')['speed'].mean()  # ← LEAKAGE
   ```

2. **Interpolation between train and test runs**

   ```python
   # Creating intermediate runs using test data
   if train_run[-1] < test_run[0]:
       interpolated = interpolate(train_run[-1], test_run[0])  # ← LEAKAGE
   ```

3. **Edge-specific patterns from full dataset**
   ```python
   # Using test traffic to learn edge characteristics
   edge_stats = all_data.groupby(['node_a', 'node_b'])['speed'].mean()  # ← LEAKAGE
   ```

### ✗ NOT FIXING (Not Leakage)

1. **Weather data usage**

   ```python
   # Using weather as input feature
   features['temperature'] = weather_data['temp']  # ← NOT leakage
   ```

2. **Time features**

   ```python
   # Using time as input feature
   features['hour'] = timestamp.hour  # ← NOT leakage
   features['dow'] = timestamp.dayofweek  # ← NOT leakage
   ```

3. **Graph structure**
   ```python
   # Using road network topology
   edge_index = build_graph_from_osm()  # ← NOT leakage (domain knowledge)
   ```

---

## Analogy: Weather vs Stock Prices

To make it crystal clear:

### Stock Price Prediction (Similar to Traffic)

**Predict:** Stock price at 16:00

- **Endogenous:** Stock price (what we predict)
- **Exogenous:** News events, market indicators, time

**NOT Leakage:** Using news/events that will happen by 16:00
**IS Leakage:** Using stock prices from test period

### Weather Impact on Traffic

**Predict:** Traffic speed at 16:00

- **Endogenous:** Traffic speed (what we predict)
- **Exogenous:** Weather forecast, time, day of week

**NOT Leakage:** Using weather forecasts available at prediction time
**IS Leakage:** Using traffic patterns from test period

---

## Practical Implications

### For Researchers

When reviewing papers:

- ✓ Using weather/time features: ACCEPTABLE
- ✗ Using future traffic patterns: REJECT (data leakage)

### For Practitioners

When deploying models:

- ✓ Integrate weather forecast APIs
- ✓ Use time-based features
- ✗ Never use future traffic data in training

### For Our Project

Current status:

- ✓ Weather usage: Correct and acceptable
- ✗ Traffic pattern augmentation: Has leakage (being fixed)

---

## Summary

| Data Type                  | Nature           | Available at Prediction? | Leakage?                |
| -------------------------- | ---------------- | ------------------------ | ----------------------- |
| **Traffic Speed (target)** | Endogenous       | NO - must predict        | YES if used from future |
| **Weather Forecast**       | Exogenous        | YES - from APIs          | NO                      |
| **Time Features**          | Exogenous        | YES - deterministic      | NO                      |
| **Graph Structure**        | Domain Knowledge | YES - fixed              | NO                      |
| **Historical Traffic**     | Endogenous       | YES - for past only      | NO if from train period |

**Key Principle:**

- If information is **observable or forecastable** at prediction time → NOT leakage
- If information is **from future and must be predicted** → IS leakage

---

## References

- Weather forecast APIs: OpenWeatherMap, Weather.com, etc.
- Time series best practices: Use only information available at prediction time
- Exogenous vs Endogenous variables in forecasting

---

**Question Answered:**

> "Isn't using weather data from the dataset that includes future timestamps also data leakage?"

**Answer:**

> No, because weather forecasts are available at prediction time from external APIs. The leakage is in using future TRAFFIC patterns, not future weather conditions.

"""
Naive baseline forecast model - persistence.
"""

def predict_speed_persistence(historical_speeds, horizon_steps):
 """
 Simple persistence: predict last known speed for all horizons.
 """
 if not historical_speeds:
 return [40.0] * len(horizon_steps) # Default speed
 last_speed = historical_speeds[-1]
 return [last_speed] * len(horizon_steps)

def forecast_node(node_id, horizons_min):
 """
 Mock forecast for a node.
 """
 # Mock historical data
 historical = [35.0, 38.0, 42.0]
 predictions = predict_speed_persistence(historical, horizons_min)
 return {
 'node_id': node_id,
 'predictions': dict(zip(horizons_min, predictions))
 }
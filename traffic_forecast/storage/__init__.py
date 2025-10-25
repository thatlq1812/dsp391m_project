"""
Traffic data storage module for historical data management.

This module provides persistent storage for traffic data to enable:
- Lag feature computation without re-collecting old data
- Historical data access for time series analysis
- Efficient data retrieval with time-based indexing
"""

from .traffic_history import TrafficHistoryStore

__all__ = ['TrafficHistoryStore']

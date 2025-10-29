"""
Adaptive Scheduler for Cost-Optimized Traffic Data Collection

Implements intelligent scheduling to reduce Google Directions API costs
by adjusting collection frequency based on time of day and day of week.

Academic Configuration v4.0:
- Peak hours (7-9 AM, 12-1 PM, 5-7 PM): 30-minute intervals
- Off-peak hours: 60-minute intervals  
- Weekends: 90-minute intervals (optional)

Cost Savings: ~85% reduction vs fixed 15-minute intervals
"""

from datetime import datetime, time, timedelta
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class TimeRange:
    """Represents a time range within a day (e.g., 07:00-09:00)"""
    
    def __init__(self, start: str, end: str):
        """
        Args:
            start: Start time in HH:MM format (24-hour)
            end: End time in HH:MM format (24-hour)
        """
        self.start = self._parse_time(start)
        self.end = self._parse_time(end)
    
    @staticmethod
    def _parse_time(time_str: str) -> time:
        """Parse HH:MM string to time object"""
        hour, minute = map(int, time_str.split(':'))
        return time(hour=hour, minute=minute)
    
    def contains(self, check_time: time) -> bool:
        """Check if given time falls within this range"""
        return self.start <= check_time < self.end
    
    def __repr__(self):
        return f"TimeRange({self.start.strftime('%H:%M')}-{self.end.strftime('%H:%M')})"


class AdaptiveScheduler:
    """
    Adaptive scheduler that adjusts collection intervals based on time.
    
    This helps reduce API costs by collecting more frequently during
    high-traffic variability periods (rush hours) and less frequently
    during stable periods (late night, weekends).
    """
    
    def __init__(self, config: dict):
        """
        Initialize scheduler from config.
        
        Args:
            config: Scheduler configuration dict from project_config.yaml
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.mode = config.get('mode', 'fixed')
        
        # Fixed mode config
        self.fixed_interval = config.get('fixed_interval_minutes', 15)
        
        # Adaptive mode config
        if self.mode == 'adaptive':
            adaptive_cfg = config.get('adaptive', {})
            
            # Peak hours configuration
            peak_cfg = adaptive_cfg.get('peak_hours', {})
            self.peak_enabled = peak_cfg.get('enabled', True)
            self.peak_interval = peak_cfg.get('interval_minutes', 30)
            self.peak_ranges = [
                TimeRange(r['start'], r['end'])
                for r in peak_cfg.get('time_ranges', [])
            ]
            
            # Off-peak configuration
            offpeak_cfg = adaptive_cfg.get('offpeak', {})
            self.offpeak_interval = offpeak_cfg.get('interval_minutes', 60)
            
            # Weekend configuration
            weekend_cfg = adaptive_cfg.get('weekend', {})
            self.weekend_enabled = weekend_cfg.get('enabled', True)
            self.weekend_use_offpeak_only = weekend_cfg.get('use_offpeak_only', True)
            self.weekend_interval = weekend_cfg.get('interval_minutes', 90)
            
            logger.info(
                f"Adaptive scheduler initialized: "
                f"{len(self.peak_ranges)} peak ranges, "
                f"peak={self.peak_interval}min, "
                f"offpeak={self.offpeak_interval}min, "
                f"weekend={self.weekend_interval}min"
            )
        else:
            logger.info(f"Fixed interval scheduler: {self.fixed_interval} minutes")
    
    def is_weekend(self, dt: Optional[datetime] = None) -> bool:
        """Check if given datetime is on weekend (Saturday=5, Sunday=6)"""
        if dt is None:
            dt = datetime.now()
        return dt.weekday() >= 5  # 5=Saturday, 6=Sunday
    
    def is_peak_hour(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if given datetime falls within peak hours.
        
        Args:
            dt: Datetime to check (default: now)
            
        Returns:
            True if within any peak time range, False otherwise
        """
        if dt is None:
            dt = datetime.now()
        
        if not self.peak_enabled:
            return False
        
        current_time = dt.time()
        return any(range.contains(current_time) for range in self.peak_ranges)
    
    def get_interval_minutes(self, dt: Optional[datetime] = None) -> int:
        """
        Get collection interval in minutes for given datetime.
        
        Args:
            dt: Datetime to check (default: now)
            
        Returns:
            Interval in minutes based on schedule
        """
        if not self.enabled or self.mode == 'fixed':
            return self.fixed_interval
        
        if dt is None:
            dt = datetime.now()
        
        # Weekend override
        if self.weekend_enabled and self.is_weekend(dt):
            if self.weekend_use_offpeak_only:
                return self.weekend_interval
            # Fall through to check peak hours on weekend
        
        # Check if peak hour
        if self.is_peak_hour(dt):
            return self.peak_interval
        else:
            return self.offpeak_interval
    
    def should_collect_now(
        self,
        last_collection: Optional[datetime] = None,
        current_time: Optional[datetime] = None
    ) -> bool:
        """
        Determine if data collection should happen now.
        
        Args:
            last_collection: When last collection occurred (None if first run)
            current_time: Current time to check (default: now)
            
        Returns:
            True if should collect, False otherwise
        """
        if current_time is None:
            current_time = datetime.now()
        
        # First collection - always run
        if last_collection is None:
            logger.info("First collection - proceeding")
            return True
        
        # Check if enough time has passed
        interval_minutes = self.get_interval_minutes(current_time)
        time_since_last = current_time - last_collection
        required_interval = timedelta(minutes=interval_minutes)
        
        should_collect = time_since_last >= required_interval
        
        if should_collect:
            logger.info(
                f"Collection triggered: {time_since_last.total_seconds()/60:.1f}min "
                f"elapsed (required: {interval_minutes}min, "
                f"schedule: {'peak' if self.is_peak_hour(current_time) else 'offpeak'})"
            )
        
        return should_collect
    
    def get_next_collection_time(
        self,
        last_collection: Optional[datetime] = None,
        from_time: Optional[datetime] = None
    ) -> datetime:
        """
        Calculate when next collection should occur.
        
        Args:
            last_collection: When last collection occurred
            from_time: Calculate from this time (default: now)
            
        Returns:
            Datetime of next scheduled collection
        """
        if from_time is None:
            from_time = datetime.now()
        
        if last_collection is None:
            return from_time  # Collect immediately
        
        # Get interval for current schedule
        interval_minutes = self.get_interval_minutes(from_time)
        next_time = last_collection + timedelta(minutes=interval_minutes)
        
        # If next time is in the past, calculate from now
        if next_time < from_time:
            next_time = from_time
        
        return next_time
    
    def get_schedule_info(self, dt: Optional[datetime] = None) -> dict:
        """
        Get detailed schedule information for given datetime.
        
        Args:
            dt: Datetime to check (default: now)
            
        Returns:
            Dict with schedule details
        """
        if dt is None:
            dt = datetime.now()
        
        is_weekend = self.is_weekend(dt)
        is_peak = self.is_peak_hour(dt)
        interval = self.get_interval_minutes(dt)
        
        if self.mode == 'fixed':
            schedule_type = 'fixed'
        elif is_weekend and self.weekend_enabled:
            schedule_type = 'weekend'
        elif is_peak:
            schedule_type = 'peak'
        else:
            schedule_type = 'offpeak'
        
        return {
            'enabled': self.enabled,
            'mode': self.mode,
            'schedule_type': schedule_type,
            'is_weekend': is_weekend,
            'is_peak_hour': is_peak,
            'interval_minutes': interval,
            'datetime': dt.isoformat(),
            'time': dt.strftime('%H:%M:%S'),
            'day_of_week': dt.strftime('%A'),
        }
    
    def get_daily_collection_count(self) -> int:
        """
        Estimate number of collections per day with current schedule.
        
        Returns:
            Approximate collections per day
        """
        if self.mode == 'fixed':
            return (24 * 60) // self.fixed_interval
        
        # Calculate for weekday (average)
        peak_minutes = sum(
            (range.end.hour * 60 + range.end.minute) -
            (range.start.hour * 60 + range.start.minute)
            for range in self.peak_ranges
        )
        offpeak_minutes = (24 * 60) - peak_minutes
        
        weekday_collections = (
            peak_minutes // self.peak_interval +
            offpeak_minutes // self.offpeak_interval
        )
        
        # Weekend
        if self.weekend_enabled:
            weekend_collections = (24 * 60) // self.weekend_interval
            # Average over week: 5 weekdays + 2 weekend days
            avg_per_day = (weekday_collections * 5 + weekend_collections * 2) / 7
            return int(avg_per_day)
        
        return weekday_collections
    
    def get_cost_estimate(
        self,
        nodes: int,
        k_neighbors: int,
        days: int = 30,
        price_per_1000: float = 5.0
    ) -> dict:
        """
        Estimate Google Directions API cost for given configuration.
        
        Args:
            nodes: Number of nodes
            k_neighbors: Neighbors per node (edges per node)
            days: Number of days to estimate
            price_per_1000: Google API price per 1000 requests
            
        Returns:
            Dict with cost breakdown
        """
        edges_per_collection = nodes * k_neighbors
        collections_per_day = self.get_daily_collection_count()
        
        total_collections = collections_per_day * days
        total_requests = edges_per_collection * total_collections
        total_cost = (total_requests / 1000) * price_per_1000
        
        return {
            'nodes': nodes,
            'k_neighbors': k_neighbors,
            'edges_per_collection': edges_per_collection,
            'collections_per_day': collections_per_day,
            'days': days,
            'total_collections': total_collections,
            'total_requests': total_requests,
            'total_cost_usd': round(total_cost, 2),
            'cost_per_day': round(total_cost / days, 2),
            'cost_per_week': round(total_cost * 7 / days, 2),
        }
    
    def print_schedule_summary(self):
        """Print a human-readable schedule summary"""
        print("=" * 70)
        print("ADAPTIVE SCHEDULER CONFIGURATION")
        print("=" * 70)
        print(f"Mode: {self.mode}")
        print(f"Enabled: {self.enabled}")
        print()
        
        if self.mode == 'fixed':
            print(f"Fixed interval: {self.fixed_interval} minutes")
            print(f"Collections per day: {self.get_daily_collection_count()}")
        else:
            print("Peak Hours:")
            for i, range in enumerate(self.peak_ranges, 1):
                print(f"  {i}. {range} -> {self.peak_interval} min interval")
            print()
            print(f"Off-Peak: {self.offpeak_interval} min interval")
            print()
            if self.weekend_enabled:
                print(f"Weekend: {self.weekend_interval} min interval")
                if self.weekend_use_offpeak_only:
                    print("  (All weekend treated as off-peak)")
            print()
            print(f"Estimated collections per day: {self.get_daily_collection_count()}")
        print("=" * 70)

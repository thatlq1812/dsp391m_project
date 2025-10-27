"""
Cache utilities for collectors to avoid repeated API calls.
"""

import os
import json
import hashlib
import time
from typing import Optional, Dict, Any


def get_cache_key(collector_name: str, params: Dict[str, Any]) -> str:
 """Generate a unique cache key based on collector name and parameters."""
 # Sort params for consistent hashing
 param_str = json.dumps(params, sort_keys=True)
 content = f"{collector_name}:{param_str}"
 return hashlib.md5(content.encode()).hexdigest()


def get_cache_path(cache_dir: str, cache_key: str) -> str:
 """Get full path for cache file."""
 return os.path.join(cache_dir, f"{cache_key}.json")


def is_cache_valid(cache_path: str, expiry_hours: int) -> bool:
 """Check if cache file exists and is within expiry time."""
 if not os.path.exists(cache_path):
 return False

 # Check file modification time
 mtime = os.path.getmtime(cache_path)
 age_hours = (time.time() - mtime) / 3600

 return age_hours < expiry_hours


def load_cache(cache_path: str) -> Optional[Dict[str, Any]]:
 """Load data from cache file."""
 try:
 with open(cache_path, 'r', encoding='utf-8') as f:
 return json.load(f)
 except (FileNotFoundError, json.JSONDecodeError):
 return None


def save_cache(cache_path: str, data: Dict[str, Any]) -> None:
 """Save data to cache file."""
 # Ensure cache directory exists
 os.makedirs(os.path.dirname(cache_path), exist_ok=True)

 with open(cache_path, 'w', encoding='utf-8') as f:
 json.dump(data, f, indent=2, ensure_ascii=False)


def get_or_create_cache(
 collector_name: str,
 params: Dict[str, Any],
 cache_dir: str,
 expiry_hours: int,
 fetch_func
) -> Dict[str, Any]:
 """
 Get data from cache or fetch and cache it.

 Args:
 collector_name: Name of the collector
 params: Parameters that affect the data
 cache_dir: Cache directory path
 expiry_hours: Cache expiry time in hours
 fetch_func: Function to fetch data if cache miss

 Returns:
 Cached or freshly fetched data
 """
 cache_key = get_cache_key(collector_name, params)
 cache_path = get_cache_path(cache_dir, cache_key)

 # Try to load from cache
 if is_cache_valid(cache_path, expiry_hours):
 cached_data = load_cache(cache_path)
 if cached_data:
 print(f"Using cached data for {collector_name} (key: {cache_key[:8]}...)")
 return cached_data

 # Cache miss or invalid - fetch fresh data
 print(f"Fetching fresh data for {collector_name}...")
 data = fetch_func()

 # Cache the result
 cache_data = {
 'timestamp': time.time(),
 'collector': collector_name,
 'params': params,
 'data': data
 }
 save_cache(cache_path, cache_data)

 return data


def clear_expired_cache(cache_dir: str, max_age_hours: int = 24*7) -> int:
 """Clear cache files older than max_age_hours. Returns number of files cleared."""
 if not os.path.exists(cache_dir):
 return 0

 cleared = 0
 now = time.time()

 for filename in os.listdir(cache_dir):
 if filename.endswith('.json'):
 filepath = os.path.join(cache_dir, filename)
 mtime = os.path.getmtime(filepath)
 age_hours = (now - mtime) / 3600

 if age_hours > max_age_hours:
 os.remove(filepath)
 cleared += 1

 return cleared

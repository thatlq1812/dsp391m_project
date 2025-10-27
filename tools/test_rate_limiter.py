#!/usr/bin/env python3
"""
Ad-hoc harness to exercise the Google collector rate limiter.
"""

import time

from traffic_forecast.collectors.google.collector import RateLimiter


def main() -> None:
 limiter = RateLimiter(requests_per_minute=120)
 print("Simulating 10 sequential requests...")
 for idx in range(10):
 limiter.wait_if_needed()
 print(f"Request {idx + 1} at {time.strftime('%X')}")


if __name__ == "__main__":
 main()

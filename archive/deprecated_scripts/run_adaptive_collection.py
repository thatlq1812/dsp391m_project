#!/usr/bin/env python3
"""
Adaptive Collection Runner
Runs traffic collection with adaptive scheduling based on time of day
"""

import subprocess
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from traffic_forecast.scheduler.adaptive_scheduler import AdaptiveScheduler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'adaptive_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_collection():
    """Execute one collection cycle"""
    try:
        logger.info("Starting collection cycle...")
        
        # Run collect_once.py
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / 'scripts' / 'collect_once.py')],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            logger.info("Collection completed successfully")
            return True
        else:
            logger.error(f"Collection failed with code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Collection timeout after 10 minutes")
        return False
    except Exception as e:
        logger.error(f"Collection error: {e}")
        return False


def main():
    """Main adaptive collection loop"""
    logger.info("=" * 70)
    logger.info("ADAPTIVE COLLECTION SCHEDULER v5.1")
    logger.info("=" * 70)
    
    # Load config from YAML
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent / "configs" / "project_config.yaml"
    with open(config_path, 'r') as f:
        project_config = yaml.safe_load(f)
    
    scheduler_config = project_config.get('scheduler', {})
    
    # Initialize scheduler
    scheduler = AdaptiveScheduler(scheduler_config)
    
    logger.info(f"Scheduler initialized:")
    logger.info(f"  Mode: {scheduler.mode}")
    if scheduler.mode == 'adaptive':
        logger.info(f"  Peak interval: {scheduler.peak_interval} min")
        logger.info(f"  Off-peak interval: {scheduler.offpeak_interval} min")
    else:
        logger.info(f"  Fixed interval: {scheduler.fixed_interval} min")
    logger.info("")
    
    # Run continuously
    collection_count = 0
    last_collection_time = None
    
    while True:
        try:
            current_time = datetime.now()
            
            # Check if should collect now
            if scheduler.should_collect_now(last_collection_time, current_time):
                # Run collection
                logger.info(f"Starting collection #{collection_count + 1}")
                success = run_collection()
                
                if success:
                    collection_count += 1
                    last_collection_time = current_time
                    logger.info(f"Collection #{collection_count} completed successfully")
                    logger.info(f"Total collections: {collection_count}")
                else:
                    logger.error("Collection failed")
            
            # Calculate next check interval (check every minute)
            time.sleep(60)
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            # Sleep 5 minutes on error before retrying
            time.sleep(300)
            time.sleep(300)


if __name__ == '__main__':
    main()

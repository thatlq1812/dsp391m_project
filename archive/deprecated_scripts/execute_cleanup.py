#!/usr/bin/env python3
"""
Execute Project Cleanup
Implement the cleanup plan analyzed in analyze_cleanup.py

Usage:
    python scripts/execute_cleanup.py --dry-run    # Preview changes
    python scripts/execute_cleanup.py --execute    # Apply changes
"""

import os
import shutil
import argparse
from pathlib import Path
import time

PROJECT_ROOT = Path('.')

def get_dir_size(path):
    """Get directory size in MB"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except:
        pass
    return total / (1024 * 1024)

def cleanup_augmented_runs(dry_run=True):
    """Remove augmented runs (Sep 2025)"""
    print("\n🗑️  STEP 1: Remove Augmented Runs")
    print("-" * 70)
    
    runs_dir = Path('data/runs')
    if not runs_dir.exists():
        print("  ⚠️  No runs directory found")
        return
    
    augmented_runs = sorted([d for d in runs_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('run_202509')])
    
    total_size = sum(get_dir_size(r) for r in augmented_runs[:100]) * len(augmented_runs) / 100
    
    print(f"  Found: {len(augmented_runs)} augmented runs (~{total_size:.1f} MB)")
    
    if dry_run:
        print(f"  🔍 DRY RUN - Would remove {len(augmented_runs)} directories")
        print(f"     Examples: {augmented_runs[0].name}, {augmented_runs[-1].name}")
    else:
        print(f"  🔥 REMOVING {len(augmented_runs)} augmented runs...")
        removed = 0
        for run_dir in augmented_runs:
            try:
                shutil.rmtree(run_dir)
                removed += 1
                if removed % 1000 == 0:
                    print(f"     Progress: {removed}/{len(augmented_runs)}")
            except Exception as e:
                print(f"     ⚠️  Failed to remove {run_dir.name}: {e}")
        
        print(f"  ✅ Removed {removed} augmented runs (~{total_size:.1f} MB freed)")

def cleanup_processed_data(dry_run=True):
    """Clean processed data, keep only original runs"""
    print("\n🗑️  STEP 2: Clean Processed Data")
    print("-" * 70)
    
    processed_dir = Path('data/processed')
    if not processed_dir.exists():
        print("  ⚠️  No processed directory found")
        return
    
    # Files to remove
    files_to_remove = [
        'all_runs_combined.parquet',  # Contains augmented data
        'train.parquet',
        'val.parquet',
        'test.parquet',
        'astgcn_sequences.npz'  # Old ASTGCN format
    ]
    
    total_size = 0
    existing_files = []
    
    for f in files_to_remove:
        file_path = processed_dir / f
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)
            total_size += size
            existing_files.append((file_path, size))
    
    print(f"  Found: {len(existing_files)} old processed files (~{total_size:.1f} MB)")
    
    if dry_run:
        print(f"  🔍 DRY RUN - Would remove:")
        for file_path, size in existing_files:
            print(f"     - {file_path.name} ({size:.1f} MB)")
    else:
        print(f"  🔥 REMOVING old processed files...")
        for file_path, size in existing_files:
            try:
                file_path.unlink()
                print(f"     ✓ Removed {file_path.name} ({size:.1f} MB)")
            except Exception as e:
                print(f"     ⚠️  Failed to remove {file_path.name}: {e}")
        
        print(f"  ✅ Freed {total_size:.1f} MB")

def cleanup_old_models(dry_run=True):
    """Remove old model checkpoints"""
    print("\n🗑️  STEP 3: Clean Old Model Checkpoints")
    print("-" * 70)
    
    saved_dir = Path('models/saved')
    if not saved_dir.exists():
        print("  ⚠️  No saved models directory found")
        return
    
    # Keep only best models
    models_to_keep = ['best_model.pt', 'stmgt_best.pt', 'enhanced_best.pt']
    
    all_files = list(saved_dir.rglob('*.pt'))
    files_to_remove = [f for f in all_files if f.name not in models_to_keep]
    
    total_size = sum(f.stat().st_size for f in files_to_remove) / (1024 * 1024)
    
    print(f"  Found: {len(files_to_remove)} old checkpoints (~{total_size:.1f} MB)")
    
    if dry_run:
        print(f"  🔍 DRY RUN - Would remove:")
        for f in files_to_remove[:5]:
            size = f.stat().st_size / (1024 * 1024)
            print(f"     - {f.relative_to(saved_dir)} ({size:.1f} MB)")
        if len(files_to_remove) > 5:
            print(f"     ... and {len(files_to_remove) - 5} more")
    else:
        print(f"  🔥 REMOVING old checkpoints...")
        for f in files_to_remove:
            try:
                f.unlink()
            except Exception as e:
                print(f"     ⚠️  Failed to remove {f.name}: {e}")
        
        print(f"  ✅ Removed {len(files_to_remove)} checkpoints ({total_size:.1f} MB freed)")

def cleanup_temp_scripts(dry_run=True):
    """Remove temporary/analysis scripts"""
    print("\n🗑️  STEP 4: Clean Temporary Scripts")
    print("-" * 70)
    
    scripts_to_remove = [
        'scripts/analyze_model_flow.py',
        'scripts/analyze_weather_strategy.py',
        'scripts/visualize_stmgt_comparison.py',
        'scripts/analyze_cleanup.py',  # This analysis script itself
    ]
    
    existing = [Path(s) for s in scripts_to_remove if Path(s).exists()]
    total_size = sum(f.stat().st_size for f in existing) / 1024
    
    print(f"  Found: {len(existing)} temporary scripts (~{total_size:.1f} KB)")
    
    if dry_run:
        print(f"  🔍 DRY RUN - Would remove:")
        for f in existing:
            print(f"     - {f}")
    else:
        print(f"  🔥 REMOVING temporary scripts...")
        for f in existing:
            try:
                f.unlink()
                print(f"     ✓ Removed {f}")
            except Exception as e:
                print(f"     ⚠️  Failed to remove {f}: {e}")
        
        print(f"  ✅ Removed {len(existing)} scripts")

def consolidate_documentation(dry_run=True):
    """Reorganize documentation"""
    print("\n📚 STEP 5: Consolidate Documentation")
    print("-" * 70)
    
    # Docs to keep (will be reorganized)
    core_docs = {
        'NOVEL_STMGT_ARCHITECTURE.md': 'docs/STMGT_ARCHITECTURE.md',
        'SUMMARY_STMGT_WEATHER.md': 'docs/PHASE_1_SUMMARY.md',
        'MODEL_ANALYSIS.md': 'docs/archive/MODEL_ANALYSIS.md',
        'ASTGCN_VS_ENHANCED.md': 'docs/archive/ASTGCN_VS_ENHANCED.md',
    }
    
    # Docs to archive
    docs_to_archive = [
        'ASTGCN_INTEGRATION.md',
        'PREPROCESSING_GUIDE.md',
    ]
    
    if dry_run:
        print(f"  🔍 DRY RUN - Would reorganize:")
        for old, new in core_docs.items():
            if Path(f'docs/{old}').exists():
                print(f"     RENAME: docs/{old} → {new}")
        
        print(f"\n  Would archive:")
        for doc in docs_to_archive:
            if Path(f'docs/{doc}').exists():
                print(f"     MOVE: docs/{doc} → docs/archive/{doc}")
    else:
        # Create archive directory
        Path('docs/archive').mkdir(exist_ok=True)
        
        # Rename core docs
        print(f"  📝 Reorganizing core documentation...")
        for old, new in core_docs.items():
            old_path = Path(f'docs/{old}')
            new_path = Path(new)
            if old_path.exists():
                try:
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(old_path, new_path)
                    if old != new.split('/')[-1]:  # Only remove if renamed
                        old_path.unlink()
                    print(f"     ✓ {old} → {new}")
                except Exception as e:
                    print(f"     ⚠️  Failed: {e}")
        
        # Archive old docs
        print(f"\n  📦 Archiving old documentation...")
        for doc in docs_to_archive:
            old_path = Path(f'docs/{doc}')
            new_path = Path(f'docs/archive/{doc}')
            if old_path.exists():
                try:
                    shutil.move(str(old_path), str(new_path))
                    print(f"     ✓ Archived {doc}")
                except Exception as e:
                    print(f"     ⚠️  Failed: {e}")
        
        print(f"  ✅ Documentation reorganized")

def create_summary_report():
    """Create cleanup summary report"""
    print("\n📊 CLEANUP SUMMARY")
    print("=" * 70)
    
    # Check what's left
    stats = {
        'Original runs': len([d for d in Path('data/runs').iterdir() 
                             if d.is_dir() and d.name.startswith('run_202510')]),
        'Augmented runs': len([d for d in Path('data/runs').iterdir() 
                              if d.is_dir() and d.name.startswith('run_202509')]),
        'Processed files': len(list(Path('data/processed').glob('*.parquet'))) if Path('data/processed').exists() else 0,
        'Model files': len(list(Path('traffic_forecast/ml/models').glob('*.py'))),
        'Documentation': len(list(Path('docs').glob('*.md'))),
    }
    
    for key, value in stats.items():
        print(f"  {key:<20} {value:>6,}")
    
    print("\n" + "=" * 70)

def main():
    parser = argparse.ArgumentParser(description='Execute project cleanup')
    parser.add_argument('--execute', action='store_true', help='Execute cleanup (default is dry-run)')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    args = parser.parse_args()
    
    # Default to dry-run if neither specified
    dry_run = not args.execute or args.dry_run
    
    print("=" * 70)
    print("🧹 PROJECT CLEANUP EXECUTION")
    print("=" * 70)
    
    if dry_run:
        print("\n⚠️  DRY RUN MODE - No changes will be made")
        print("    Run with --execute to apply changes")
    else:
        print("\n🔥 EXECUTION MODE - Changes will be applied")
        print("    Waiting 5 seconds... Press Ctrl+C to cancel")
        time.sleep(5)
    
    # Execute cleanup steps
    cleanup_augmented_runs(dry_run)
    cleanup_processed_data(dry_run)
    cleanup_old_models(dry_run)
    cleanup_temp_scripts(dry_run)
    consolidate_documentation(dry_run)
    
    # Summary
    if not dry_run:
        create_summary_report()
    
    print("\n" + "=" * 70)
    if dry_run:
        print("✅ DRY RUN COMPLETE")
        print("    Run with --execute to apply these changes")
    else:
        print("✅ CLEANUP COMPLETE")
        print("    Project is now clean and ready for Phase 2")
    print("=" * 70)

if __name__ == '__main__':
    main()

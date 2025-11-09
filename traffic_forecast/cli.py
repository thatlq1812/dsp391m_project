#!/usr/bin/env python3
"""
STMGT Traffic Forecasting - Command Line Interface
Interactive CLI tool for model management and monitoring

Author: THAT Le Quang
Date: 2025-11-09
"""

import click
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich import box
import yaml
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

console = Console()


@click.group()
@click.version_option(version="2.0.0")
def cli():
    """
    STMGT Traffic Forecasting CLI
    
    Manage models, monitor training, and control API server.
    """
    pass


# ============================================================================
# MODEL COMMANDS
# ============================================================================

@cli.group()
def model():
    """Model management commands"""
    pass


@model.command("list")
@click.option("--format", type=click.Choice(["table", "json"]), default="table", help="Output format")
def model_list(format):
    """List all trained models"""
    try:
        from traffic_forecast.utils.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        models = registry.list_models()
        
        if format == "json":
            console.print_json(data=models)
            return
        
        # Table format
        table = Table(title="Available Models", box=box.ROUNDED)
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("MAE", justify="right", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Date", style="blue")
        
        for model in models:
            table.add_row(
                model.get("name", "N/A"),
                model.get("type", "N/A"),
                f"{model.get('mae', 0):.2f}",
                model.get("status", "unknown"),
                model.get("date", "N/A")
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@model.command("info")
@click.argument("model_name")
def model_info(model_name):
    """Show detailed model information"""
    try:
        from traffic_forecast.utils.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        model = registry.get_model(model_name)
        
        if not model:
            console.print(f"[red]Model '{model_name}' not found[/red]")
            sys.exit(1)
        
        # Display model info in panel
        info_text = f"""
[cyan]Model:[/cyan] {model.get('name', 'N/A')}
[cyan]Type:[/cyan] {model.get('type', 'N/A')}
[cyan]Status:[/cyan] {model.get('status', 'unknown')}

[yellow]Performance:[/yellow]
  MAE:  {model.get('mae', 0):.4f} km/h
  RMSE: {model.get('rmse', 0):.4f} km/h
  R²:   {model.get('r2', 0):.4f}

[yellow]Configuration:[/yellow]
  Parameters: {model.get('params', 0):,}
  Batch Size: {model.get('batch_size', 0)}
  Epochs:     {model.get('epochs', 0)}

[yellow]Files:[/yellow]
  Checkpoint: {model.get('checkpoint_path', 'N/A')}
  Config:     {model.get('config_path', 'N/A')}
  
[yellow]Created:[/yellow] {model.get('date', 'N/A')}
        """
        
        console.print(Panel(info_text, title=f"Model: {model_name}", border_style="cyan"))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@model.command("compare")
@click.argument("model_names", nargs=-1, required=True)
def model_compare(model_names):
    """Compare multiple models"""
    try:
        from traffic_forecast.utils.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        
        # Table for comparison
        table = Table(title="Model Comparison", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        
        models_data = []
        for name in model_names:
            model = registry.get_model(name)
            if model:
                models_data.append(model)
                table.add_column(name, style="yellow", justify="right")
        
        if not models_data:
            console.print("[red]No valid models found[/red]")
            sys.exit(1)
        
        # Add metric rows
        metrics = ["mae", "rmse", "r2", "params", "epochs"]
        metric_labels = ["MAE (km/h)", "RMSE (km/h)", "R²", "Parameters", "Epochs"]
        
        for metric, label in zip(metrics, metric_labels):
            row = [label]
            for model in models_data:
                value = model.get(metric, 0)
                if metric in ["mae", "rmse", "r2"]:
                    row.append(f"{value:.4f}")
                else:
                    row.append(f"{value:,}")
            table.add_row(*row)
        
        console.print(table)
        
        # Find best model
        best_model = min(models_data, key=lambda m: m.get("mae", float("inf")))
        console.print(f"\n[green]Best model (lowest MAE):[/green] {best_model.get('name')} ({best_model.get('mae'):.4f} km/h)")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


# ============================================================================
# API COMMANDS
# ============================================================================

@cli.group()
def api():
    """API server management"""
    pass


@api.command("start")
@click.option("--host", default="0.0.0.0", help="Host address")
@click.option("--port", default=8000, help="Port number")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def api_start(host, port, reload):
    """Start the FastAPI server"""
    try:
        import subprocess
        
        console.print(f"[cyan]Starting API server on {host}:{port}...[/cyan]")
        
        cmd = ["uvicorn", "traffic_api.main:app", f"--host={host}", f"--port={port}"]
        if reload:
            cmd.append("--reload")
        
        subprocess.run(cmd, cwd=PROJECT_ROOT)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@api.command("status")
def api_status():
    """Check API server status"""
    try:
        import requests
        
        console.print("[cyan]Checking API server...[/cyan]")
        
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            data = response.json()
            
            status_text = f"""
[green]API Server: Online[/green]

Status: {data.get('status', 'unknown')}
Model Loaded: {data.get('model_loaded', False)}
Device: {data.get('device', 'unknown')}
Timestamp: {data.get('timestamp', 'N/A')}
            """
            
            console.print(Panel(status_text, title="API Status", border_style="green"))
            
        except requests.exceptions.RequestException:
            console.print("[red]API Server: Offline[/red]")
            console.print("Start with: [cyan]stmgt api start[/cyan]")
            sys.exit(1)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@api.command("test")
@click.option("--endpoint", default="/health", help="Endpoint to test")
def api_test(endpoint):
    """Test API endpoint"""
    try:
        import requests
        
        url = f"http://localhost:8000{endpoint}"
        console.print(f"[cyan]Testing: {url}[/cyan]")
        
        response = requests.get(url, timeout=10)
        
        console.print(f"\n[yellow]Status Code:[/yellow] {response.status_code}")
        console.print(f"[yellow]Response Time:[/yellow] {response.elapsed.total_seconds():.3f}s")
        console.print("\n[yellow]Response:[/yellow]")
        console.print_json(data=response.json())
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


# ============================================================================
# TRAINING COMMANDS
# ============================================================================

@cli.group()
def train():
    """Training management"""
    pass


@train.command("status")
def train_status():
    """Show current training status"""
    try:
        from dashboard.realtime_stats import get_training_stats
        
        stats = get_training_stats()
        
        if not stats:
            console.print("[yellow]No active training found[/yellow]")
            return
        
        status_text = f"""
[cyan]Training Run:[/cyan] {stats.get('run_name', 'N/A')}
[cyan]Status:[/cyan] {stats.get('status', 'unknown')}

[yellow]Progress:[/yellow]
  Epoch: {stats.get('current_epoch', 0)}/{stats.get('total_epochs', 0)}
  Progress: {stats.get('progress', 0):.1f}%
  
[yellow]Current Metrics:[/yellow]
  Train Loss: {stats.get('train_loss', 0):.4f}
  Val Loss:   {stats.get('val_loss', 0):.4f}
  Val MAE:    {stats.get('val_mae', 0):.4f} km/h

[yellow]Best Metrics:[/yellow]
  Best Val Loss: {stats.get('best_val_loss', 0):.4f}
  Best Epoch:    {stats.get('best_epoch', 0)}

[yellow]Time:[/yellow]
  Elapsed: {stats.get('elapsed_time', 'N/A')}
  Estimated Remaining: {stats.get('estimated_remaining', 'N/A')}
        """
        
        console.print(Panel(status_text, title="Training Status", border_style="cyan"))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@train.command("logs")
@click.option("--lines", default=20, help="Number of lines to show")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
def train_logs(lines, follow):
    """Show training logs"""
    try:
        log_file = PROJECT_ROOT / "logs" / "training.log"
        
        if not log_file.exists():
            console.print("[yellow]No training logs found[/yellow]")
            return
        
        if follow:
            # Tail -f equivalent
            import subprocess
            subprocess.run(["tail", "-f", str(log_file)])
        else:
            # Show last N lines
            with open(log_file, "r") as f:
                all_lines = f.readlines()
                for line in all_lines[-lines:]:
                    console.print(line.strip())
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


# ============================================================================
# DATA COMMANDS
# ============================================================================

@cli.group()
def data():
    """Data management"""
    pass


@data.command("info")
def data_info():
    """Show dataset information"""
    try:
        import pandas as pd
        
        data_file = PROJECT_ROOT / "data" / "processed" / "all_runs_extreme_augmented.parquet"
        
        if not data_file.exists():
            console.print("[red]Data file not found[/red]")
            sys.exit(1)
        
        df = pd.read_parquet(data_file)
        
        info_text = f"""
[cyan]Dataset:[/cyan] all_runs_extreme_augmented.parquet

[yellow]Shape:[/yellow]
  Rows:    {len(df):,}
  Columns: {len(df.columns)}

[yellow]Columns:[/yellow]
  {', '.join(df.columns)}

[yellow]Date Range:[/yellow]
  Start: {df['timestamp'].min()}
  End:   {df['timestamp'].max()}

[yellow]Statistics:[/yellow]
  Unique Edges: {df['node_a_id'].nunique() if 'node_a_id' in df.columns else 'N/A'}
  Avg Speed:    {df['speed_kmh'].mean():.2f} km/h
  Min Speed:    {df['speed_kmh'].min():.2f} km/h
  Max Speed:    {df['speed_kmh'].max():.2f} km/h

[yellow]File:[/yellow]
  Size: {data_file.stat().st_size / 1024 / 1024:.2f} MB
  Path: {data_file}
        """
        
        console.print(Panel(info_text, title="Dataset Information", border_style="cyan"))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


# ============================================================================
# SYSTEM COMMANDS
# ============================================================================

@cli.command("info")
def system_info():
    """Show system information"""
    try:
        import torch
        import psutil
        import platform
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        
        info_text = f"""
[cyan]System Information[/cyan]

[yellow]Hardware:[/yellow]
  CPU Usage:    {cpu_percent}%
  Memory:       {memory.percent}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)
  Disk:         {disk.percent}% ({disk.used / 1024**3:.1f}GB / {disk.total / 1024**3:.1f}GB)

[yellow]Software:[/yellow]
  OS:           {platform.system()} {platform.release()}
  Python:       {platform.python_version()}
  PyTorch:      {torch.__version__}
  CUDA:         {torch.cuda.is_available()}
  Device:       {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}

[yellow]Project:[/yellow]
  Root:         {PROJECT_ROOT}
  API:          http://localhost:8000
  Docs:         http://localhost:8000/docs
        """
        
        console.print(Panel(info_text, title="System Info", border_style="cyan"))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    cli()

import os
import sys
import yaml
import wandb
import subprocess
from pathlib import Path

# Get the absolute path to the repository root
repo_root = Path(__file__).parent.parent.absolute()

def main():
    # Load sweep configuration
    sweep_config_path = repo_root / "config" / "sweeps" / "wandb_sweep.yaml"
    
    with open(sweep_config_path, "r") as f:
        # Skip the hydra-specific parts by parsing only the wandb section
        sweep_yaml = yaml.safe_load(f)["wandb"]["sweep"]
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_yaml,
        project=sweep_yaml.get("project", "cathe_optimization"),
        entity=sweep_yaml.get("entity", "dmmiller597")
    )
    
    print(f"Sweep initialized with ID: {sweep_id}")
    print(f"View sweep at: https://wandb.ai/{sweep_yaml.get('entity', 'dmmiller597')}/{sweep_yaml.get('project', 'cathe_optimization')}/sweeps/{sweep_id}")
    
    # Number of agents to run
    num_agents = 1
    if len(sys.argv) > 1:
        num_agents = int(sys.argv[1])
    
    # Launch agents
    for i in range(num_agents):
        # Each agent will run a Hydra command with the sweep configuration
        cmd = f"python {repo_root}/src/train.py --config-name=sweeps/wandb_sweep"
        
        # Run the agent in a separate process
        process = subprocess.Popen(cmd, shell=True)
        print(f"Started agent {i+1} with PID: {process.pid}")

if __name__ == "__main__":
    main() 
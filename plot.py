import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import os

experiments = []
if os.path.exists("experiments.jsonl"):
    with open("experiments.jsonl") as f:
        for line in f:
            experiments.append(json.loads(line))

if not experiments:
    print("no experiments yet")
else:
    ids = [e["id"] for e in experiments]
    losses = [e["val_loss"] for e in experiments]
    kept = [e for e in experiments if e["kept"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(ids, losses, color="gray", alpha=0.3, label="Discarded", s=10) 
    if kept:
        kept_ids = [e["id"] for e in kept]
        kept_losses = [e["val_loss"] for e in kept]
        ax.scatter(kept_ids, kept_losses, color="lime", edgecolors="black", label="Kept", s=20, zorder=2)
        ax.step(kept_ids, kept_losses, color="green", where='post', alpha=0.5, zorder=1, label="Running best")
        for e in kept:
            ax.annotate(e["name"], (e["id"], e["val_loss"]), 
                        textcoords="offset points", xytext=(0, 5),
                        fontsize=8, rotation=45, ha='left', 
                        color="green", fontweight='normal')
            
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Validation Loss (lower is better)")
    ax.set_title(f"Research Progress: {len(experiments)} Experiments, {len(kept)} Kept Improvments")
    ax.legend()
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.3)
    ax.grid(True, which='both', axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments.png")
    print("saved experiments.png")
import json
import matplotlib.pyplot as plt
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
    
    plt.figure(figsize=(12, 6))
    plt.scatter(ids, losses, color="gray", alpha=0.5, label="Discarded")
    if kept:
        kept_ids = [e["id"] for e in kept]
        kept_losses = [e["val_loss"] for e in kept]
        plt.scatter(kept_ids, kept_losses, color="green", label="Kept")
        plt.plot(kept_ids, kept_losses, color="green")
    for e in experiments:
        plt.annotate(e["name"], (e["id"], e["val_loss"]), 
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=8, rotation=45, ha='left')
    plt.xlabel("Experiment #")
    plt.ylabel("Val Loss")
    plt.title(f"Experiments: {len(experiments)} total")
    plt.legend()
    plt.savefig("experiments.png")
    print("saved experiments.png")
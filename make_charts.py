"""Charts from the results CSV.
  python make_charts.py results_all.csv
Outputs: latency_bar.png (avg + P95 per config) and accuracy_vs_fps.png (mAP50-95 vs FPS; needs mAP filled)."""
import sys, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results_all.csv")
for c in ["FPS", "Avg ms", "P95 ms", "mAP50", "mAP50-95"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["cfg"] = df["Model"] + " " + df["Format"] + " " + df["Imgsz"].astype(str)

# 1) latency bar chart
d = df.dropna(subset=["Avg ms"]).sort_values("Avg ms")
fig, ax = plt.subplots(figsize=(11, max(6, 0.28 * len(d))))
y = range(len(d))
ax.barh([i + 0.2 for i in y], d["Avg ms"], height=0.4, label="Avg ms")
ax.barh([i - 0.2 for i in y], d["P95 ms"], height=0.4, label="P95 ms")
ax.axvline(100, color="red", ls="--", lw=1, label="P95 limit (100 ms)")
ax.set_yticks(list(y)); ax.set_yticklabels(d["cfg"], fontsize=7)
ax.set_xlabel("ms (lower is better)"); ax.set_title("Latency per configuration"); ax.legend()
plt.tight_layout(); plt.savefig("latency_bar.png", dpi=150); plt.close()

# 2) accuracy vs FPS scatter
s = df.dropna(subset=["FPS", "mAP50-95"])
if s.empty:
    print("No mAP values yet -> accuracy_vs_fps.png skipped (run benchmark_all.py without --no-acc)")
else:
    fig, ax = plt.subplots(figsize=(9, 6))
    for fmt, g in s.groupby("Format"):
        ax.scatter(g["FPS"], g["mAP50-95"], label=fmt, s=45)
    for _, r in s.iterrows():
        ax.annotate(f'{r["Model"]} {r["Imgsz"]}', (r["FPS"], r["mAP50-95"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axvline(10, color="red", ls="--", lw=1, label="10 FPS minimum")
    ax.set_xlabel("FPS"); ax.set_ylabel("mAP50-95 (person)"); ax.set_title("Accuracy vs FPS"); ax.legend()
    plt.tight_layout(); plt.savefig("accuracy_vs_fps.png", dpi=150); plt.close()
print("done")

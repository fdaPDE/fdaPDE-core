import os
import pandas as pd
import matplotlib.pyplot as plt

# === Cartella dei file CSV ===
csv_folder = "Comparisons/Statistics"
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

# === Metriche da analizzare ===
metrics = [
    "MinEdge", "MaxEdge", "MinArea", "MaxArea",
    "MinAngle", "MaxAngle", "MaxAspectRatio", "MinAltitude"
]

# === Dizionario per raccogliere i dati ===
data = {metric: {} for metric in metrics}

# === Lettura dei file e popolamento ===
for csv_file in csv_files:
    filepath = os.path.join(csv_folder, csv_file)
    df = pd.read_csv(filepath)

    df_main = df[df["Metric"].isin(metrics)].copy()
    for _, row in df_main.iterrows():
        metric = row["Metric"]
        value = row["Value"]
        label = os.path.splitext(csv_file)[0]
        data[metric][label] = value

# === Generazione grafici a barre ===
output_folder = "Comparisons/Statistics/Plots"
os.makedirs(output_folder, exist_ok=True)

for metric, values in data.items():
    plt.figure(figsize=(7, 4))
    names = list(values.keys())
    vals = [float(v) for v in values.values()]
    plt.bar(names, vals)
    #plt.ylabel(metric)
    plt.ylim(0.9*min(vals), max(vals) * 1.1)  
    plt.title(f"{metric} Comparison Across Meshers")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{metric}.png")
    plt.close()

print("Plots saved in:", output_folder)

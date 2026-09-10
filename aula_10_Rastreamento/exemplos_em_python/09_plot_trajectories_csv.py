import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plota trajetórias salvas pelo exemplo CamShift+Kalman.")
    parser.add_argument("--csv", default="output/trajectory_camshift_kalman.csv")
    parser.add_argument("--output", default="output/trajectory_plot_from_csv.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = {"measured_x", "measured_y", "kalman_x", "kalman_y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV sem colunas obrigatórias: {missing}")

    plt.figure(figsize=(8, 5))
    plt.plot(df["measured_x"], df["measured_y"], "r.-", label="CamShift medido")
    plt.plot(df["kalman_x"], df["kalman_y"], "b.-", label="Kalman predito")
    plt.gca().invert_yaxis()
    plt.xlabel("x [pixels]")
    plt.ylabel("y [pixels]")
    plt.title("Trajetórias sobrepostas")
    plt.grid(True)
    plt.legend()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Gráfico salvo em: {out}")
    plt.show()


if __name__ == "__main__":
    main()

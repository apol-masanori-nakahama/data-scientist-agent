from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def ensure_sample_csv(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 400
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "feature_num": rng.normal(0, 1, n),
        "feature_cat": rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2]),
        "target": rng.choice([0, 1], size=n, p=[0.6, 0.4]),
    })
    df.to_csv(path, index=False)
    return path


def make_artifacts(csv_path: Path, artifacts: Path) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    y = df["target"].astype("category").cat.codes
    X = pd.get_dummies(df.drop(columns=["target"]), drop_first=True).fillna(0.0).astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(confusion_matrix(yte, ypred)).plot(ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(artifacts / "model_confusion.png", dpi=200)
    plt.close(fig)
    # Simple PDP-like line plots for up to 5 features
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    top = list(num_cols[:5])
    for feat in top:
        s = X[feat].astype(float)
        xs = np.linspace(float(np.percentile(s, 5)), float(np.percentile(s, 95)), 25)
        preds = []
        Xtmp = Xtr.copy()
        for v in xs:
            Xtmp[feat] = v
            preds.append(float(clf.predict_proba(Xtmp)[:, 1].mean()))
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(xs, preds)
        ax.set_title(f"PDP: {feat}")
        fig.tight_layout()
        fig.savefig(artifacts / f"model_pdp_{feat}.png", dpi=200)
        plt.close(fig)


def make_readme_screenshots(artifacts: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # collage of 6 images
    sel = []
    if (artifacts / "model_confusion.png").exists():
        sel.append(artifacts / "model_confusion.png")
    sel += sorted(artifacts.glob("model_pdp_*.png"))
    sel = sel[:6]
    if sel:
        fig = plt.figure(figsize=(12, 7))
        for i in range(6):
            ax = fig.add_subplot(2, 3, i + 1)
            if i < len(sel):
                img = plt.imread(str(sel[i]))
                ax.imshow(img)
                ax.set_title(sel[i].name, fontsize=8)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(outdir / "modeling_figures_inline.png", dpi=200)
        plt.close(fig)
    # insights image from text (if exists)
    ins = artifacts / "insights.md"
    text = ins.read_text(encoding="utf-8") if ins.exists() else "Insights will appear here after running the UI."
    fig = plt.figure(figsize=(10, 14))
    plt.axis("off")
    plt.text(0.01, 0.99, text, va="top", ha="left", fontsize=10, family="monospace", wrap=True)
    fig.tight_layout()
    fig.savefig(outdir / "insights_llm.png", dpi=200)
    plt.close(fig)


def main():
    root = Path.cwd()
    csv = ensure_sample_csv(root / "data/sample.csv")
    artifacts = root / "data/artifacts"
    make_artifacts(csv, artifacts)
    make_readme_screenshots(artifacts, root / "docs/screenshots")
    print("Screenshots updated under docs/screenshots/")


if __name__ == "__main__":
    main()



from __future__ import annotations
import os
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.inspection import permutation_importance
import statsmodels.api as sm


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], out_png: str, labels: Optional[Sequence[str]] = None) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format='d', cmap='Blues', ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def save_roc_pr_curves(y_true: np.ndarray, y_score: np.ndarray, out_roc_png: str, out_pr_png: str) -> None:
    # binary only
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curve'); ax.legend()
    fig.tight_layout(); fig.savefig(out_roc_png); plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('PR Curve')
    fig.tight_layout(); fig.savefig(out_pr_png); plt.close(fig)


def save_permutation_importance_png(model, X: pd.DataFrame, y: np.ndarray, out_png: str, n_repeats: int = 5, random_state: int = 42) -> None:
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    importances = pd.Series(r["importances_mean"], index=list(X.columns))
    top = importances.abs().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(6, max(3, len(top) * 0.28)))
    top.iloc[::-1].plot(kind='barh', ax=ax)
    ax.set_title('Permutation Importance (mean)')
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


def save_slice_metrics_csv(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, ycol: str, out_csv: str) -> None:
    # Align arrays to df index to avoid positional/index mismatches
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true, index=df.index)
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=df.index)
    rows = []
    for col in df.columns:
        if col == ycol:
            continue
        if not (df[col].dtype == 'object' or str(df[col].dtype).startswith('category')):
            continue
        for val, idx in df.groupby(col).groups.items():
            idx_list = list(idx)
            yt = y_true.loc[idx_list].to_numpy()
            yp = y_pred.loc[idx_list].to_numpy()
            if len(yt) == 0:
                continue
            acc = float((yt == yp).mean())
            # macro-f1 for binary-ish slices
            p = np.mean(yt)
            f1 = float((2 * (yt & (yp == 1)).sum()) / (max(1, (yt.sum() + (yp == 1).sum())))) if p in (0.0, 1.0) else None
            rows.append({'feature': col, 'value': str(val), 'count': int(len(idx_list)), 'accuracy': acc, 'f1_binary_like': f1})
    cols = ['feature','value','count','accuracy','f1_binary_like']
    df_rows = pd.DataFrame(rows, columns=cols)
    if df_rows.empty:
        df_rows.to_csv(out_csv, index=False)
        return
    df_rows.sort_values(['feature', 'count'], ascending=[True, False]).to_csv(out_csv, index=False)


def save_threshold_curve(y_true: np.ndarray, y_score: np.ndarray, out_png: str) -> float:
    thresholds = np.linspace(0.0, 1.0, 101)
    f1s = []
    for t in thresholds:
        yp = (y_score >= t).astype(int)
        tp = int(((yp == 1) & (y_true == 1)).sum())
        fp = int(((yp == 1) & (y_true == 0)).sum())
        fn = int(((yp == 0) & (y_true == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        f1s.append(f1)
    best_idx = int(np.argmax(f1s))
    best_t = float(thresholds[best_idx])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, f1s)
    ax.axvline(best_t, color='red', linestyle='--', label=f'best={best_t:.2f}')
    ax.set_xlabel('threshold'); ax.set_ylabel('F1'); ax.set_title('Threshold optimization (F1)'); ax.legend()
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)
    return best_t


def save_partial_dependence_png(model, X: pd.DataFrame, features: list[str], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for feat in features:
        if feat not in X.columns:
            continue
        xs = np.linspace(float(np.percentile(X[feat], 5)), float(np.percentile(X[feat], 95)), 25)
        preds = []
        for v in xs:
            Xtmp = X.copy()
            Xtmp[feat] = v
            if hasattr(model, 'predict_proba'):
                p = model.predict_proba(Xtmp)[:, 1].mean()
            else:
                p = model.predict(Xtmp).mean()
            preds.append(float(p))
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(xs, preds)
        ax.set_title(f'PDP: {feat}')
        ax.set_xlabel(feat); ax.set_ylabel('mean prediction')
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f'model_pdp_{feat}.png')); plt.close(fig)


def save_partial_correlation_csv(X_num: pd.DataFrame, out_csv: str) -> None:
    if X_num.shape[1] < 2:
        pd.DataFrame().to_csv(out_csv, index=False)
        return
    std = X_num.std(ddof=0).replace(0, np.nan)
    Z = (X_num - X_num.mean()) / std
    Z = Z.fillna(0.0)
    cov = np.cov(Z.values, rowvar=False)
    prec = np.linalg.pinv(cov)
    D = np.sqrt(np.diag(prec))
    P = -prec / (D[:, None] * D[None, :])
    np.fill_diagonal(P, 1.0)
    dfp = pd.DataFrame(P, index=X_num.columns, columns=X_num.columns)
    dfp.to_csv(out_csv, index=True)


def save_missing_impact_csv(df: pd.DataFrame, ycol: str, out_csv: str) -> None:
    rows = []
    y = df[ycol]
    is_classification = y.dtype == 'object' or y.nunique() <= max(20, int(0.05 * len(y)))
    y_bin = (y.astype('category').cat.codes if is_classification else y.astype(float))
    for col in df.columns:
        if col == ycol:
            continue
        miss = df[col].isna()
        rate = float(miss.mean())
        if rate == 0.0:
            continue
        if is_classification:
            delta = float(y_bin[~miss].mean() - y_bin[miss].mean())
        else:
            delta = float(y_bin[~miss].mean() - y_bin[miss].mean())
        rows.append({'column': col, 'missing_rate': rate, 'target_mean_diff_notnull_minus_null': delta})
    cols = ['column', 'missing_rate', 'target_mean_diff_notnull_minus_null']
    dfr = pd.DataFrame(rows, columns=cols)
    if dfr.empty:
        dfr.to_csv(out_csv, index=False)
        return
    dfr.sort_values('missing_rate', ascending=False).to_csv(out_csv, index=False)



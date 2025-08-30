# src/agents/model_agent.py
from __future__ import annotations
from typing import Literal, Tuple, Callable, Optional
import pandas as pd
from pandas.api.types import is_numeric_dtype
import os, numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np, json, os

Task = Literal["regression","classification","timeseries"]

def infer_task_and_target(df: pd.DataFrame, hint: str | None = None) -> Tuple[Task, str]:
    if hint and hint in df.columns:  # 明示指定最優先
        y = hint
        task: Task = "classification" if df[hint].dtype == "object" or df[hint].nunique() <= max(20, int(0.05*len(df))) else "regression"
        return task, y
    # 単純規則：二値/少数カテゴリは分類、数値連続値は回帰、indexが日付でラグが効きそうなら時系列
    for col in df.columns[::-1]:
        if df[col].dtype == "object" and df[col].nunique()<min(30, int(0.1*len(df))):
            return "classification", col
    num = [c for c in df.columns if is_numeric_dtype(df[c])]
    if num:
        return "regression", num[-1]
    return "classification", df.columns[-1]

def train_candidates(
    df: pd.DataFrame,
    task: Task,
    ycol: str,
    random_state: int = 42,
    progress: Optional[Callable[[str, float], None]] = None,
) -> dict:
    """Train candidate models and report simple progress.

    Parameters
    - df, task, ycol, random_state: usual training args
    - progress: optional callback(stage: str, frac: float in [0,1])
    """
    fast = os.getenv("FAST_TEST", "0") == "1"
    cv_folds = 2 if fast else 5
    rf_reg_estimators = 50 if fast else 200
    rf_clf_estimators = 80 if fast else 300
    parallel_jobs = 1 if fast or os.getenv("NO_MULTIPROC", "0") == "1" else -1
    if progress:
        progress("preprocess: selecting/engineering features", 0.02)
    # ベースは数値特徴量のみ
    X_num = df.drop(columns=[ycol]).select_dtypes(exclude="object").fillna(0.0)
    # 数値特徴が無い場合のフォールバック: one-hot エンコード
    if X_num.shape[1] == 0:
        X = pd.get_dummies(df.drop(columns=[ycol]), drop_first=True).fillna(0.0)
    else:
        X = X_num
    X = robust_preprocess_features(X)
    y = df[ycol]
    if progress:
        progress("baseline computation", 0.08)
    res = {}
    if task == "regression":
        base = np.full_like(y, y.mean(), dtype=float)
        rmse_base = float(np.sqrt(mean_squared_error(y, base)))
        models = {
            "linreg": LinearRegression(),
            "rf_reg": RandomForestRegressor(n_estimators=rf_reg_estimators, random_state=random_state)
        }
        total_steps = len(models) + 2  # preprocess+baseline + each model + select best
        step_idx = 2
        for name, m in models.items():
            if progress:
                progress(f"cross-val: {name}", max(0.1, step_idx/total_steps))
            folds = min(cv_folds, max(2, min(len(X), 5)))
            scores = cross_val_score(m, X, y, cv=folds, scoring="neg_root_mean_squared_error", n_jobs=parallel_jobs)
            res[name] = {"rmse": float(-scores.mean()), "rmse_base": rmse_base}
            step_idx += 1
    elif task == "classification":
        y_enc = y.astype("category").cat.codes
        base = np.full_like(y_enc, y_enc.mode()[0])
        f1b = float(f1_score(y_enc, base, average="macro"))
        models = {
            "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "rf_clf": RandomForestClassifier(n_estimators=rf_clf_estimators, random_state=random_state)
        }
        total_steps = len(models) + 2
        step_idx = 2
        for name, m in models.items():
            if progress:
                progress(f"cross-val: {name}", max(0.1, step_idx/total_steps))
            folds = min(cv_folds, max(2, min(len(X), 5)))
            scores = cross_val_score(m, X, y_enc, cv=folds, scoring="f1_macro", n_jobs=parallel_jobs)
            res[name] = {"f1_macro": float(scores.mean()), "f1_base": f1b}
            step_idx += 1
    else:
        # 簡易時系列：ラグ特徴量で回帰
        y = df[ycol].astype(float)
        for lag in [1,2,3]:
            df[f"lag{lag}"] = df[ycol].shift(lag)
        dff = df.dropna()
        X, y = dff.drop(columns=[ycol]), dff[ycol]
        models = {"linreg_lag": LinearRegression(), "rf_reg_lag": RandomForestRegressor(n_estimators=200, random_state=random_state)}
        res = {}
        total_steps = len(models) + 2
        step_idx = 2
        for name, m in models.items():
            if progress:
                progress(f"time series CV: {name}", max(0.1, step_idx/total_steps))
            n_splits = min(3 if fast else 5, max(2, min(len(X), 5))-1)
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = cross_val_score(m, X, y, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=parallel_jobs)
            res[name] = {"rmse": float(-scores.mean())}
            step_idx += 1
    # ベスト選択
    best = min(res, key=lambda k: res[k].get("rmse", 1e9) if task!="classification" else -res[k].get("f1_macro", -1e9))
    if progress:
        progress("model selection", 1.0)
    return {"task": task, "target": ycol, "scores": res, "best": best}


def robust_preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """Trim extreme outliers and drop constant/degenerate columns.

    - Clip numeric columns to [q0.5%, q99.5%] (winsorization)
    - Replace inf/-inf with NaN, then fill with median (or 0 if all NaN)
    - Drop columns with zero variance or single unique value
    """
    if X.empty:
        return X
    Xc = X.copy()
    # numeric clip
    num_cols = [c for c in Xc.columns if is_numeric_dtype(Xc[c])]
    for c in num_cols:
        s = pd.to_numeric(Xc[c], errors="coerce")
        lo, hi = s.quantile(0.005), s.quantile(0.995)
        if pd.isna(lo) or pd.isna(hi) or lo >= hi:
            # median fill only
            med = s.median()
            Xc[c] = s.replace([np.inf, -np.inf], np.nan).fillna(0.0 if pd.isna(med) else med)
        else:
            s = s.clip(lo, hi)
            s = s.replace([np.inf, -np.inf], np.nan)
            med = s.median()
            Xc[c] = s.fillna(0.0 if pd.isna(med) else med)
    # drop constant/zero-variance columns
    keep = []
    for c in Xc.columns:
        vals = Xc[c]
        if is_numeric_dtype(vals):
            if float(vals.std(ddof=0)) > 1e-12 and vals.nunique(dropna=True) > 1:
                keep.append(c)
        else:
            if vals.nunique(dropna=True) > 1:
                keep.append(c)
    if keep:
        Xc = Xc[keep]
    return Xc

# src/agents/model_agent.py （続き）
MODEL_SYSTEM = """You are a modeling coach.
Given scores and logs (and optionally analyst insights), suggest at most 2 python steps to improve validation score (feature engineering or model tweak).
Use only pandas/numpy/sklearn and save helpful plots/CSVs under data/artifacts/.
Return a JSON list of {"action":"python","code": "..."} steps.
"""

def reflect_and_improve(llm, scores_json: str, extra_context: str | None = None) -> list[dict]:
    """Ask LLM for concrete improvement steps.

    Parameters
    - llm: chat-capable client
    - scores_json: modeling result as JSON string
    - extra_context: optional text to provide (e.g., LLM insights)
    """
    user_content = scores_json if not extra_context else (scores_json + "\n\nINSIGHTS:\n" + extra_context)
    msg = [{"role":"system","content":MODEL_SYSTEM},{"role":"user","content":user_content}]
    import json as _json
    try:
        return _json.loads(llm.chat(msg))
    except Exception:
        return []

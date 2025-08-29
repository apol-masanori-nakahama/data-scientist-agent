# app.py を EDAモードに更新
import sys, json
from src.core.types import Step, RunLog
from src.llm.client import LLMClient
from src.planning.loop import execute_plan
from src.agents.eda_agent import initial_eda_plan
from src.planning.loop import next_eda_plan
from src.reports.report import render_eda
try:
    import streamlit as st  # type: ignore[reportMissingImports]
except Exception:
    st = None  # type: ignore[assignment]
import os, json, time
from src.agents.model_agent import infer_task_and_target, train_candidates
from dotenv import load_dotenv


load_dotenv()

sample = False

if st:
    st.set_page_config(page_title="AI EDA Agent", layout="wide")
    st.title("Autonomous EDA & Modeling Agent")
    with st.sidebar:
        st.header("設定")
        up = st.file_uploader("CSVをアップロード", type=["csv"])
        sample = st.checkbox("サンプルデータで試す", value=False)
        run = st.button("実行")
else:
    up = None
    run = False
if st and run and (up or sample):
    csv_path = "data/upload.csv"
    if up:
        with open(csv_path, "wb") as f: f.write(up.read())
    else:
        import pandas as pd
        os.makedirs("data", exist_ok=True)
        import numpy as np
        n = 300
        df = pd.DataFrame({
            "feature_num": np.random.randn(n)*2+3,
            "feature_cat": np.random.choice(["A","B","C"], size=n, p=[0.5,0.3,0.2]),
            "target": np.random.choice([0,1], size=n)
        })
        df.to_csv(csv_path, index=False)
    st.write("実行開始…")
    llm = LLMClient()
    log = RunLog(meta={"input_csv": csv_path})

    eda_tab, figs_tab, desc_tab, artifacts_tab = st.tabs(["Overview","Figures","Describe","Artifacts"])

    with st.status("EDA中...", expanded=True) as status:
        steps = initial_eda_plan(llm)
        prog = st.progress(0.0)
        logbox = st.empty()
        report_box = st.empty()
        total = max(1, len(steps))
        def _on(idx, step, obs):
            p = (idx+1)/total
            prog.progress(min(1.0, p))
            logbox.write({
                "step": idx+1,
                "action": step.action,
                "ok": obs.success,
                "stdout": obs.stdout[-400:],
                "stderr": obs.stderr[-400:]
            })
        log = execute_plan(steps, csv_path, log, on_step=_on)
        for i in range(2):
            steps = next_eda_plan(llm, log)
            total = max(1, len(steps))
            log = execute_plan(steps, csv_path, log, on_step=_on)
        md = render_eda(csv_path)
        with eda_tab:
            st.subheader("レポート")
            report_box.markdown(open("data/artifacts/eda_report.md","r",encoding="utf-8").read(), unsafe_allow_html=True)
            # Inline figures fallback (Markdown image links may not render reliably)
            from pathlib import Path as _P
            _art = _P("data/artifacts")
            _imgs = sorted(_art.glob("cell_hist_*.png"))
            if _imgs:
                st.subheader("Figures (inline)")
                _cols = st.columns(3)
                for _i, _p in enumerate(_imgs):
                    with _cols[_i % 3]:
                        st.image(str(_p), caption=_p.name)
            _mimgs = sorted(_art.glob("model_*.png"))
            if _mimgs:
                st.subheader("Modeling Figures (inline)")
                _cols2 = st.columns(3)
                for _i, _p in enumerate(_mimgs):
                    with _cols2[_i % 3]:
                        st.image(str(_p), caption=_p.name)

    with st.status("モデリング中...", expanded=True) as mstatus:
        import pandas as pd  # type: ignore[reportMissingImports]
        try:
            df = pd.read_csv(csv_path)
            st.write("データ読込完了", df.shape)
            task, ycol = infer_task_and_target(df)
            st.write("推定タスク", task, "/ 目的変数", ycol)
            res = train_candidates(df, task, ycol)
            with eda_tab:
                st.subheader("モデル評価")
                st.json(res)
            open("data/artifacts/model_scores.json","w").write(json.dumps(res,ensure_ascii=False,indent=2))
            # 追加の分析
            try:
                import numpy as np
                from sklearn.model_selection import train_test_split
                from sklearn.linear_model import LogisticRegression
                from src.agents.error_analysis import (
                    ensure_dir, save_confusion_matrix, save_roc_pr_curves, save_permutation_importance_png, save_slice_metrics_csv,
                    save_threshold_curve, save_partial_dependence_png, save_partial_correlation_csv, save_missing_impact_csv
                )
                import warnings as _warnings
                _warnings.filterwarnings("ignore", module="sklearn.utils.extmath")
                ensure_dir("data/artifacts")
                y = df[ycol]
                X = pd.get_dummies(df.drop(columns=[ycol]), drop_first=True).fillna(0.0)
                y_codes = y.astype('category').cat.codes
                mask = (y_codes >= 0)
                X = X.loc[mask]
                y_codes = y_codes.loc[mask]
                strat = y_codes if len(np.unique(y_codes)) > 1 else None
                Xtr, Xte, ytr, yte = train_test_split(X, y_codes, test_size=0.3, random_state=42, stratify=strat)
                clf = LogisticRegression(max_iter=2000)
                clf.fit(Xtr, ytr)
                ypred = clf.predict(Xte).astype(int)
                if len(np.unique(ytr)) == 2:
                    yscore = clf.predict_proba(Xte)[:,1]
                    if np.isfinite(yscore).all():
                        save_roc_pr_curves(yte, yscore, 'data/artifacts/model_roc.png', 'data/artifacts/model_pr.png')
                        save_threshold_curve(yte, yscore, 'data/artifacts/model_threshold.png')
                save_confusion_matrix(list(map(int, yte.tolist())), list(map(int, ypred.tolist())), 'data/artifacts/model_confusion.png', labels=list(df[ycol].astype('category').cat.categories))
                save_permutation_importance_png(clf, Xte, yte, 'data/artifacts/model_permutation_importance.png')
                save_slice_metrics_csv(df.loc[Xte.index], yte, ypred, ycol, 'data/artifacts/model_slice_metrics.csv')
                # PDP for top 3 numeric features
                top_num = list(X.columns[:3])
                save_partial_dependence_png(clf, Xte, top_num, 'data/artifacts')
                # Partial correlation among numeric features
                save_partial_correlation_csv(df.select_dtypes(include='number'), 'data/artifacts/model_partial_corr.csv')
                # Missingness impact
                save_missing_impact_csv(df, ycol, 'data/artifacts/model_missing_impact.csv')
                # LLM Insights
                try:
                    from pathlib import Path as _P
                    art = _P('data/artifacts'); art.mkdir(parents=True, exist_ok=True)
                    ctx_parts = []
                    ctx_parts.append('Model scores: ' + json.dumps(res, ensure_ascii=False))
                    for _fn in ['model_slice_metrics.csv','model_missing_impact.csv','model_partial_corr.csv']:
                        _p = art/_fn
                        if _p.exists():
                            try:
                                _dfh = pd.read_csv(_p).head(20).to_string(index=False)
                                ctx_parts.append(f'{_fn} (head):\n{_dfh}')
                            except Exception:
                                pass
                    from src.agents.explain import generate_insights
                    _ins = generate_insights(llm, "\n\n".join(ctx_parts), rounds=5)
                    (art/'insights.md').write_text(_ins, encoding='utf-8')
                    with eda_tab:
                        st.subheader('Insights (LLM)')
                        st.markdown(_ins)
                        st.download_button('Insights Markdown', data=_ins.encode('utf-8'), file_name='insights.md')
                except Exception as _ie:
                    st.warning(f"Insights生成で警告: {_ie}")
            except Exception as ee:
                st.warning(f"追加分析で警告: {ee}")
            mstatus.update(label="モデリング完了", state="complete")
            # レポートを再レンダ（モデリング図を含めるため）
            md = render_eda(csv_path)
            with eda_tab:
                report_box.markdown(open("data/artifacts/eda_report.md","r",encoding="utf-8").read(), unsafe_allow_html=True)
            # モデリング図のインライン描画（モデル生成後に改めて表示）
            from pathlib import Path as _P2
            _art2 = _P2('data/artifacts')
            _mimgs2 = sorted(_art2.glob('model_*.png'))
            if _mimgs2:
                with eda_tab:
                    st.subheader("Modeling Figures (inline)")
                    _colsM = st.columns(3)
                    for _i, _p in enumerate(_mimgs2):
                        with _colsM[_i % 3]:
                            st.image(str(_p), caption=_p.name)
        except Exception as e:
            st.error(f"モデリングでエラー: {e}")

    from pathlib import Path
    art = Path("data/artifacts")
    with figs_tab:
        st.subheader("図表")
        imgs = sorted(art.glob("cell_hist_*.png"))
        if imgs:
            cols = st.columns(3)
            for i, p in enumerate(imgs):
                with cols[i%3]:
                    st.image(str(p), caption=p.name)
        else:
            st.info("図がまだありません")
    with desc_tab:
        st.subheader("describe")
        desc = art / "cell_desc.csv"
        if desc.exists():
            import pandas as pd
            st.dataframe(pd.read_csv(desc))
        else:
            st.info("describe がまだありません")
    with artifacts_tab:
        st.subheader("ダウンロード")
        if (art/"eda_report.md").exists():
            st.download_button("EDA Markdown", data=open(art/"eda_report.md","rb").read(), file_name="eda_report.md")
        if (art/"model_scores.json").exists():
            st.download_button("Model Scores JSON", data=open(art/"model_scores.json","rb").read(), file_name="model_scores.json")
    
if __name__ == "__main__" and os.getenv("CLI","0") == "1":
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/sample.csv"
    log = RunLog(meta={"input_csv": csv})
    try:
        llm = LLMClient()
    except Exception:
        llm = None  # type: ignore
    has_llm = bool(llm) and getattr(llm, "_client", None) is not None

    # 1) 初期プラン実行（LLMなしならfew-shotにフォールバック）
    if has_llm:
        steps = initial_eda_plan(llm)
    else:
        from src.agents.eda_agent import EDA_FEWSHOT
        steps = [Step(**s) for s in EDA_FEWSHOT]
    log = execute_plan(steps, csv, log)

    # 2) 反省→再計画を最大2回
    for _ in range(2):
        if has_llm:
            steps = next_eda_plan(llm, log)
            log = execute_plan(steps, csv, log)
            if any(s.action=="stop" for s in steps):
                break
        else:
            break

    # 3) レポート化
    path = render_eda(csv)
    print(json.dumps({"report_md": path, "turns": len(log.turns)}, ensure_ascii=False))
    # 4) モデリング（CLI）
    import pandas as pd  # type: ignore[reportMissingImports]
    df = pd.read_csv(csv)
    task, ycol = infer_task_and_target(df, hint=os.getenv("TARGET"))
    result = train_candidates(df, task, ycol)
    print("Modeling:", json.dumps(result, ensure_ascii=False))
    if has_llm:
        try:
            from src.agents.model_agent import reflect_and_improve
            from src.runners.code_runner import run_python
            for _ in range(2):
                steps2 = reflect_and_improve(llm, json.dumps(result, ensure_ascii=False))
                if not steps2:
                    break
                for s in steps2:
                    if s.get("action")=="python":
                        run_python(s.get("code",""), input_csv=csv)
                df = pd.read_csv(csv)
                result = train_candidates(df, task, ycol)
        except Exception:
            pass
    open("data/artifacts/model_scores.json","w",encoding="utf-8").write(json.dumps(result, ensure_ascii=False, indent=2))
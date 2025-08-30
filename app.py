# app.py を EDAモードに更新
import sys, json
from src.core.types import Step, RunLog, ActionType
from typing import cast
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
            # LLM ステータスの可視化
            try:
                _prov = os.getenv("LLM_PROVIDER", "anthropic")
                _model = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
                st.caption(f"LLM provider: {_prov}")
                st.caption(f"LLM model: {_model}")
                _has = False
                try:
                    _tmp = LLMClient()
                    _has = _tmp.is_ready(deep=False)
                    _deep_ok = False
                    if _has:
                        # Deep check trigger
                        if st.button("LLM 接続確認", use_container_width=True):
                            _deep_ok = _tmp.is_ready(deep=True)
                    if _has:
                        st.success("LLM ready")
                        if _deep_ok:
                            st.caption("Deep check: OK")
                    else:
                        st.warning("LLM not ready (check API key and provider)")
                except Exception:
                    _has = False
                    st.warning("LLM not ready (init error)")
            except Exception:
                pass
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

    eda_tab, figs_tab, desc_tab, insights_tab, artifacts_tab = st.tabs(["Overview","Figures","Describe","Insights","Artifacts"])

    with st.status("EDA中...", expanded=True) as status:
        reflect_rounds_ui = 2
        phase_count = 1 + reflect_rounds_ui
        phase_idx = 0
        steps = initial_eda_plan(llm)
        prog = st.progress(0.0)
        logbox = st.empty()
        stagebox = st.empty()
        report_box = st.empty()
        total = max(1, len(steps))
        def _on(idx, step, obs):
            # 各フェーズを同じ重みで進捗化
            local = (idx + 1) / max(1, total)
            overall = (phase_idx + local) / phase_count
            status.update(label=f"EDA中... {int(overall*100)}% (Phase {phase_idx+1}/{phase_count})")
            prog.progress(min(1.0, overall))
            # ステージ表示（action とコード/ノート先頭）
            preview = (step.note or (step.code or "").strip().splitlines()[0:1])
            if isinstance(preview, list):
                preview = preview[0] if preview else ""
            stagebox.write({
                "phase": f"{phase_idx+1}/{phase_count}",
                "step": f"{idx+1}/{total}",
                "action": step.action,
                "preview": (preview or "")[:120]
            })
            logbox.write({
                "ok": obs.success,
                "stdout": obs.stdout[-400:],
                "stderr": obs.stderr[-400:]
            })
        log = execute_plan(steps, csv_path, log, on_step=_on)
        phase_idx += 1
        for i in range(reflect_rounds_ui):
            steps = next_eda_plan(llm, log)
            total = max(1, len(steps))
            log = execute_plan(steps, csv_path, log, on_step=_on)
            phase_idx += 1
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
            mprog = st.progress(0.0)
            mstage = st.empty()
            def _mcb(stage: str, frac: float) -> None:
                mstatus.update(label=f"モデリング中... {int(max(0.0,min(1.0,frac))*100)}%")
                mprog.progress(min(1.0, max(0.0, frac)))
                mstage.write({"stage": stage, "progress": f"{frac*100:.1f}%"})
            res = train_candidates(df, task, ycol, progress=_mcb)
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
                mstatus.update(label="追加分析中...")
                subprog = st.progress(0.0)
                substage = st.empty()
                # Build fine-grained stage plan dynamically
                y = df[ycol]
                X = pd.get_dummies(df.drop(columns=[ycol]), drop_first=True).fillna(0.0)
                y_codes = y.astype('category').cat.codes
                mask = (y_codes >= 0)
                X = X.loc[mask]
                y_codes = y_codes.loc[mask]
                strat = y_codes if len(np.unique(y_codes)) > 1 else None
                top_num = list(X.columns[:3])
                binary = len(np.unique(y_codes)) == 2
                stages: list[str] = [
                    "prepare data",
                    "train/test split",
                    "fit(LogisticRegression)",
                    "predict labels",
                ]
                if binary:
                    stages += [
                        "predict probabilities",
                        "save ROC",
                        "save PR",
                        "save threshold curve",
                    ]
                stages += [
                    "save confusion matrix",
                    "permutation importance (compute+save)",
                    "slice metrics (save)",
                ]
                for f in top_num:
                    stages.append(f"save PDP: {f}")
                stages += [
                    "save partial correlation",
                    "save missingness impact",
                ]
                total_steps = len(stages)
                cur = [0]
                def _sub(stage: str):
                    subprog.progress(min(1.0, (cur[0]+1)/total_steps))
                    substage.write({"stage": stage, "step": f"{cur[0]+1}/{total_steps}"})
                    cur[0] += 1
                # Execute with detailed updates
                _sub("prepare data")
                # data prepared above
                _sub("train/test split")
                Xtr, Xte, ytr, yte = train_test_split(X, y_codes, test_size=0.3, random_state=42, stratify=strat)
                clf = LogisticRegression(max_iter=2000)
                _sub("fit(LogisticRegression)")
                clf.fit(Xtr, ytr)
                _sub("predict labels")
                ypred = clf.predict(Xte).astype(int)
                if binary:
                    _sub("predict probabilities")
                    yscore = clf.predict_proba(Xte)[:,1]
                    if np.isfinite(yscore).all():
                        _sub("save ROC")
                        save_roc_pr_curves(yte, yscore, 'data/artifacts/model_roc.png', 'data/artifacts/model_pr.png')
                        _sub("save PR")
                        # ROC/PRは同一呼出で両方保存するため、PR用のステップも表示
                        _ = None
                        _sub("save threshold curve")
                        save_threshold_curve(yte, yscore, 'data/artifacts/model_threshold.png')
                _sub("save confusion matrix")
                save_confusion_matrix(list(map(int, yte.tolist())), list(map(int, ypred.tolist())), 'data/artifacts/model_confusion.png', labels=list(df[ycol].astype('category').cat.categories))
                _sub("permutation importance (compute+save)")
                save_permutation_importance_png(clf, Xte, yte, 'data/artifacts/model_permutation_importance.png')
                _sub("slice metrics (save)")
                save_slice_metrics_csv(df.loc[Xte.index], yte, ypred, ycol, 'data/artifacts/model_slice_metrics.csv')
                # PDP per feature
                for f in top_num:
                    _sub(f"save PDP: {f}")
                    save_partial_dependence_png(clf, Xte, [f], 'data/artifacts')
                _sub("save partial correlation")
                save_partial_correlation_csv(df.select_dtypes(include='number'), 'data/artifacts/model_partial_corr.csv')
                _sub("save missingness impact")
                save_missing_impact_csv(df, ycol, 'data/artifacts/model_missing_impact.csv')
                # Insights 生成は後段で必ず実施（追加分析の失敗に影響させない）
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
            # ---- LLM Insights（追加分析の成否に関わらず実行） ----
            try:
                from pathlib import Path as _P
                art = _P('data/artifacts'); art.mkdir(parents=True, exist_ok=True)
                ctx_parts = []
                ctx_parts.append('Model scores: ' + json.dumps(res, ensure_ascii=False))
                for _fn in ['model_slice_metrics.csv','model_missing_impact.csv','model_partial_corr.csv']:
                    _p = art/_fn
                    if _p.exists():
                        try:
                            import pandas as _pd
                            _dfh = _pd.read_csv(_p).head(20).to_string(index=False)
                            ctx_parts.append(f'{_fn} (head):\n{_dfh}')
                        except Exception:
                            pass
                _ins = ""
                has_llm_ui = llm.is_ready(deep=False)
                with insights_tab:
                    st.subheader('Insights (LLM)')
                    if has_llm_ui:
                        with st.status("Insights 生成中...", expanded=True):
                            iprog = st.progress(0.0)
                            ilog = st.empty()
                            def _ipcb(round_idx: int, total: int) -> None:
                                frac = min(1.0, max(0.0, round_idx/float(total)))
                                iprog.progress(frac)
                                ilog.write({"round": f"{round_idx}/{total}", "status": "calling LLM"})
                            from src.agents.explain import generate_insights
                            _ins = generate_insights(llm, "\n\n".join(ctx_parts), rounds=5, progress=_ipcb)
                            (art/'insights.md').write_text(_ins or '', encoding='utf-8')
                            ilog.write({"done": True})
                    else:
                        # LLM未設定/無効時は既存ファイルを表示、なければ案内を表示
                        _p_ins = art/'insights.md'
                        if _p_ins.exists():
                            try:
                                _ins = _p_ins.read_text(encoding='utf-8')
                            except Exception:
                                _ins = ''
                        if not _ins.strip():
                            st.info('LLM が無効か未設定のため Insights は未生成です。OPENAI_API_KEY/ANTHROPIC_API_KEY と LLM_PROVIDER/LLM_MODEL を設定してください。')
                    if _ins.strip():
                        st.markdown(_ins)
                        st.download_button('Insights Markdown', data=_ins.encode('utf-8'), file_name='insights.md')

                # Reflect insights into follow-up improvements and re-train
                try:
                    if has_llm_ui and _ins.strip():
                        from src.agents.model_agent import reflect_and_improve
                        from src.runners.code_runner import run_python as _run_py
                        with insights_tab:
                            st.subheader("示唆の反映と再学習")
                            rprog = st.progress(0.0)
                            rlog = st.empty()
                            steps2 = reflect_and_improve(llm, json.dumps(res, ensure_ascii=False), extra_context=_ins)
                            if steps2:
                                for si, _s in enumerate(steps2):
                                    if _s.get('action') == 'python':
                                        rlog.write({"apply_step": si+1, "total": len(steps2)})
                                        _run_py(_s.get('code', ''), input_csv=csv_path)
                                        rprog.progress((si+1)/len(steps2))
                                # Re-train after applying suggestions
                                df2 = pd.read_csv(csv_path)
                                res2 = train_candidates(df2, task, ycol)
                                open("data/artifacts/model_scores.json","w").write(json.dumps(res2,ensure_ascii=False,indent=2))
                                with eda_tab:
                                    st.subheader("モデル評価（改善後）")
                                    st.json(res2)
                                # Re-render report and inline figures
                                md = render_eda(csv_path)
                                report_box.markdown(open("data/artifacts/eda_report.md","r",encoding="utf-8").read(), unsafe_allow_html=True)
                                from pathlib import Path as _P3
                                _art3 = _P3('data/artifacts')
                                _mimgs3 = sorted(_art3.glob('model_*.png'))
                                if _mimgs3:
                                    with eda_tab:
                                        st.subheader("Modeling Figures (inline, 改善後)")
                                        _colsM2 = st.columns(3)
                                        for _i, _p in enumerate(_mimgs3):
                                            with _colsM2[_i % 3]:
                                                st.image(str(_p), caption=_p.name)
                            else:
                                rlog.write({"info": "改善手順は生成されませんでした"})
                except Exception as _re:
                    with insights_tab:
                        st.warning(f"反映再実行で警告: {_re}")
            except Exception as _ie:
                with insights_tab:
                    st.warning(f"Insights生成で警告: {_ie}")
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
    # CLI では既定で LLM を使わない（ネット依存/遅延回避）。有効化するには USE_LLM=1 を設定。
    use_llm_cli = os.getenv("USE_LLM", "0") == "1"
    has_llm = bool(llm) and llm.is_ready(deep=False) and use_llm_cli

    # 1) 初期プラン実行（LLMなしならfew-shotにフォールバック）
    if has_llm:
        assert llm is not None
        steps = initial_eda_plan(llm)
    else:
        from src.agents.eda_agent import EDA_FEWSHOT
        steps = [
            Step(
                action=cast(ActionType, s.get("action")),
                code=s.get("code"),
                note=s.get("note")
            ) for s in EDA_FEWSHOT
        ]
    # CLI 進捗表示（簡易）
    phase_count = 1 + (2 if has_llm else 0)
    phase_idx = 0
    total = max(1, len(steps))
    def _on_cli(idx, step, obs):
        local = (idx + 1) / max(1, total)
        overall = (phase_idx + local) / phase_count
        preview = step.note or (step.code or "").strip().splitlines()[0:1]
        if isinstance(preview, list):
            preview = preview[0] if preview else ""
        print(json.dumps({
            "event": "eda_step",
            "phase": f"{phase_idx+1}/{phase_count}",
            "step": f"{idx+1}/{total}",
            "action": step.action,
            "ok": obs.success,
            "overall": round(overall, 3),
            "preview": (preview or "")[:120]
        }, ensure_ascii=False))
    log = execute_plan(steps, csv, log, on_step=_on_cli)

    # 2) 反省→再計画を最大2回
    for _ in range(2):
        if has_llm:
            assert llm is not None
            steps = next_eda_plan(llm, log)
            total = max(1, len(steps))
            phase_idx += 1
            log = execute_plan(steps, csv, log, on_step=_on_cli)
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

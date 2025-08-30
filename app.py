# app.py を EDAモードに更新
import sys, json
from src.core.types import Step, RunLog, ActionType
from typing import cast
from src.llm.client import LLMClient
from src.planning.loop import execute_plan
from src.agents.eda_agent import initial_eda_plan
from src.planning.loop import next_eda_plan
from src.reports.report import render_eda
from src.utils.integrated_progress import get_global_progress_system, reset_global_progress_system
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
            # Insights ラウンド数（UIで制御）
            try:
                _def_rounds = int(os.getenv("INSIGHT_ROUNDS", "3"))
            except Exception:
                _def_rounds = 3
            _opts = [1, 2, 3, 4, 5]
            _idx = _opts.index(_def_rounds) if _def_rounds in _opts else _opts.index(3)
            insight_rounds = st.selectbox(
                "Insights ラウンド数",
                _opts,
                index=_idx,
                help="LLM内省の反復回数（多いほど高品質だが遅くなります）",
            )
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
    
    # 統合進捗システムの初期化
    reset_global_progress_system()  # 前回の状態をクリア
    progress_system = get_global_progress_system()

    # 統合進捗ダッシュボードをStreamlitに追加
    progress_containers = progress_system.get_streamlit_containers()
    
    eda_tab, figs_tab, desc_tab, insights_tab, artifacts_tab, progress_tab = st.tabs(["Overview","Figures","Describe","Insights","Artifacts","進捗"])

    with st.status("EDA中...", expanded=True) as status:
        reflect_rounds_ui = 2
        phase_count = 1 + reflect_rounds_ui
        phase_idx = 0
        
        # EDA初期プラン
        progress_system.start_phase("EDA初期分析", message="データの基本的な探索を開始します")
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
            
            # 統合進捗システムに情報を送信
            step_name = f"{step.action}: {(preview or '')[:50]}"
            progress_system.update_step(step_name, f"ステップ {idx+1}/{total}")
        
        log = execute_plan(steps, csv_path, log, on_step=_on, 
                          progress_manager=progress_system.progress_manager, 
                          phase_name="EDA初期分析")
        
        phase_idx += 1
        
        # 反省・改善フェーズ
        for i in range(reflect_rounds_ui):
            progress_system.start_phase(f"EDA改善 ラウンド{i+1}", message=f"分析結果を改善します（{i+1}/{reflect_rounds_ui}）")
            steps = next_eda_plan(llm, log)
            total = max(1, len(steps))
            log = execute_plan(steps, csv_path, log, on_step=_on,
                              progress_manager=progress_system.progress_manager,
                              phase_name=f"EDA改善 ラウンド{i+1}")
            phase_idx += 1
        
        progress_system.complete_phase("EDA分析完了")
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
                has_llm_ui = (llm is not None) and (getattr(llm, '_client', None) is not None)
                # Prepare Overview tab placeholders so users see insights without switching tabs
                with eda_tab:
                    st.subheader('Insights (LLM)')
                    _draft_box_ov = st.empty()
                    _ov_prog = st.progress(0.0)
                    _ov_log = st.empty()
                    _ov_eta = st.empty()
                with insights_tab:
                    st.subheader('Insights (LLM)')
                    # Show intermediate drafts while generating (Insights tab)
                    _draft_box = st.empty()
                    if has_llm_ui:
                        with st.status("Insights 生成中...", expanded=True):
                            iprog = st.progress(0.0)
                            ilog = st.empty()
                            _eta_box = st.empty()
                            _round_start: dict[int, float] = {}
                            _round_times: dict[int, float] = {}
                            stream_accum = [""]
                            def _ipcb(round_idx: int, total: int) -> None:
                                # Called at the start of each round
                                frac = min(1.0, max(0.0, (round_idx-1)/float(total)))
                                iprog.progress(frac)
                                # Mirror to Overview tab
                                try:
                                    _ov_prog.progress(frac)
                                except Exception:
                                    pass
                                _round_start[round_idx] = time.time()
                                ilog.write({
                                    "round": f"{round_idx}/{total}",
                                    "status": "calling LLM",
                                    "started_at": time.strftime('%H:%M:%S')
                                })
                                try:
                                    _ov_log.write({
                                        "round": f"{round_idx}/{total}",
                                        "status": "calling LLM",
                                        "started_at": time.strftime('%H:%M:%S')
                                    })
                                except Exception:
                                    pass
                                # ETA based on previous rounds (if any)
                                if _round_times:
                                    avg = sum(_round_times.values())/len(_round_times)
                                    rem = max(0, total - (round_idx-1))
                                    _eta_box.info(f"ETA 約 {int(avg*rem)} 秒（平均 {avg:.1f}s/ラウンド）")
                                    try:
                                        _ov_eta.info(f"ETA 約 {int(avg*rem)} 秒（平均 {avg:.1f}s/ラウンド）")
                                    except Exception:
                                        pass
                                else:
                                    _eta_box.caption("ETA 計測中…（最初のラウンド完了後に推定）")
                                    try:
                                        _ov_eta.caption("ETA 計測中…（最初のラウンド完了後に推定）")
                                    except Exception:
                                        pass
                            def _idcb(draft_text: str, round_idx: int, total: int) -> None:
                                # Stream the evolving draft so users can see progress
                                _draft_box.markdown(draft_text or '')
                                try:
                                    _draft_box_ov.markdown(draft_text or '')
                                except Exception:
                                    pass
                                # Round finished timing and ETA update
                                stt = _round_start.get(round_idx, time.time())
                                dur = max(0.0, time.time() - stt)
                                _round_times[round_idx] = dur
                                ilog.write({
                                    "round": f"{round_idx}/{total}",
                                    "status": "round finished",
                                    "duration_sec": f"{dur:.1f}",
                                })
                                try:
                                    _ov_log.write({
                                        "round": f"{round_idx}/{total}",
                                        "status": "round finished",
                                        "duration_sec": f"{dur:.1f}",
                                    })
                                except Exception:
                                    pass
                                avg = sum(_round_times.values())/len(_round_times)
                                rem = max(0, total - round_idx)
                                _eta_box.info(f"ETA 約 {int(avg*rem)} 秒（平均 {avg:.1f}s/ラウンド）")
                                iprog.progress(min(1.0, round_idx/float(total)))
                                try:
                                    _ov_eta.info(f"ETA 約 {int(avg*rem)} 秒（平均 {avg:.1f}s/ラウンド）")
                                    _ov_prog.progress(min(1.0, round_idx/float(total)))
                                except Exception:
                                    pass
                                # Persist draft after each round as well
                                try:
                                    (art/f'insights_round_{round_idx}.md').write_text(draft_text or '', encoding='utf-8')
                                    (art/'insights.md').write_text(draft_text or '', encoding='utf-8')
                                except Exception:
                                    pass
                            def _istream(delta_text: str, round_idx: int, total: int) -> None:
                                # Token/fragment-level streaming update (more granular than on_draft)
                                stream_accum[0] += (delta_text or '')
                                _draft_box.markdown(stream_accum[0])
                                try:
                                    _draft_box_ov.markdown(stream_accum[0])
                                except Exception:
                                    pass
                            from src.agents.explain import generate_insights
                            _rounds = insight_rounds if 'insight_rounds' in locals() and insight_rounds is not None else int(os.getenv("INSIGHT_ROUNDS", "3"))
                            _ins = generate_insights(llm, "\n\n".join(ctx_parts), rounds=_rounds, progress=_ipcb, on_draft=_idcb, on_stream=_istream)
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
                        _draft_box.markdown(_ins)
                        st.download_button('Insights Markdown', data=_ins.encode('utf-8'), file_name='insights.md')
                # Also show final insights and download on Overview
                if _ins.strip():
                    with eda_tab:
                        _draft_box_ov.markdown(_ins)
                        st.download_button('Insights Markdown', data=_ins.encode('utf-8'), file_name='insights.md', key='dl_insights_overview')

                # Reflect insights into follow-up improvements and re-train
                try:
                    if has_llm_ui and _ins.strip():
                        from src.agents.model_agent import reflect_and_improve
                        from src.runners.code_runner import run_python as _run_py
                        # Prepare Overview placeholders
                        with eda_tab:
                            st.subheader("示唆の反映と再学習")
                            rprog_ov = st.progress(0.0)
                            rlog_ov = st.empty()
                        with insights_tab:
                            st.subheader("示唆の反映と再学習")
                            rprog = st.progress(0.0)
                            rlog = st.empty()
                            steps2 = reflect_and_improve(llm, json.dumps(res, ensure_ascii=False), extra_context=_ins)
                            if steps2:
                                for si, _s in enumerate(steps2):
                                    if _s.get('action') == 'python':
                                        msg = {"apply_step": si+1, "total": len(steps2)}
                                        rlog.write(msg)
                                        try:
                                            rlog_ov.write(msg)
                                        except Exception:
                                            pass
                                        _run_py(_s.get('code', ''), input_csv=csv_path)
                                        frac = (si+1)/len(steps2)
                                        rprog.progress(frac)
                                        try:
                                            rprog_ov.progress(frac)
                                        except Exception:
                                            pass
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
    
    # 進捗タブの内容
    with progress_tab:
        st.subheader("🚀 リアルタイム進捗ダッシュボード")
        
        # 進捗概要
        overall_status = progress_system.get_overall_status()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("現在のフェーズ", overall_status["current_phase"])
        with col2:
            progress_percent = int(overall_status["progress"]["current_state"]["progress"] * 100)
            st.metric("進捗率", f"{progress_percent}%")
        with col3:
            st.metric("ステータス", overall_status["status"])
        
        # チェックポイント管理
        st.subheader("💾 チェックポイント管理")
        
        col_cp1, col_cp2 = st.columns(2)
        with col_cp1:
            if st.button("手動チェックポイント作成"):
                checkpoint_id = progress_system.save_checkpoint("手動作成")
                if checkpoint_id:
                    st.success(f"チェックポイント作成完了: {checkpoint_id}")
                else:
                    st.error("チェックポイント作成に失敗しました")
        
        with col_cp2:
            checkpoints = progress_system.list_checkpoints()
            if checkpoints:
                checkpoint_names = [f"{cp.checkpoint_id} ({cp.description})" for cp in checkpoints]
                selected_cp = st.selectbox("復元するチェックポイント", ["選択してください"] + checkpoint_names)
                
                if selected_cp != "選択してください" and selected_cp is not None:
                    checkpoint_id = selected_cp.split(" (")[0]
                    if st.button("復元実行"):
                        if progress_system.restore_checkpoint(checkpoint_id):
                            st.success("チェックポイント復元完了")
                            st.experimental_rerun()
                        else:
                            st.error("チェックポイント復元に失敗しました")
        
        # 詳細進捗情報
        st.subheader("📊 詳細進捗情報")
        progress_details = overall_status["progress"]
        st.json(progress_details)
        
        # システム設定
        st.subheader("⚙️ システム設定")
        with st.expander("進捗システム設定"):
            st.write("**通知設定**")
            st.code(f"""
NOTIFICATION_EMAIL_ENABLED={os.getenv('NOTIFICATION_EMAIL_ENABLED', 'false')}
NOTIFICATION_WEBHOOK_ENABLED={os.getenv('NOTIFICATION_WEBHOOK_ENABLED', 'false')}
NOTIFICATION_DESKTOP_ENABLED={os.getenv('NOTIFICATION_DESKTOP_ENABLED', 'false')}
            """)
            
            st.write("**チェックポイント設定**")
            if "checkpoint_info" in overall_status:
                cp_info = overall_status["checkpoint_info"]
                st.write(f"保存済みチェックポイント: {cp_info['checkpoint_count']}個")
                st.write(f"総サイズ: {cp_info['total_size_mb']:.1f}MB")
                st.write(f"自動保存: {'有効' if cp_info['auto_save_enabled'] else '無効'}")
                st.write(f"保存間隔: {cp_info['auto_save_interval']/60:.1f}分")
        
        # ログ表示
        st.subheader("📝 進捗ログ")
        log_file = Path("data/artifacts/progress.log")
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_lines = f.readlines()[-20:]  # 最新20行
                
                log_text = "".join(log_lines)
                st.code(log_text, language="text")
            except Exception as e:
                st.error(f"ログ読み込みエラー: {e}")
        else:
            st.info("ログファイルがまだ作成されていません")
    
if __name__ == "__main__" and os.getenv("CLI","0") == "1":
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/sample.csv"
    log = RunLog(meta={"input_csv": csv})
    
    # CLI用の統合進捗システム初期化（Streamlit無効、コンソール出力有効）
    from src.utils.integrated_progress import create_integrated_progress_system
    cli_progress = create_integrated_progress_system(
        enable_dashboard=False,  # CLI ではStreamlitダッシュボード無効
        enable_notifications=True,
        enable_checkpoints=True,
        enable_console_output=True
    )
    
    try:
        llm = LLMClient()
    except Exception:
        llm = None  # type: ignore
    # CLI では既定で LLM を使わない（ネット依存/遅延回避）。有効化するには USE_LLM=1 を設定。
    use_llm_cli = os.getenv("USE_LLM", "0") == "1"
    has_llm = bool(llm) and llm.is_ready(deep=False) and use_llm_cli

    # 1) 初期プラン実行（LLMなしならfew-shotにフォールバック）
    cli_progress.start_phase("EDA初期分析", message="CLI モードでEDA分析を開始")
    
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
    
    # CLI 進捗表示（簡易 + 統合進捗システム）
    phase_count = 1 + (2 if has_llm else 0)
    phase_idx = 0
    total = max(1, len(steps))
    
    def _on_cli(idx, step, obs):
        local = (idx + 1) / max(1, total)
        overall = (phase_idx + local) / phase_count
        preview = step.note or (step.code or "").strip().splitlines()[0:1]
        if isinstance(preview, list):
            preview = preview[0] if preview else ""
        
        # 従来のJSON出力
        print(json.dumps({
            "event": "eda_step",
            "phase": f"{phase_idx+1}/{phase_count}",
            "step": f"{idx+1}/{total}",
            "action": step.action,
            "ok": obs.success,
            "overall": round(overall, 3),
            "preview": (preview or "")[:120]
        }, ensure_ascii=False))
        
        # 統合進捗システム更新
        step_name = f"{step.action}: {(preview or '')[:50]}"
        cli_progress.update_step(step_name, f"ステップ {idx+1}/{total}")
    
    log = execute_plan(steps, csv, log, on_step=_on_cli,
                      progress_manager=cli_progress.progress_manager,
                      phase_name="EDA初期分析")

    # 2) 反省→再計画を最大2回
    for i in range(2):
        if has_llm:
            assert llm is not None
            cli_progress.start_phase(f"EDA改善 ラウンド{i+1}", message=f"分析結果を改善します（{i+1}/2）")
            steps = next_eda_plan(llm, log)
            total = max(1, len(steps))
            phase_idx += 1
            log = execute_plan(steps, csv, log, on_step=_on_cli,
                              progress_manager=cli_progress.progress_manager,
                              phase_name=f"EDA改善 ラウンド{i+1}")
            if any(s.action=="stop" for s in steps):
                break
        else:
            break
    
    cli_progress.complete_phase("EDA分析完了")

    # 3) レポート化
    path = render_eda(csv)
    print(json.dumps({"report_md": path, "turns": len(log.turns)}, ensure_ascii=False))
    # 4) モデリング（CLI）
    cli_progress.start_phase("モデリング", message="機械学習モデルの訓練を開始")
    
    import pandas as pd  # type: ignore[reportMissingImports]
    df = pd.read_csv(csv)
    task, ycol = infer_task_and_target(df, hint=os.getenv("TARGET"))
    
    def _model_progress(stage: str, frac: float):
        cli_progress.update_step(stage, f"進捗: {int(frac*100)}%")
    
    result = train_candidates(df, task, ycol, progress=_model_progress)
    print("Modeling:", json.dumps(result, ensure_ascii=False))
    
    if has_llm:
        try:
            from src.agents.model_agent import reflect_and_improve
            from src.runners.code_runner import run_python
            
            for i in range(2):
                cli_progress.start_phase(f"モデル改善 ラウンド{i+1}", message=f"モデルの性能改善を実行（{i+1}/2）")
                steps2 = reflect_and_improve(llm, json.dumps(result, ensure_ascii=False))
                if not steps2:
                    break
                
                for j, s in enumerate(steps2):
                    if s.get("action")=="python":
                        cli_progress.update_step(f"改善コード実行 {j+1}/{len(steps2)}")
                        run_python(s.get("code",""), input_csv=csv)
                
                df = pd.read_csv(csv)
                result = train_candidates(df, task, ycol, progress=_model_progress)
        except Exception as e:
            cli_progress.error_phase(f"モデル改善でエラー: {str(e)}")
    
    cli_progress.complete_phase("全処理完了")
    open("data/artifacts/model_scores.json","w",encoding="utf-8").write(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 最終的な進捗サマリーを出力
    print("\n" + "="*60)
    print("🎉 処理完了サマリー")
    print("="*60)
    from src.utils.dashboard import print_dashboard_summary
    print_dashboard_summary(cli_progress.progress_manager)

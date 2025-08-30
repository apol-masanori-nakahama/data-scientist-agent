from __future__ import annotations
from typing import Any, Callable, Optional

INSIGHT_SYSTEM_JA = (
    "あなたはシニアデータサイエンティストです。以下の EDA・モデリング成果から、\n"
    "各セクション300文字程度（最低300文字、箇条書き禁止）の短い段落で日本語解説を書いてください。焦点: \n"
    "1) 重要な特徴と関係性、2) データ品質/選択バイアスのリスク、\n"
    "3) 有用な特徴量エンジニアリングと追加データ収集案、\n"
    "4) モデルとしきい値の提案、5) 次に取る具体アクション、\n"
    "6) データをよく説明できるモデル種別と理由（線形/木系/GBDT/NN 等のどれが妥当か、非線形・相互作用・カテゴリ比率・外れ値耐性・解釈性/精度のトレードオフ、推奨可視化や重要度指標も明記）。\n"
    "各段落は具体的な数値・条件・優先度を含め、冗長表現は避ける。\n"
)


def generate_insights(
    llm: Any,
    context_text: str,
    rounds: int = 5,
    progress: Optional[Callable[[int, int], None]] = None,
    on_draft: Optional[Callable[[str, int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    on_stream: Optional[Callable[[str, int, int], None]] = None,
) -> str:
    """Generate iterative insights with self-critique over multiple rounds.

    Parameters
    ----------
    llm: Any
        LLM client exposing chat(messages: list[dict]) -> str
    context_text: str
        Concatenated plain-text context (scores, heads of tables, etc.)
    rounds: int
        Number of refinement rounds (>=1)
    on_draft: Optional[Callable[[str, int, int], None]]
        Optional callback invoked after each round with the current draft and
        (round_index, total_rounds). Useful for streaming intermediate output to UI.
    on_stream: Optional[Callable[[str, int, int], None]]
        Optional callback invoked for token/text chunks during a round, if the
        underlying LLM client supports streaming (chat_stream). Called very
        frequently; should be lightweight.
    """
    if rounds < 1:
        rounds = 1
    draft = ""
    for i in range(rounds):
        if should_stop and should_stop():
            break
        if progress:
            # report starting round i+1
            try:
                progress(i + 1, rounds)
            except Exception:
                pass
        if i == 0:
            messages = [
                {"role": "system", "content": INSIGHT_SYSTEM_JA},
                {"role": "user", "content": context_text},
            ]
        else:
            review_instr = (
                "先程の出力を批判的にレビューし、より具体・高密度に改善する。" \
                "重複排除、優先順位付け、根拠の簡潔提示（該当指標・方向・規模）を行い、" \
                "各箇条の粒度を揃える。前ラウンドよりも踏み込んだ提案・次アクションを付す。"
            )
            messages = [
                {"role": "system", "content": INSIGHT_SYSTEM_JA + "\n" + review_instr},
                {"role": "user", "content": f"前ラウンド:\n{draft}\n\nコンテキスト:\n{context_text}"},
            ]
        try:
            # Prefer streaming if available and on_stream is provided
            if on_stream is not None and hasattr(llm, "chat_stream"):
                chunks: list[str] = []
                def _delta_cb(t: str) -> None:
                    chunks.append(t)
                    try:
                        on_stream(t, i + 1, rounds)
                    except Exception:
                        pass
                # call provider streaming
                try:
                    draft = llm.chat_stream(messages, on_delta=_delta_cb)
                except TypeError:
                    # Some implementations may not accept keyword name 'on_delta'
                    draft = llm.chat_stream(messages, _delta_cb)
            else:
                draft = llm.chat(messages)
            if on_draft:
                try:
                    on_draft(draft or "", i + 1, rounds)
                except Exception:
                    # UI update failures should not abort generation
                    pass
        except Exception:
            break
    return draft

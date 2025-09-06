# src/runners/code_runner.py
from __future__ import annotations
import subprocess, sys, tempfile, os, json, ast, textwrap, uuid

ALLOWED_IMPORTS = {
    "pandas","numpy","matplotlib","matplotlib.pyplot","sklearn",
    "sklearn.model_selection","sklearn.metrics","sklearn.linear_model",
    "sklearn.ensemble","scipy","statsmodels","math","statistics","warnings","io","json","pathlib"
}

class ImportGuard(ast.NodeVisitor):
    def __init__(self): self.bad = []
    def visit_Import(self, node):
        for n in node.names:
            if n.name not in ALLOWED_IMPORTS:
                self.bad.append(n.name)
    def visit_ImportFrom(self, node):
        mod = node.module or ""
        fullname = mod if node.level == 0 else mod
        if fullname not in ALLOWED_IMPORTS:
            self.bad.append(fullname)

def run_python(code: str, input_csv: str | None = None, workdir: str = "data/artifacts") -> dict:
    os.makedirs(workdir, exist_ok=True)
    
    # セキュリティ: input_csvのパストラバーサル攻撃を防ぐ
    if input_csv:
        # パスを正規化し、危険な文字をチェック
        normalized_path = os.path.normpath(input_csv)
        if os.path.isabs(normalized_path):
            # 絶対パスの場合、許可されたディレクトリ内かチェック
            allowed_dirs = [os.path.abspath("data"), os.path.abspath("tmp")]
            if not any(normalized_path.startswith(allowed_dir) for allowed_dir in allowed_dirs):
                return {"ok": False, "stdout": "", "stderr": f"Access denied: {input_csv} is outside allowed directories", "artifacts": []}
        else:
            # 相対パスの場合、../ を含む危険なパスを拒否
            if ".." in normalized_path or normalized_path != input_csv:
                return {"ok": False, "stdout": "", "stderr": f"Invalid path: {input_csv} contains dangerous characters", "artifacts": []}
            # 現在のディレクトリからの相対パスに変換
            normalized_path = os.path.abspath(input_csv)
        
        # ファイルの存在確認
        if not os.path.exists(normalized_path):
            return {"ok": False, "stdout": "", "stderr": f"File not found: {input_csv}", "artifacts": []}
    
    # 静的にImport検査
    tree = ast.parse(code)
    guard = ImportGuard(); guard.visit(tree)
    if guard.bad:
        return {"ok": False, "stdout": "", "stderr": f"Disallowed imports: {guard.bad}", "artifacts": []}

    # ファイル化
    script_path = os.path.join(workdir, f"cell_{uuid.uuid4().hex}.py")
    code = textwrap.dedent(code)
    with open(script_path, "w", encoding="utf-8") as f:
        # 安全な実行のための前処理
        f.write("import os\n")
        f.write("os.environ['MPLBACKEND']='Agg'\n")  # ヘッドレス描画
        f.write("try:\n")
        f.write("    import resource\n")
        f.write("    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))\n")
        f.write("    resource.setrlimit(resource.RLIMIT_AS, (1600000000, 1600000000))\n")
        f.write("except Exception:\n")
        f.write("    pass\n")
        if input_csv:
            # 正規化されたパスを使用
            f.write(f"INPUT_CSV=r'''{normalized_path}'''\n")
        f.write(code)

    # 実行
    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd(), text=True
    )
    try:
        out, err = proc.communicate(timeout=35)
    except subprocess.TimeoutExpired:
        proc.kill()
        return {"ok": False, "stdout":"", "stderr": "TimeoutExpired", "artifacts": []}

    # 成果物列挙
    artifacts = []
    for fn in os.listdir(workdir):
        if fn.startswith("cell_") and (fn.endswith(".png") or fn.endswith(".csv") or fn.endswith(".html") or fn.endswith(".json")):
            artifacts.append(os.path.join(workdir, fn))
    return {"ok": proc.returncode == 0, "stdout": out, "stderr": err, "artifacts": artifacts}
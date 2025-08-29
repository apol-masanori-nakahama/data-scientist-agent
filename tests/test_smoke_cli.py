import os, subprocess, sys, json, tempfile, shutil

ROOT = os.path.dirname(os.path.dirname(__file__))

def run(cmd: list[str], env=None, timeout=90):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT, env=env or os.environ.copy())
    out, err = proc.communicate(timeout=timeout)
    return proc.returncode, out, err

def test_cli_smoke(tmp_path):
    # Prepare sample CSV
    import numpy as np, pandas as pd
    csv = tmp_path/"sample.csv"
    n = 64
    df = pd.DataFrame({
        "feature_num": np.random.randn(n),
        "feature_cat": np.random.choice(["A","B"], size=n),
        "target": np.random.choice([0,1], size=n)
    })
    df.to_csv(csv, index=False)

    # Run CLI with FAST_TEST
    env = os.environ.copy()
    env.setdefault("CLI", "1")
    env.setdefault("FAST_TEST", "1")
    code, out, err = run([sys.executable, "app.py", str(csv)], env)
    assert code == 0, f"CLI failed: {err}\n{out}"

    # Check artifacts exist
    art = os.path.join(ROOT, "data", "artifacts")
    assert os.path.exists(os.path.join(art, "eda_report.md"))
    assert os.path.exists(os.path.join(art, "model_scores.json"))


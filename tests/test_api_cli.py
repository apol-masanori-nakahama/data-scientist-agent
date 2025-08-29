import os, sys, subprocess, json, tempfile, textwrap

ROOT = os.path.dirname(os.path.dirname(__file__))


def run(cmd, env=None, timeout=90):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=ROOT, text=True, env=env or os.environ.copy())
    out, err = p.communicate(timeout=timeout)
    return p.returncode, out, err


def test_cli_entry(tmp_path):
    # make small csv
    csv = tmp_path/"sample.csv"
    csv.write_text("target,feature_cat,feature_num\n0,A,1\n1,B,2\n", encoding='utf-8')
    code, out, err = run([sys.executable, "scripts/analyze.py", "--csv", str(csv), "--fast", "--no-mp"]) 
    assert code == 0 or "report_md" in out



from __future__ import annotations
import argparse, json
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root
from src.services.analyzer import run_analysis, AnalysisOptions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to input CSV")
    ap.add_argument("--llm", action="store_true", help="Use LLM for planning and insights")
    ap.add_argument("--reflect", type=int, default=2, help="Reflect rounds")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-mp", action="store_true")
    ap.add_argument("--s3", action="store_true", help="Upload artifacts to S3 if S3_BUCKET is set")
    args = ap.parse_args()
    res = run_analysis(
        args.csv,
        use_llm=args.llm,
        options=AnalysisOptions(
            reflect_rounds=args.reflect,
            fast_test=args.fast,
            no_multiproc=args.no_mp,
            upload_s3=args.s3,
        ),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



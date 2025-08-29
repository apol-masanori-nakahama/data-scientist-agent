import os, sys, json

def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/sample.csv"
    os.environ.setdefault("CLI", "1")
    from app import RunLog, LLMClient, initial_eda_plan, execute_plan, next_eda_plan, render_eda
    log = RunLog(meta={"input_csv": csv})
    llm = LLMClient()
    steps = initial_eda_plan(llm)
    log = execute_plan(steps, csv, log)
    for _ in range(2):
        steps = next_eda_plan(llm, log)
        log = execute_plan(steps, csv, log)
    path = render_eda(csv)
    print(json.dumps({"report_md": path, "turns": len(log.turns)}, ensure_ascii=False))

if __name__ == "__main__":
    main()


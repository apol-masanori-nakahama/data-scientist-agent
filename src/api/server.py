from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil, tempfile, time, asyncio, os
from src.services.analyzer import run_analysis, AnalysisOptions
from prometheus_client import Counter, Histogram, generate_latest

API_KEY = os.getenv("API_KEY")  # optional: set via env at startup

REQUESTS = Counter("api_requests_total", "Total API requests", ["path", "method", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["path"]) 
RATE_LIMIT_HITS = Counter("api_rate_limit_hits_total", "Rate limit rejections", ["key"]) 

app = FastAPI(title="Data Scientist Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
def health():
    return {"ok": True}


def _enforce_api_key(x_api_key: str | None) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


# ---- Simple in-memory rate limiter (fixed window) ----
_WINDOW_SECONDS = 60
_MAX_REQUESTS = 60
_RL_BUCKET: dict[str, list[float]] = {}


def _rate_limit_check(key: str) -> None:
    now = time.time()
    bucket = _RL_BUCKET.setdefault(key, [])
    # drop old
    cutoff = now - _WINDOW_SECONDS
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)
    if len(bucket) >= _MAX_REQUESTS:
        RATE_LIMIT_HITS.labels(key=key).inc()
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    bucket.append(now)


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    use_llm: bool = Form(False),
    reflect_rounds: int = Form(2),
    upload_s3: bool = Form(False),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _enforce_api_key(x_api_key)
    # simplest key selection: api key or anonymous
    rl_key = x_api_key or "anon"
    _rate_limit_check(rl_key)
    start = time.time()
    with tempfile.TemporaryDirectory() as d:
        csv_path = Path(d) / "input.csv"
        with csv_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        res = run_analysis(
            str(csv_path),
            use_llm=use_llm,
            options=AnalysisOptions(reflect_rounds=reflect_rounds, upload_s3=upload_s3),
        )
        LATENCY.labels(path="/analyze").observe(time.time() - start)
        REQUESTS.labels(path="/analyze", method="POST", status="200").inc()
        return JSONResponse(res)


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type="text/plain; version=0.0.4; charset=utf-8")


# ---- Async jobs (in-memory) ----
_JOBS: dict[str, dict] = {}


@app.post("/jobs/analyze")
async def create_job(
    request: Request,
    file: UploadFile = File(...),
    use_llm: bool = Form(False),
    reflect_rounds: int = Form(2),
    upload_s3: bool = Form(False),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _enforce_api_key(x_api_key)
    client_host = request.client.host if request.client else None
    rl_key = x_api_key or client_host or "anon"
    _rate_limit_check(rl_key)

    job_id = str(int(time.time() * 1000))
    _JOBS[job_id] = {"status": "queued", "result": None, "created_at": time.time()}

    async def _worker(payload: bytes):
        _JOBS[job_id]["status"] = "running"
        try:
            with tempfile.TemporaryDirectory() as d:
                csv_path = Path(d) / "input.csv"
                with csv_path.open("wb") as f:
                    f.write(payload)
                res = run_analysis(
                    str(csv_path),
                    use_llm=use_llm,
                    options=AnalysisOptions(reflect_rounds=reflect_rounds, upload_s3=upload_s3),
                )
                _JOBS[job_id]["result"] = res
                _JOBS[job_id]["status"] = "done"
        except Exception as e:
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["result"] = {"error": str(e)}

    payload = await file.read()
    asyncio.create_task(_worker(payload))
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job



from __future__ import annotations
from pathlib import Path
from typing import Iterable

try:
    import boto3  # type: ignore
except Exception:
    boto3 = None  # type: ignore[assignment]


def upload_directory_to_s3(local_dir: str | Path, bucket: str, prefix: str = "") -> list[str]:
    """
    Upload all files under local_dir to s3://bucket/prefix/...

    Returns list of s3 URIs uploaded.
    """
    local = Path(local_dir)
    if not local.exists():
        return []
    if boto3 is None:
        # boto3 が未インストールの場合は空アップロードとして扱う（呼び出し側でエラーハンドル推奨）
        return []
    s3 = boto3.client("s3")
    uploaded: list[str] = []
    for p in local.rglob("*"):
        if p.is_file():
            rel = p.relative_to(local)
            key = f"{prefix.rstrip('/')}/{rel.as_posix()}" if prefix else rel.as_posix()
            s3.upload_file(str(p), bucket, key)
            uploaded.append(f"s3://{bucket}/{key}")
    return uploaded


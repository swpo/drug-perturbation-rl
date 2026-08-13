"""Shared download helper — idempotent, with progress, streaming."""

from pathlib import Path
import requests

DEFAULT_CACHE = Path("/tmp/bioreasoning_phenotype_data")


def download(url: str, dest: Path, min_size: int = 1000, chunk_mb: int = 4) -> Path:
    """Download `url` to `dest` if not already present and large enough."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_size:
        size_mb = dest.stat().st_size / 1e6
        print(f"  cached: {dest.name} ({size_mb:.1f} MB)", flush=True)
        return dest

    print(f"  downloading {url} -> {dest}", flush=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        last_pct = -10
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = int(downloaded * 100 / total)
                    if pct >= last_pct + 10:
                        print(f"    {downloaded/1e6:.0f}/{total/1e6:.0f} MB ({pct}%)", flush=True)
                        last_pct = pct
        tmp.rename(dest)
    print(f"  done: {dest} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)
    return dest

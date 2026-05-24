"""Upload all best.pt weight files to GCS, mirroring the local runs/ structure."""
import ssl, sys, pathlib

ssl._create_default_https_context = ssl._create_unverified_context

from google.cloud import storage

bucket_name = sys.argv[1]
repo_root = pathlib.Path(sys.argv[2])
runs_root = repo_root / "ml-engine" / "runs"

client = storage.Client()
bucket = client.bucket(bucket_name)

for pt in sorted(runs_root.rglob("best.pt")):
    blob_name = "runs/" + pt.relative_to(runs_root).as_posix()
    blob = bucket.blob(blob_name)
    if blob.exists():
        print(f"Skipped  {blob_name} (already exists)")
        continue
    blob.upload_from_filename(str(pt), content_type="application/octet-stream")
    print(f"Uploaded {blob_name}")

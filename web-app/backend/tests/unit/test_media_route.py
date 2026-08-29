"""
Unit tests for the /media/annotated route's path containment.
Calls the route function directly — no DB, no TestClient lifespan.
"""
import asyncio
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse

from main import serve_annotated


@pytest.mark.unit
def test_serves_file_inside_media_dir(tmp_path):
    (tmp_path / "aircraft").mkdir()
    (tmp_path / "aircraft" / "abc_annotated.mp4").write_bytes(b"video")

    with patch("main._ANNOTATED_DIR", tmp_path):
        resp = asyncio.run(serve_annotated("aircraft/abc_annotated.mp4"))

    assert isinstance(resp, FileResponse)


@pytest.mark.unit
@pytest.mark.parametrize("attack", [
    "../secret.txt",
    "../../secret.txt",
    "aircraft/../../secret.txt",
    "./../secret.txt",
])
def test_rejects_traversal_out_of_media_dir(tmp_path, attack):
    """A readable file one level up must not be reachable."""
    (tmp_path / "secret.txt").write_text("credentials")
    media = tmp_path / "media"
    (media / "aircraft").mkdir(parents=True)

    with patch("main._ANNOTATED_DIR", media):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(serve_annotated(attack))

    assert exc.value.status_code == 404


@pytest.mark.unit
def test_rejects_missing_file(tmp_path):
    with patch("main._ANNOTATED_DIR", tmp_path):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(serve_annotated("aircraft/nope.mp4"))

    assert exc.value.status_code == 404

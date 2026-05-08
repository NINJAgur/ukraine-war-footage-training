import hashlib

import pytest


def _sha256(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


@pytest.mark.unit
def test_hash_is_deterministic():
    url = "https://funker530.com/video/abc123"
    assert _sha256(url) == _sha256(url)


@pytest.mark.unit
def test_different_urls_produce_different_hashes():
    a = _sha256("https://funker530.com/video/abc")
    b = _sha256("https://funker530.com/video/def")
    assert a != b


@pytest.mark.unit
def test_hash_length_is_64():
    result = _sha256("https://example.com/some-video")
    assert len(result) == 64


@pytest.mark.unit
def test_hash_is_hex_only():
    result = _sha256("https://funker530.com/video/test")
    assert all(c in "0123456789abcdef" for c in result)


@pytest.mark.unit
def test_same_url_different_encodings_produce_same_hash():
    # Both represent the same URL string — encoding via .encode() is always UTF-8
    url = "https://funker530.com/video/test-title"
    h1 = hashlib.sha256(url.encode("utf-8")).hexdigest()
    h2 = hashlib.sha256(url.encode()).hexdigest()
    assert h1 == h2


@pytest.mark.unit
def test_empty_url_hashes_consistently():
    assert _sha256("") == _sha256("")
    assert len(_sha256("")) == 64


@pytest.mark.unit
def test_trailing_slash_changes_hash():
    assert _sha256("https://funker530.com/video") != _sha256("https://funker530.com/video/")

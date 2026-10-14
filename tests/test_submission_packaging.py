"""Submission integrity: paths, tampering, missing assets and secret exclusion."""
import json
from pathlib import Path
import zipfile

import pytest

from scripts.package_submission import build_archive, safe_path, source_allowed, verify_directory


@pytest.mark.parametrize('name', ['../secret', '/absolute', 'C:/secret', 'a/../../secret', r'a\..\secret'])
def test_reject_nonportable_paths(tmp_path, name):
    with pytest.raises(ValueError):
        safe_path(tmp_path, name)


@pytest.mark.parametrize('name', ['.env', 'src/.env', 'env/pyvenv.cfg', 'src/__pycache__/a.pyc',
                                  'dist/old.zip', 'pdfdocx/engine.py', 'AGENTS.md', 'tests/test_pdfdocx_mcp.py'])
def test_source_excludes_private_and_generated_files(name):
    assert not source_allowed(name)


def test_archive_roundtrip_and_tamper_detection(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'README.md').write_text('submission', encoding='utf-8')
    archive = tmp_path / 'delivery.zip'
    build_archive(root, {'README.md': root / 'README.md'}, archive, 'source')
    unpacked = tmp_path / 'unpacked'
    with zipfile.ZipFile(archive) as z:
        z.extractall(unpacked)
    assert verify_directory(unpacked) == []
    (unpacked / 'README.md').write_text('changed', encoding='utf-8')
    assert verify_directory(unpacked) == ['sha256_mismatch:README.md']
    (unpacked / 'README.md').unlink()
    assert verify_directory(unpacked) == ['missing:README.md']
    with pytest.raises(FileExistsError):
        build_archive(root, {}, archive, 'source')


def test_missing_input_never_creates_release(tmp_path):
    archive = tmp_path / 'delivery.zip'
    with pytest.raises(FileNotFoundError):
        build_archive(tmp_path, {'README.md': tmp_path / 'missing'}, archive, 'source')
    assert not archive.exists()


def test_manifest_path_escape_is_rejected(tmp_path):
    (tmp_path / 'SUBMISSION_MANIFEST.json').write_text(json.dumps({
        'schema': 'vsl_submission_v1', 'files': [{'path': '../outside', 'sha256': 'x'}]
    }), encoding='utf-8')
    with pytest.raises(ValueError):
        verify_directory(tmp_path)

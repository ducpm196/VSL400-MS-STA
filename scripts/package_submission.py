"""Build source/demo ZIPs with a byte-level inventory; never modify runtime assets.

Uses only the standard library. Source selection uses Git's index plus explicit
release additions; demo selection follows the active registry and input manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'SUBMISSION_MANIFEST.json'
BACKBONE = 'data/pretrained/torch/hub/checkpoints/SLOWFAST_8x8_R50.pyth'
ADDITIONS = (
    'scripts/package_submission.py', 'scripts/verify_submission.py',
    'tests/test_submission_packaging.py', 'Docs/SUBMISSION.md',
    'requirements-demo-lock.txt', 'Docs/SUBMISSION_REVIEW.md',
    'Docs/TONG_QUAN_DO_AN.md',
)
ROOT_FILES = {
    'README.md', '.env.example', '.gitignore', '.gitattributes',
    'metadata.csv', 'pytest.ini',
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    """Reject traversal, drive paths and symlinks leaving the selected root."""
    parts = PurePosixPath(relative).parts
    if not parts or '\\' in relative or ':' in relative or relative.startswith('/') or '..' in parts:
        raise ValueError(f'unsafe_relative_path:{relative}')
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'path_outside_root:{relative}')
    return path


def read_json(root: Path, relative: str) -> dict:
    return json.loads(safe_path(root, relative).read_text(encoding='utf-8-sig'))


def source_allowed(relative: str) -> bool:
    # This optional document-conversion tool is not part of the VSL application.
    if relative == 'tests/test_pdfdocx_mcp.py':
        return False
    p = PurePosixPath(relative)
    if any(part in {'.git', '__pycache__', '.pytest_cache', 'node_modules'} for part in p.parts):
        return False
    if p.name == '.env' or (p.name.startswith('.env.') and p.name != '.env.example'):
        return False
    if p.suffix.lower() in {'.pyc', '.pyo'}:
        return False
    return (relative in ROOT_FILES or (len(p.parts) == 1 and p.name.startswith('requirements')
            and p.suffix == '.txt') or p.parts[0] in {'src', 'scripts', 'tests', 'Docs', 'data', '.github'})


def source_files(root: Path) -> dict[str, Path]:
    output = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root)
    names = set(output.decode('utf-8').split('\0')) | set(ADDITIONS)
    return {name: safe_path(root, name) for name in sorted(names) if name and source_allowed(name)}


def demo_files(root: Path, backbone: Path | None = None) -> dict[str, Path]:
    files: dict[str, Path] = {}

    def add(name: str, expected: str | None = None) -> None:
        path = safe_path(root, name)
        if not path.is_file():
            raise FileNotFoundError(f'demo_dependency_missing:{name}')
        if expected and sha256(path) != expected:
            raise ValueError(f'demo_dependency_sha_mismatch:{name}')
        files[name] = path

    for path in sorted((root / 'src').rglob('*.py')):
        add(path.relative_to(root).as_posix())
    for name in ('README.md', '.env.example', 'Docs/SUBMISSION.md', 'Docs/TONG_QUAN_DO_AN.md', 'metadata.csv',
                 'scripts/demo_vsl400.ps1', 'scripts/demo_health_probe.py',
                 'scripts/package_submission.py', 'scripts/verify_submission.py',
                 'requirements.txt', 'requirements-core.txt', 'requirements-demo.txt',
                 'requirements-demo-lock.txt', 'data/processed/vsl400_classifier.pth',
                 'data/processed/vsl400_labels.json', 'data/processed/ms_sta_registry_active.json'):
        add(name)
    pointer = read_json(root, 'data/processed/ms_sta_registry_active.json')
    add(pointer['registry_path'], pointer['registry_sha256'])
    registry = read_json(root, pointer['registry_path'])
    if (registry.get('active_variant') != 'b4_v6_ctr' or registry.get('e4_ready') is not False
            or registry.get('defense_ready') is not True or registry.get('demo_allowed') is not True):
        raise ValueError('active_registry_not_defense_b4_v6')
    add(registry['checkpoint'], registry['checkpoint_sha256'])
    add(registry['final_results_rel'], registry['final_results_sha256'])
    add('Docs/results/ms_sta_v6_d4/D4_ACCEPTANCE_REPORT.json', registry['d4_report_sha256'])
    add('Docs/results/ms_sta_v6_oneshot/PRETEST_FREEZE.json')
    add(registry['split_manifest_rel'])
    add(registry['demo_set_rel'])
    # Evidence pages read these small sealed reports. No checkpoint directories,
    # raw prediction dumps, historical slide decks or research run logs.
    for folder in ('ms_sta_v6_defense_go', 'ms_sta_v6_oneshot', 'e4_system_kd_v1'):
        for path in sorted((root / 'Docs/results' / folder).glob('*.json')):
            add(path.relative_to(root).as_posix())
    add('Docs/results/project_finalization/SCIENTIFIC_CLAIM_LOCK.json')
    input_manifest = read_json(root, 'Docs/results/ms_sta_v6_defense_go/E4_INPUT_DEMO_MANIFEST.json')
    add(input_manifest['metadata_rel'], input_manifest['metadata_sha256'])
    add(input_manifest['split_manifest_rel'], input_manifest['split_manifest_sha256'])
    for sample in input_manifest['samples']:
        for key, hash_key in (('video_path_rel', 'video_sha256'), ('rgb_cache_rel', 'rgb_cache_sha256'),
                              ('skeleton_cache_rel', 'skeleton_cache_sha256')):
            add(sample[key], sample[hash_key])
    # Use the same raw-video resolver as the API; preserve original relative paths.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.data_prep.raw_video_resolve import RawVideoResolver
    resolver = RawVideoResolver(root)
    ids = set(read_json(root, registry['demo_set_rel'])['video_ids'])
    found = set()
    with (root / 'metadata.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            sid = row['video_id']
            if sid not in ids or row.get('view', 'front') != 'front' or row['dataset'] != 'VSL400':
                continue
            video = resolver.resolve(row)
            if video is None:
                raise FileNotFoundError(f'demo_video_missing:{sid}')
            add(video.relative_to(root).as_posix())
            roots = registry['feature_roots']
            add(f"{roots['rgb_root_rel']}/rgb_token_v3/{sid}.npz")
            add(f"{roots['skeleton_root_rel']}/skeleton_v3/{sid}.npz")
            found.add(sid)
    if found != ids:
        raise ValueError(f'demo_metadata_missing:{sorted(ids - found)}')
    local = safe_path(root, BACKBONE)
    if backbone is None:
        torch_home = Path(os.environ.get('TORCH_HOME', Path.home() / '.cache/torch'))
        backbone = local if local.is_file() else torch_home / 'hub/checkpoints/SLOWFAST_8x8_R50.pyth'
    if not backbone.is_file():
        raise FileNotFoundError(f'pretrained_slowfast_missing:{backbone}')
    files[BACKBONE] = backbone.resolve()
    return files


def build_archive(root: Path, files: dict[str, Path], output: Path, profile: str) -> dict:
    if output.exists():
        raise FileExistsError(f'refuse_overwrite:{output}')
    inventory = []
    for name, path in sorted(files.items()):
        safe_path(root, name)
        if not path.is_file():
            raise FileNotFoundError(f'package_input_missing:{name}')
        inventory.append({'path': name, 'size': path.stat().st_size, 'sha256': sha256(path)})
    manifest = {'schema': 'vsl_submission_v1', 'profile': profile, 'files': inventory,
                'scope': 'source and evidence; external artifacts required' if profile == 'source'
                else 'offline demo: 2 curated test clips + 6 validation visualization clips',
                'python': '3.11', 'secrets_included': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of a previous release.
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for row in inventory:
            archive.write(files[row['path']], row['path'])
        archive.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    # Verify archived bytes, not just source bytes or ZIP CRC.
    with zipfile.ZipFile(output) as archive:
        for row in inventory:
            with archive.open(row['path']) as stream:
                h = hashlib.file_digest(stream, 'sha256').hexdigest()
            if h != row['sha256']:
                raise ValueError(f'archive_sha_mismatch:{row["path"]}')
    return manifest


def verify_directory(root: Path) -> list[str]:
    manifest = read_json(root, MANIFEST)
    if manifest.get('schema') != 'vsl_submission_v1':
        raise ValueError('unsupported_submission_manifest')
    errors = []
    for row in manifest['files']:
        path = safe_path(root, row['path'])
        if not path.is_file():
            errors.append(f'missing:{row["path"]}')
        elif sha256(path) != row['sha256']:
            errors.append(f'sha256_mismatch:{row["path"]}')
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('source', 'demo'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, help='Local SLOWFAST_8x8_R50.pyth (never downloaded)')
    args = parser.parse_args()
    files = source_files(ROOT) if args.profile == 'source' else demo_files(ROOT, args.backbone)
    manifest = build_archive(ROOT, files, args.output, args.profile)
    print(json.dumps({'archive': str(args.output), 'files': len(files),
                      'sha256': sha256(args.output), 'bytes': args.output.stat().st_size,
                      'uncompressed_bytes': sum(r['size'] for r in manifest['files'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

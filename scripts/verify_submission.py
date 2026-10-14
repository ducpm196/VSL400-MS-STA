"""Check an extracted submission; optionally run real offline API/UI inference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.package_submission import BACKBONE, MANIFEST, demo_files, verify_directory


def smoke_test(ui: bool = False) -> dict:
    """Exercise actual startup and inference; prohibit pretrained downloads."""
    local_backbone = ROOT / BACKBONE
    if local_backbone.is_file():
        os.environ['TORCH_HOME'] = str(ROOT / 'data/pretrained/torch')
    else:
        user_cache = Path.home() / '.cache/torch'
        if (user_cache / 'hub/checkpoints/SLOWFAST_8x8_R50.pyth').is_file():
            os.environ['TORCH_HOME'] = str(user_cache)
        else:
            os.environ['TORCH_HOME'] = str(ROOT / 'data/pretrained/torch')
    os.environ.pop('MS_STA_REGISTRY_PATH', None)
    import torch
    from unittest.mock import patch
    from fastapi.testclient import TestClient

    # Keep CPU smoke fast and stable on high-core machines; model math unchanged.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    from src.deployment.backend_api import app
    from src.deployment.input_evidence import load_e4_input_demo_manifest, validate_e4_input_demo_manifest
    errors = validate_e4_input_demo_manifest(load_e4_input_demo_manifest(root=ROOT), root=ROOT)
    if errors:
        raise RuntimeError(f'input_manifest_invalid:{errors}')
    with patch('torch.hub.download_url_to_file', side_effect=RuntimeError('offline_download_forbidden')):
        with TestClient(app) as client:
            health_response = client.get('/health')
            health_response.raise_for_status()
            health = health_response.json()
            expected = {'ready': True, 'ms_sta_ready': True, 'engine': 'ms_sta_v6_b4_v6_ctr',
                        'e4_ready': False, 'ms_sta_input_mode': 'cache_backed_offline',
                        'campaign_status': 'DEFENSE_GO',
                        'ms_sta_checkpoint_sha256': '0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1'}
            for key, value in expected.items():
                if health.get(key) != value:
                    raise RuntimeError(f'health_contract:{key}:{health.get(key)!r}')
            demo = json.loads((ROOT / 'data/processed/demo_set_ms_sta.json').read_text(encoding='utf-8'))
            predictions = []
            for sid in demo['video_ids']:
                response = client.post('/predict_clip', json={
                    'video_id': sid, 'mode': 'isolated', 'use_ms_sta': True,
                    'top_k': 5, 'use_llm': False, 'return_attention': True,
                })
                response.raise_for_status()
                body = response.json()
                if (body.get('engine') != expected['engine'] or len(body.get('predictions', [])) != 5
                        or body.get('requires_human_confirmation') is not True
                        or body.get('final_vietnamese_text') is not None):
                    raise RuntimeError(f'prediction_contract:{sid}')
                predictions.append(body)
            invalid = client.post('/predict_clip', json={
                'video_id': '__missing__', 'use_ms_sta': True, 'use_llm': False})
            if invalid.status_code != 404:
                raise RuntimeError(f'unknown_clip_not_refused:{invalid.status_code}')
            pages = []
            if ui:
                from streamlit.testing.v1 import AppTest
                def get(url, **kwargs):
                    return client.get('/' + url.split('/', 3)[-1])
                def post(url, **kwargs):
                    return client.post('/' + url.split('/', 3)[-1], json=kwargs.get('json'))
                paths = [ROOT / 'src/deployment/app.py'] + sorted((ROOT / 'src/deployment/pages').glob('*.py'))
                with patch('requests.get', side_effect=get), patch('requests.post', side_effect=post):
                    for path in paths:
                        view = AppTest.from_file(str(path), default_timeout=120).run()
                        # The Limitations page intentionally uses st.error as a
                        # prominent statement that continuous translation is absent.
                        if view.exception or (view.error and path.name != '4_Limitations.py'):
                            raise RuntimeError(f'ui_failure:{path.name}:{view.exception}:{view.error}')
                        pages.append(path.name)
            return {'health': {k: health.get(k) for k in expected}, 'device': health.get('device'),
                    'predictions': predictions, 'unknown_clip_status': invalid.status_code,
                    'input_manifest_errors': errors, 'ui_pages': pages,
                    'network_downloads': 'forbidden', 'transport': 'in-process FastAPI TestClient'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='Real startup + both curated predictions')
    parser.add_argument('--ui', action='store_true', help='Also execute all Streamlit pages using AppTest')
    parser.add_argument('--output', type=Path, help='Write machine-readable verification report')
    args = parser.parse_args()
    report = {'integrity': 'NOT_RUN', 'smoke': 'NOT_RUN'}
    if (ROOT / MANIFEST).is_file():
        errors = verify_directory(ROOT)
        if errors:
            raise RuntimeError('\n'.join(errors))
        report['integrity'] = 'PASS'
    else:
        report['integrity'] = 'development checkout; no archive manifest'
    # Static dependency and sealed-SHA checks, also usable by the launcher.
    if (ROOT / BACKBONE).is_file():
        report['demo_dependency_count'] = len(demo_files(ROOT))
    if args.smoke or args.ui:
        report['smoke'] = smoke_test(ui=args.ui)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

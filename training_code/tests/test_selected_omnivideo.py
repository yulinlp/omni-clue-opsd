import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('selected', Path(__file__).resolve().parents[1] / 'scripts/prepare_omnivideo_selected.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    rows = [dict(question_id=sid, video_id='v', question='Q?', options=['one','two','three','four'], answer='A', duration=10, task='comparison', analysis={'designated_segments':'00:01 - 00:03'}) for sid in ['a','b']]
    annotation = tmp_path/'annotations.jsonl'
    annotation.write_text('\n'.join(json.dumps(row) for row in rows))
    ids = tmp_path/'ids.txt'
    ids.write_text('b\na\n')
    (tmp_path/'v.mp4').write_bytes(b'fixture')
    return annotation, ids


def test_preserves_exact_order_and_direct_media_directory(tmp_path):
    annotation, ids = fixture(tmp_path)
    rows = module.prepare(annotation, ids, tmp_path, 2)
    assert [r['sample_id'] for r in rows] == ['b','a']
    assert rows[0]['video_path'] == str(tmp_path/'v.mp4')
    assert rows[0]['evidence_spans'] == [[1.0,3.0]]


@pytest.mark.parametrize('value,match', [('a\na\n','unique'),('a\nc\n','absent')])
def test_rejects_bad_id_list(tmp_path, value, match):
    annotation, ids = fixture(tmp_path)
    ids.write_text(value)
    with pytest.raises(ValueError, match=match):
        module.prepare(annotation, ids, tmp_path, 2)


def test_rejects_missing_media(tmp_path):
    annotation, ids = fixture(tmp_path)
    (tmp_path/'v.mp4').unlink()
    with pytest.raises(ValueError, match='missing/empty'):
        module.prepare(annotation, ids, tmp_path, 2)


def test_rejects_missing_evidence(tmp_path):
    annotation, ids = fixture(tmp_path)
    annotation.write_text(annotation.read_text().replace('00:01 - 00:03',''))
    with pytest.raises(ValueError, match='missing duration/evidence'):
        module.prepare(annotation, ids, tmp_path, 2)

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from omni_opsd.worldsense import clue_only_agentic as c


def item():
    return dict(sample_id='test::task0', question='Which animal?', choices=['Cat', 'Dog', 'Bird'], answer='B',
                media={'duration': 30}, caption_candidate='UNVERIFIED_CAPTION', observation='SECRET_OBSERVATION')


def media(item, spans, cache, full=False):
    return Path('/fake.mp4'), dict(spans=spans, file_sha256=str(spans), bytes=100,
            mapping=[{'montage': [0, sum(e-s for s,e in spans)], 'original': spans[0]}])


def xml(answer='B', analysis='A dog is visible.'):
    return f'<analysis>{analysis}</analysis><answer>{answer}</answer>'


def failed():
    return dict(answer='A', analysis='ORIGINAL_WRONG_ANALYSIS', raw_response=xml('A','ORIGINAL_WRONG_ANALYSIS'),
                clue_intervals=[[0., 2.]], passed=False)


class FakeAPI:
    def __init__(self, folder, actions, answers=()):
        self.folder=folder; self.actions=iter(actions); self.answers=iter(answers); self.calls=[]
    def call(self, stage, messages, max_tokens, parser):
        self.calls.append((stage, copy.deepcopy(messages), max_tokens))
        raw=xml(next(self.answers)) if stage.startswith('blind_') else json.dumps(next(self.actions))
        return parser(raw)


def inspect(start=2,end=8):
    return dict(action='inspect', start=start,end=end, focus='Locate the animal',reason='Need more context')
def submit(start=2,end=8):
    return dict(action='submit',clue_intervals=[[start,end]], evidence_summary='A dog is visible.',reason='Sufficient evidence')


def test_exact_xml_and_token_limit():
    assert c.parse_xml(xml(), 'ABC', lambda _:120)['answer']=='B'
    with pytest.raises(ValueError):c.parse_xml(xml(), 'ABC',lambda _:121)
    for raw in [xml('D'), xml()+'\nB', '<analysis></analysis><answer>B</answer>',
                '<analysis><analysis>x</analysis><answer>A</answer></analysis><answer>B</answer>',
                '<answer>B</answer><analysis>x</analysis>', '<analysis>x</analysis><answer>B or C</answer>']:
        with pytest.raises(ValueError):c.parse_xml(raw,'ABC')


def test_blind_has_no_privileged_data(monkeypatch,tmp_path):
    monkeypatch.setattr(c.q,'render_media',media)
    api=FakeAPI(tmp_path,[],['B'])
    result=c.blind_verify(item(), [[2,8]],api,tmp_path)
    prompt=api.calls[0][1][0]['content']; data=json.loads(prompt.split('DATA:\n')[1])
    assert set(data)=={'question','options','clip_duration_seconds','clip_parts'}
    assert 'UNVERIFIED_CAPTION' not in prompt and 'SECRET_OBSERVATION' not in prompt
    assert result['passed'] and not result['observation_checked']
    assert api.calls[0][2]==120


def test_stateful_repair_receives_exact_failure_and_rechecks_independently(monkeypatch,tmp_path):
    monkeypatch.setattr(c.q,'render_media',media)
    api=FakeAPI(tmp_path/'requests',[inspect(),submit(),inspect(8,14),submit(8,14)],['C','B'])
    result=c.repair(item(),failed(),api,tmp_path)
    assert result['status']=='repaired_clue_pass'
    assert [x['answer'] for x in result['checks']]==['C','B']
    first=api.calls[0][1][1]['content']
    assert 'ORIGINAL_WRONG_ANALYSIS' in first and '"incorrect_answer": "A"' in first
    calls=[call for call in api.calls if call[0].startswith('agent_')]
    after_failure=json.dumps(calls[2][1])
    assert '"incorrect_answer": "C"' in after_failure.replace('\\"','"')
    assert len(calls[2][1])>len(calls[0][1])
    assert any(m.get('media') for m in calls[1][1])
    for stage,messages,_ in api.calls:
        assert 'SECRET_OBSERVATION' not in json.dumps(messages)
        if stage.startswith('blind_'):
            assert len(messages)==1 and 'ORIGINAL_WRONG_ANALYSIS' not in str(messages)
            assert 'UNVERIFIED_CAPTION' not in str(messages)


def test_inspect_required_and_same_failed_clue_not_requeried(monkeypatch,tmp_path):
    monkeypatch.setattr(c.q,'render_media',media)
    actions=[submit(),inspect(),inspect(),submit(0,2),submit()]
    api=FakeAPI(tmp_path/'requests',actions,['B'])
    result=c.repair(item(),failed(),api,tmp_path)
    assert result['status']=='repaired_clue_pass'
    assert [t.get('rejected') for t in result['trace']]==['inspect_required',None,'duplicate_inspect','identical_clue_already_failed',None]
    assert sum(s.startswith('blind_') for s,_,_ in api.calls)==1


def test_failure_budget_terminates_without_observation(monkeypatch,tmp_path):
    monkeypatch.setattr(c.q,'render_media',media)
    api=FakeAPI(tmp_path/'requests',[inspect(),submit()],['C'])
    result=c.repair(item(),failed(),api,tmp_path,max_checks=1)
    assert result['status']=='needs_review' and result['reason']=='verification_budget_exhausted'
    assert len(api.calls)==3


@pytest.mark.parametrize('start,end',[(0,121),(-1,3),(5,4),(0,float('nan')),(True,10)])
def test_inspect_boundaries(start,end):
    with pytest.raises(ValueError):c.parse_action(json.dumps(inspect(start,end)),300)


def test_media_pruning_keeps_text():
    history=[dict(role='user',content=f'view {i}',media={'metadata':{'bytes':4_000_000}}) for i in range(3)]
    c.trim_media(history)
    assert sum(bool(m.get('media')) for m in history)==1
    assert history[0]['content'].startswith('view 0')


class Stream(list):
    def close(self):pass


class Client:
    def __init__(self):self.calls=0
    def _create_raw(self,messages,**kwargs):
        self.calls+=1
        raw=xml() if self.calls>1 else '<analysis>truncated'
        return Stream([SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content=raw),
                                              finish_reason='stop' if self.calls>1 else 'length')])])
    def is_content_block(self,exc):return False


def test_truncated_reply_retried_and_completed_reply_cached(tmp_path):
    client=Client(); api=c.RecordedAPI(client,tmp_path)
    messages=[dict(role='user',content='Question/options without gold')]
    for _ in range(2):
        out=api.call('blind_test',messages,120,lambda raw:c.parse_xml(raw,'ABC'))
        assert out['answer']=='B'
    assert client.calls==2
    record=json.loads((tmp_path/'blind_test.json').read_text())
    assert record['status']=='complete'
    assert record['attempts'][0]['finish_reason']=='length'
    assert record['attempts'][-1]['format_retry_note_added']
    with pytest.raises(ValueError):api.call('blind_test',[dict(role='user',content='changed')],120,str)

import importlib.util,json
from pathlib import Path
from unittest.mock import Mock
import pytest

spec=importlib.util.spec_from_file_location('codex_reverify',Path(__file__).resolve().parents[1]/'scripts/run_worldsense_codex_reverify.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_run_is_blocked_before_parent_stops(monkeypatch):
 monkeypatch.setattr(m,'parent_finished',lambda:False)
 credentials=Mock();monkeypatch.setattr(m,'configure_credentials',credentials)
 with pytest.raises(RuntimeError,match='has not ended'):m.run(Mock())
 credentials.assert_not_called()


def test_finished_marker_and_dead_pid_are_both_required(monkeypatch,tmp_path):
 monkeypatch.setattr(m,'PARENT',tmp_path)
 assert not m.parent_finished()
 (tmp_path/'monitor').mkdir();(tmp_path/'monitor/final.json').write_text('{"status":"needs_operator"}')
 (tmp_path/'launch.json').write_text('{"pid":42}')
 monkeypatch.setattr(m,'alive',lambda pid:True)
 assert not m.parent_finished()
 monkeypatch.setattr(m,'alive',lambda pid:False)
 assert m.parent_finished()


def item():
 return dict(sample_id='x::task0',question='When does the event occur?',choices=['Beginning','Middle','End'],answer='C',media={'duration':90},split_provenance={'previous_training_1453':False},annotation={'observation':'SECRET_OBS'},caption_candidate='SECRET_CAPTION')


def proposal(tmp_path,spans=None):
 p=tmp_path/'annotation.json';p.write_text(json.dumps(dict(proposed_clue_intervals=spans or [[70,85]],review_status='reannotated',answer_assessment={'reason_zh':'SECRET_REVIEW'})))
 return dict(annotation=str(p),annotation_sha256=m.sha(p),same_as_a_previous_clue=False)


def test_blind_data_has_original_mapping_without_gold_or_review(monkeypatch,tmp_path):
 calls=[]
 class API:
  def __init__(self,*args):pass
  def call(self,stage,messages,limit,parser):
   calls.append(messages)
   return parser('<analysis>The event appears late in the original video.</analysis><answer>C</answer>')
 monkeypatch.setattr(m.c,'RecordedAPI',API)
 monkeypatch.setattr(m,'render',lambda *args:(tmp_path/'video.mp4',{'mapping':[{'montage':[0,15],'original':[70,85]}],'file_sha256':'fake'}))
 r=m.verify_one(item(),proposal(tmp_path),tmp_path,lambda s:10,None)
 assert r['passed'] and r['gold_answer']=='C'
 prompt=calls[0][0]['content'];data=json.loads(prompt.split('DATA:\n')[1])
 assert set(data)=={'question','options','original_video_duration_seconds','clip_to_original_time_mapping'}
 assert data['original_video_duration_seconds']==90
 assert all(s not in str(calls) for s in ['SECRET_REVIEW','SECRET_OBS','SECRET_CAPTION'])


def test_subsecond_candidate_rejected_before_model_request(monkeypatch,tmp_path):
 render=Mock();monkeypatch.setattr(m,'render',render)
 with pytest.raises(ValueError,match='Sub-second'):m.verify_one(item(),proposal(tmp_path,[[70,70.5]]),tmp_path,lambda _:0,None)
 render.assert_not_called()


def test_changed_review_rejected(monkeypatch,tmp_path):
 p=proposal(tmp_path);Path(p['annotation']).write_text('{"proposed_clue_intervals":[[60,85]]}')
 with pytest.raises(ValueError,match='changed'):m.verify_one(item(),p,tmp_path,lambda _:0,None)


def test_appearance_only_sensitive_inference_never_reaches_media_or_api(monkeypatch,tmp_path):
 p=proposal(tmp_path);path=Path(p['annotation']);a=json.loads(path.read_text())
 a['verification_exclusion']={'code':'sensitive_attribute_inference_from_appearance','reason_zh':'无可核实身份来源'}
 path.write_text(json.dumps(a));p['annotation_sha256']=m.sha(path)
 render=Mock();api=Mock();monkeypatch.setattr(m,'render',render);monkeypatch.setattr(m.c,'RecordedAPI',api)
 r=m.verify_one(item(),p,tmp_path,lambda _:0,None)
 assert r['status']=='not_evaluable' and r['passed'] is None and not r['api_request_sent']
 render.assert_not_called();api.assert_not_called()


def test_note_only_revision_reuses_same_blind_input_identity():
 sample=dict(item(),video_sha256='source-sha')
 first=dict(proposed_clue_intervals=[[70,85]],limitations_zh=['audio unverified'])
 changed=dict(first,limitations_zh=['ASR added'],answer_assessment={'dataset_gold':'C'})
 assert m.candidate_identity(sample,first)==m.candidate_identity(sample,changed)
 changed['proposed_clue_intervals']=[[60,85]]
 assert m.candidate_identity(sample,first)!=m.candidate_identity(sample,changed)


def test_technical_retry_count_is_bounded(monkeypatch,tmp_path):
 monkeypatch.setattr(m,'ROOT',tmp_path);p=proposal(tmp_path);p['question_id']='x::task0'
 assert m.still_pending(p)
 target=m.result_path(p);target.parent.mkdir(parents=True)
 target.write_text(json.dumps(dict(status='technical_error',execution_runs=2)))
 assert m.still_pending(p)
 target.write_text(json.dumps(dict(status='technical_error',execution_runs=3)))
 assert not m.still_pending(p)
 target.write_text(json.dumps(dict(status='answer_mismatch',execution_runs=1)))
 assert not m.still_pending(p)

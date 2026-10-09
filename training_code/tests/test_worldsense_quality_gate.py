import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from omni_opsd.worldsense import quality_gate as q


def item():
    return {'sample_id':'v::task0','video_id':'v','question':'What is shown?',
            'choices':['cat','dog','bird','fish'],'answer':'B','media':{'duration':10.},
            'annotation':{'clue_intervals':[[1.,3.],[6.,8.]],'observation':'A dog appears.'},
            'caption_candidate':{'caption':'OLD_CAPTION_SENTINEL'}}


def blind(answer='B'):
    return {'answer':answer,'evidence':[{'start':1.,'end':2.,'modality':'visual','fact':'A dog appears.'}],'reason':'visible'}


def audit(verdict='PASS'):
    return {'claims':[{'claim':'A dog appears.','verdict':verdict,'start':1.,'end':2.,'modality':'visual','evidence':'visible dog'}],
            'all_claims_checked':True,'observation_video':verdict,'observation_gold':verdict,
            'gold_supported':verdict,'clue_covers_answer':verdict}


class FakeAPI:
    def __init__(self,values):self.values=values;self.calls=[]
    def call(self,stage,prompt,media,metadata,validator,max_tokens=4000):
        self.calls.append((stage,prompt))
        value=copy.deepcopy(self.values[stage]);validator(value);return value


def media(*args,**kwargs):
    return Path('/tmp/fake.mp4'),{'mapping':[{'montage':[0.,2.],'original':[1.,3.]}]}


def test_blind_prompt_has_no_privileged_annotation():
    x=item();x['annotation']['observation']='SECRET_OBSERVATION';x['answer']='GOLD_LABEL_SENTINEL'
    prompt=q.blind_prompt(x,[])
    assert 'SECRET_OBSERVATION' not in prompt and 'OLD_CAPTION_SENTINEL' not in prompt and 'GOLD_LABEL_SENTINEL' not in prompt
    assert all(choice in prompt for choice in x['choices'])


def test_original_pass_skips_all_reannotation():
    x=item();api=FakeAPI({'original_blind_montage_v2':blind(),'original_observation':audit()})
    with patch.object(q,'render_media',media):r=q.process_item(x,api,None,None)
    assert r['status']=='retained_original' and r['annotation']==x['annotation']
    assert [c[0] for c in api.calls]==['original_blind_montage_v2','original_observation']


def test_wrong_blind_answer_skips_observation_audit():
    api=FakeAPI({'original_blind_montage_v2':blind('A')})
    with patch.object(q,'render_media',media):r=q.two_gates(item(),item()['annotation'],api,None,'original')
    assert not r['passed'] and len(api.calls)==1


@pytest.mark.parametrize('status',['FAIL','UNKNOWN'])
def test_claim_rejection_cannot_be_overridden_by_global_pass(status):
    a=audit();a['claims'][0]['verdict']=status
    q.validate_observation(a,10.)
    assert not q.observation_passes(a)


def test_missing_boolean_not_pass():
    a=audit();a['all_claims_checked']='true'
    with pytest.raises(ValueError):q.validate_observation(a,10.)


@pytest.mark.parametrize('spans',[[[-1,2]],[[1,11]],[[True,4]],[[float('nan'),2]]])
def test_invalid_intervals_are_rejected(spans):
    with pytest.raises(ValueError):q.intervals_checked(spans,10.)


def test_overlap_union_does_not_duplicate_or_add_video():
    spans=[[3,5],[1,4],[7,8]]
    assert q.intervals_checked(spans,10.)==[[1.,5.],[7.,8.]]
    assert spans==[[3,5],[1,4],[7,8]]


def test_clue_evidence_cannot_cite_an_omitted_gap():
    a=blind();a['evidence'][0].update(start=4,end=5)
    with pytest.raises(ValueError):q.validate_blind(a,[[1,3],[6,8]])


def test_three_option_question_cannot_answer_nonexistent_d():
    with pytest.raises(ValueError):q.validate_blind(blind('D'),[[1,3]],'ABC')
    q.validate_blind(blind('C'),[[1,3]],'ABC')


def test_repair_must_repass_both_checks():
    values={'original_blind_montage_v2':blind('A'),
            'repair1_facts_v2':{'facts':[{'start':1,'end':2,'modality':'visual','fact':'A dog appears.'}],
                             'inferences':[],'clue_intervals':[[1,3]],'uncertainties':[]},
            'repair1_blind_montage_v2':blind(),
            'repair1_write':{'observation':'A dog appears.','uncertainties':[]},
            'repair1_observation':audit()}
    api=FakeAPI(values)
    with patch.object(q,'render_media',media):r=q.process_item(item(),api,None,None)
    assert r['status']=='reannotated_verified' and r['reannotation_attempts']==1
    values['repair1_observation']=audit('UNKNOWN');api=FakeAPI(values)
    with patch.object(q,'render_media',media):r=q.process_item(item(),api,None,None,max_repairs=1)
    assert r['status']=='needs_review' and r['annotation'] is None


def test_transport_error_does_not_reannotate():
    class Broken:
        def call(self,*a,**k):raise RuntimeError('network failure')
    with patch.object(q,'render_media',media),pytest.raises(RuntimeError):
        q.process_item(item(),Broken(),None,None)


def test_montage_time_conversion_handles_omitted_gap():
    mapping=[{'montage':[0,5],'original':[31,36]},{'montage':[5,10],'original':[70,75]}]
    result=q.original_evidence_times({'answer':'B','evidence':[{'start':4,'end':7,'fact':'an event','modality':'visual'}]},mapping)
    assert result['evidence'][0]['original_intervals']==[[35,36],[70,72]]
    assert result['evidence_time_base']=='montage_seconds'


def test_decimal_rounding_tolerance_does_not_accept_large_overrun():
    a=blind();a['evidence'][0].update(start=0,end=28.53)
    q.validate_blind(a,[[0,28.528]])
    a['evidence'][0]['end']=29
    with pytest.raises(ValueError):q.validate_blind(a,[[0,28.528]])


def test_rejected_audit_bad_timestamp_routes_to_repair_not_admission():
    a=audit('FAIL');a['claims'][0]['end']=11
    q.validate_observation(a,10)
    assert not q.observation_passes(a) and not a['claims'][0]['timestamp_within_media']
    a=audit('PASS');a['claims'][0]['end']=11
    with pytest.raises(ValueError):q.validate_observation(a,10)


def test_counting_question_can_preserve_more_than_four_events():
    spans=[[i*2,i*2+1] for i in range(9)]
    assert q.intervals_checked(spans,20)==spans


def test_truncated_json_never_passes_or_enters_success_cache(tmp_path):
    mp4=tmp_path/'tiny.mp4';mp4.write_bytes(b'media')
    class Client:
        model=q.MODEL
        def _generate_text(self,*a,**k):return json.dumps(blind()),'length'
    api=q.ReviewAPI(Client(),tmp_path/'requests')
    with pytest.raises(RuntimeError):api.call('blind','prompt',mp4,{'file_sha256':'test'},lambda o:None)
    assert json.loads((tmp_path/'requests/blind.json').read_text())['status']=='invalid_response'

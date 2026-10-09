import copy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from omni_opsd.worldsense import quality_gate as q
from omni_opsd.worldsense import quality_gate_v2 as v
from test_worldsense_quality_gate import item, blind, audit, FakeAPI, media


def proposal(uncertain=True):
    return dict(facts=[dict(start=1,end=2,modality='visual',fact='A dog appears.')],
                inferences=[],clue_intervals=[[1,3]],uncertainties=['Millisecond timing approximate.'] if uncertain else [])


def review(blocking=False):
    return dict(decisions=[dict(index=0,severity='blocking' if blocking else 'non_blocking',
                               reason='Species is visible independently of precise time.',evidence=proposal()['facts'])],
                focus_intervals=[[1,3]] if blocking else [])


def previous():
    return {'checks':[dict(reason='unresolved_facts',proposal=proposal())]}


def success(prefix='v2_repair1'):
    return {prefix+'_facts':proposal(), prefix+'_triage':review(),
            prefix+'_blind':blind(),prefix+'_write':dict(observation='A dog appears.',uncertainties=[]),
            prefix+'_observation':audit()}


def test_nonblocking_uncertainty_still_requires_both_gates():
    api=FakeAPI(success())
    with patch.object(q,'render_media',media):
        result=v.process_item(item(),previous(),api,None)
    assert result['status']=='reannotated_verified'
    stages=[s for s,_ in api.calls]
    assert 'v2_repair1_blind' in stages and 'v2_repair1_observation' in stages
    assert all('_focus' not in s for s in stages)


def test_nonblocking_does_not_override_failed_observation_audit():
    values=success(); values['v2_repair1_observation']=audit('FAIL')
    with patch.object(q,'render_media',media):
        result=v.process_item(item(),previous(),FakeAPI(values),None,max_repairs=1)
    assert result['status']=='needs_review' and result['annotation'] is None


def test_blocking_issues_get_focus_and_second_round_not_break():
    values=success('v2_repair2')
    values.update({'v2_repair1_facts':proposal(),'v2_repair1_triage':review(True),
                   'v2_repair1_focus':dict(findings=proposal()['facts'],remaining_uncertainties=['Still unclear.']),
                   'v2_repair1_facts_after_focus':proposal(),'v2_repair1_triage_after_focus':review(True)})
    api=FakeAPI(values)
    with patch.object(q,'render_media',media):
        result=v.process_item(item(),previous(),api,None)
    assert result['status']=='reannotated_verified' and result['reannotation_attempts']==2
    second=next(prompt for stage,prompt in api.calls if stage=='v2_repair2_facts')
    assert 'Still unclear.' in second and 'blocking_uncertainty_after_focus' in second
    assert 'v2_repair1_blind' not in [s for s,_ in api.calls]


def test_blind_request_does_not_receive_repair_feedback():
    prior=previous(); prior['checks'][0]['proposal']['uncertainties']=['PRIVILEGED_SENTINEL']
    api=FakeAPI(success())
    with patch.object(q,'render_media',media):
        v.process_item(item(),prior,api,None)
    prompt=next(p for s,p in api.calls if s=='v2_repair1_blind')
    assert 'PRIVILEGED_SENTINEL' not in prompt and 'golden_answer' not in prompt and 'OLD_CAPTION_SENTINEL' not in prompt


def test_no_original_gold_label_in_fact_feedback():
    check=dict(reason='observation_or_gold_not_verified',observation_audit=audit('FAIL'))
    check['observation_audit']['reason']='SECRET_GOLD_LABEL'
    result=v.feedback(check)
    assert 'SECRET_GOLD_LABEL' not in str(result)
    assert result['rejected_observation_claims'][0]['claim']=='A dog appears.'


@pytest.mark.parametrize('mutation', ['missing','duplicate','bad_severity','missing_reason','out_of_bounds_focus'])
def test_triage_schema_cannot_silently_clear_unknowns(mutation):
    result=review()
    if mutation=='missing':result['decisions']=[]
    elif mutation=='duplicate':result['decisions']*=2
    elif mutation=='bad_severity':result['decisions'][0]['severity']='PASS'
    elif mutation=='missing_reason':result['decisions'][0]['reason']=''
    else:result['focus_intervals']=[[0,101]]
    with pytest.raises(ValueError):v.validate_triage(result,['An uncertainty'],100)


def test_idontknow_option_guidance_distinguishes_missing_evidence_from_unreadability():
    assert 'I don\'t know' in v.TRIAGE
    assert 'inability to read/hear' in v.AUDIT


def test_long_focus_keeps_all_evidence_without_exceeding_per_request_budget():
    original=[[94,150],[150,168.035]]
    batches=v.focus_batches(original,168.035)
    assert batches==[[[94.,154.]],[[154.,168.035]]]
    assert all(sum(e-s for s,e in batch)<=60 for batch in batches)
    result=review(True);result['focus_intervals']=original
    v.validate_triage(result,['uncertain'],168.035)


def test_focus_split_preserves_gaps_and_full_coverage():
    original=[[0,25],[50,100],[200,290]]
    batches=v.focus_batches(original,300)
    assert all(sum(e-s for s,e in batch)<=60 for batch in batches)
    assert q.intervals_checked([span for batch in batches for span in batch],300)==original

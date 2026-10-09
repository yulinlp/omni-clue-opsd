"""CPU-only recovery and protocol checks; no live evaluation files are touched."""
import argparse
import contextlib
from datetime import datetime
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'scripts'))
sys.path.insert(0, str(CODE / 'src'))
import controller_worldsense_card_pool as controller
import run_worldsense_training_matched_eval as legacy


class CardPoolControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir='/tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task = {'label': 'fixture', 'model': '/fixture/model', 'adapter': '-'}
        self.rows = [dict(case_id=f'case-{i}', messages=[dict(role='user', content=f'Question {i}')])
                     for i in range(518)]
        self.data = self.root / 'data.jsonl'
        self.labels = self.root / 'labels.jsonl'
        controller.write(self.data, self.rows)
        controller.write(self.labels, [])
        for mode in ('mcq', 'openqa'):
            (self.root / mode / 'fixture').mkdir(parents=True)
        self.save('tasks.json', [self.task])
        self.save('inference_cache_views.json', {})
        self.save('card_pool_state.json', dict(units={}, legacy_tasks={}))
        self.save('queue_state.json', {'mcq/fixture': dict(status='complete'),
                                      'openqa/fixture': dict(status='generated')})
        self.save('mcq/fixture/summary.json', dict(total=518, scoring='mcq-v2:fixture'))
        self.addCleanup(patch.stopall)
        patch.object(controller, 'validate', return_value=(self.data, self.labels, self.rows)).start()

    def save(self, name, value):
        controller.atomic_json(self.root / name, value)

    def test_existing_judge_units_are_adopted_without_duplicate_add(self):
        unit = dict(task_key='score/fixture', kind='judge', status='running', partition=0)
        self.save('card_pool_state.json', dict(units={'judge/fixture/0': unit}, legacy_tasks={}))
        with patch.object(controller, 'add_units') as add, \
                patch.object(controller, 'units_for_judge') as prepare, \
                patch.object(controller, 'cpu_score') as score:
            self.assertFalse(controller.tick(self.root))
            self.assertFalse(controller.tick(self.root))
        add.assert_not_called()
        prepare.assert_not_called()
        score.assert_not_called()
        self.assertEqual(controller.load(self.root / 'queue_state.json')['score/fixture']['scheduler'], 'card-pool')
        self.assertEqual(controller.load(self.root / 'card_pool_state.json')['units'], {'judge/fixture/0': unit})

    def test_empty_judge_finalizer_resumes_after_interruption(self):
        state = controller.load(self.root / 'queue_state.json')
        state['score/fixture'] = dict(status='running', scheduler='card-pool')
        self.save('queue_state.json', state)
        folder = self.root / 'openqa/fixture'
        (folder / 'judge_v2').mkdir()
        controller.write(folder / 'judge_v2_pending_inputs.jsonl', [])
        with patch.object(controller, 'cpu_score', side_effect=RuntimeError('interrupted finalizer')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted finalizer'):
                controller.tick(self.root)
        self.assertEqual(controller.load(self.root / 'queue_state.json')['score/fixture']['status'], 'running')
        with patch.object(controller, 'cpu_score') as score:
            self.assertTrue(controller.tick(self.root))
        empty = folder / 'judge_v2/empty_results.jsonl'
        score.assert_called_once_with(self.root, self.task, 'openqa', judge_results=[empty])
        self.assertEqual(controller.read(empty), [])
        state = controller.load(self.root / 'queue_state.json')
        self.assertEqual(state['score/fixture']['status'], 'complete')
        self.assertEqual(state['openqa/fixture']['status'], 'complete')

    def test_completed_score_reconciles_generated_openqa(self):
        state = controller.load(self.root / 'queue_state.json')
        state['score/fixture'] = dict(status='complete')
        self.save('queue_state.json', state)
        self.save('openqa/fixture/summary.json', dict(total=518, scoring_pipeline_reliable_on_calibration=True))
        with patch.object(controller, 'cpu_score') as score:
            self.assertTrue(controller.tick(self.root))
        score.assert_not_called()
        self.assertEqual(controller.load(self.root / 'queue_state.json')['openqa/fixture']['status'], 'complete')

    def orphan_fixture(self):
        key = 'judge/fixture/0'
        folder = self.root / 'openqa/fixture/judge_v2/card_0'
        folder.mkdir(parents=True)
        unit = dict(status='running', allocation_stage='claimed-not-started', claim_id='nonce',
                    kind='judge', task_key='score/fixture', partition=0, worker='fixture-worker',
                    worker_pid=100, hostname='fixture-host', assigned_device='7',
                    claimed_epoch=time.time() - 60, folder=str(folder), attempt=1,
                    command=['unchanged-infer'], expected_rows=10)
        self.save('card_pool_state.json', dict(units={key: unit}, legacy_tasks={}))
        h = self.root / 'card_pool_workers/fixture-worker'
        h.mkdir(parents=True)
        self.save('card_pool_workers/fixture-worker/heartbeat.json',
                  dict(worker_pid=100, hostname='fixture-host', status='running',
                       heartbeat_at=datetime.now().astimezone().isoformat(),
                       devices={'7': dict(status='owned_unit', unit_id='judge/other/2')}))
        return key, unit, folder

    def test_later_live_heartbeat_recovers_only_unstarted_orphan(self):
        key, before, folder = self.orphan_fixture()
        recovered = controller.recover_unstarted_claims(self.root)
        self.assertEqual([r['unit_id'] for r in recovered], [key])
        after = controller.load(self.root / 'card_pool_state.json')['units'][key]
        self.assertEqual(after['status'], 'pending')
        self.assertNotIn('assigned_device', after)
        self.assertNotIn('claim_id', after)
        for field in ('command', 'expected_rows', 'attempt', 'partition', 'folder'):
            self.assertEqual(after[field], before[field])
        self.assertEqual(after['recovered_unstarted_claims'][0]['claim_id'], 'nonce')
        self.assertEqual(controller.recover_unstarted_claims(self.root), [])

    def test_active_slot_output_and_missing_heartbeat_are_never_recovered(self):
        key, unit, folder = self.orphan_fixture()
        h = self.root / 'card_pool_workers/fixture-worker/heartbeat.json'
        heartbeat = controller.load(h)
        heartbeat['devices']['7']['unit_id'] = key
        controller.atomic_json(h, heartbeat)
        self.assertEqual(controller.recover_unstarted_claims(self.root), [])
        heartbeat['devices']['7']['unit_id'] = 'judge/other/2'
        controller.atomic_json(h, heartbeat)
        for name in ('pid', 'assigned_device.json', 'infer.log', 'results.jsonl'):
            evidence = folder / name
            evidence.write_text('launch evidence\n')
            self.assertEqual(controller.recover_unstarted_claims(self.root), [], name)
            evidence.unlink()
        h.unlink()
        self.assertEqual(controller.recover_unstarted_claims(self.root), [])
        self.assertEqual(controller.load(self.root / 'card_pool_state.json')['units'][key], unit)

    def test_fresh_claim_stale_or_different_worker_cannot_be_recovered(self):
        key, unit, folder = self.orphan_fixture()
        h = self.root / 'card_pool_workers/fixture-worker/heartbeat.json'
        original = controller.load(h)
        for change in ({'worker_pid': 999}, {'hostname': 'another-host'},
                       {'heartbeat_at': datetime.fromtimestamp(time.time()-120).astimezone().isoformat()}):
            controller.atomic_json(h, dict(original, **change))
            self.assertEqual(controller.recover_unstarted_claims(self.root), [])
        controller.atomic_json(h, original)
        state = controller.load(self.root / 'card_pool_state.json')
        state['units'][key]['claimed_epoch'] = time.time()-1
        self.save('card_pool_state.json', state)
        self.assertEqual(controller.recover_unstarted_claims(self.root), [])

    def test_reservation_replaced_before_lock_is_not_recovered(self):
        key, unit, folder = self.orphan_fixture()
        @contextlib.contextmanager
        def changed_reservation(*args, **kwargs):
            state = controller.load(self.root / 'card_pool_state.json')
            state['units'][key]['claim_id'] = 'new-owner-nonce'
            self.save('card_pool_state.json', state)
            yield
        with patch.object(controller, 'DirectoryLock', changed_reservation):
            self.assertEqual(controller.recover_unstarted_claims(self.root), [])
        after = controller.load(self.root / 'card_pool_state.json')['units'][key]
        self.assertEqual(after['status'], 'running')
        self.assertEqual(after['claim_id'], 'new-owner-nonce')

    def generation_units(self):
        units = []
        for i in range(8):
            target = self.root / f'partition-{i}.jsonl'
            controller.write(target, [dict(row, response='A') for row in self.rows[i::8]])
            units.append(dict(partition=i, result_path=str(target)))
        return units

    def test_merge_requires_eight_disjoint_partitions_and_exact_ids(self):
        units = self.generation_units()
        controller.merge_generation(self.root, self.task, 'mcq', list(reversed(units)))
        result = self.root / 'mcq/fixture/results.jsonl'
        good_bytes = result.read_bytes()
        merged = controller.read(result)
        self.assertEqual(len(merged), 518)
        self.assertEqual({r['case_id'] for r in merged}, {r['case_id'] for r in self.rows})
        for bad in (units[:-1], units[:-1] + [dict(units[-1], partition=6)]):
            with self.assertRaisesRegex(ValueError, 'exactly the original eight'):
                controller.merge_generation(self.root, self.task, 'mcq', bad)
        shard = controller.read(Path(units[0]['result_path']))
        shard[1]['case_id'] = shard[0]['case_id']
        controller.write(Path(units[0]['result_path']), shard)
        with self.assertRaisesRegex(ValueError, 'duplicate prediction ID'):
            controller.merge_generation(self.root, self.task, 'mcq', units)
        self.assertEqual(result.read_bytes(), good_bytes, 'Rejected merge replaced the prior valid result')

    def test_pool_command_matches_legacy_rng_and_partition_commands(self):
        class FinishedProcess:
            returncode = 0
            pid = 123
            def poll(self):
                return 0
            def wait(self, timeout=None):
                return 0

        for mode, adapter in (('mcq', '-'), ('openqa', '/fixture/adapter')):
            commands = []
            def fake_infer(command, **kwargs):
                commands.append(command)
                source = Path(command[command.index('--val_dataset') + 1])
                target = Path(command[command.index('--result_path') + 1])
                legacy.write(target, [dict(row, response='A') for row in legacy.read(source)])
                return FinishedProcess()
            args = argparse.Namespace(root=self.root, mode=mode, model=self.task['model'],
                                      adapter=adapter, label='fixture', devices='0,1,2,3,4,5,6,7',
                                      validate_only=False, score_only=False)
            with patch.object(legacy, 'validate', return_value=(self.data, self.labels, self.rows)), \
                    patch.object(legacy.subprocess, 'Popen', side_effect=fake_infer), \
                    patch.object(legacy.subprocess, 'run'), contextlib.redirect_stdout(io.StringIO()):
                legacy.run(args)
            self.assertEqual(len(commands), 8)
            for i, command in enumerate(commands):
                folder = self.root / mode / 'fixture/shards' / f'card_{i}'
                self.assertEqual(command, controller.generation_command(
                    self.task['model'], adapter, folder / 'input.jsonl', folder / 'results.jsonl', mode))
                self.assertEqual(legacy.read(folder / 'input.jsonl'), self.rows[i::8])
                for name, value in (('--seed', '20260904'), ('--temperature', '1'),
                                    ('--top_p', '1'), ('--top_k', '-1'),
                                    ('--repetition_penalty', '1'), ('--val_dataset_shuffle', 'false')):
                    self.assertEqual(command[command.index(name) + 1], value)
                self.assertEqual(command[command.index('--max_new_tokens') + 1], '8' if mode == 'mcq' else '512')


if __name__ == '__main__':
    unittest.main()

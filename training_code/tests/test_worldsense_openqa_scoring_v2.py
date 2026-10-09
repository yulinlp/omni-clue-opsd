"""CPU regressions for actual order misgrades, conservative count rules, uncertainty."""
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training_code/scripts'))
import score_worldsense_openqa_v2 as scoring


class OpenQAScoringV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = REPO / 'training_runs/worldsense_observation_eval_20261001'
        cls.labels = {r['sample_id']: r for r in scoring.read(root / 'data/worldsense.openqa.labels.jsonl')}
        cls.old = {}
        for arm in ('base', 'sft_lora_epoch3'):
            cls.old[arm] = {r['sample_id']: r for r in scoring.read(root / 'openqa' / arm / 'scored.jsonl')}

    def test_real_base_wrong_event_orders(self):
        for key in ('BeWavfZy::task0', 'IWWlNWAT::task0', 'PFWtSgmy::task0', 'heOrfYdb::task0'):
            with self.subTest(key=key):
                row = self.old['base'][key]
                result, method, evidence = scoring.deterministic_rule(self.labels[key], row['final_answer'])
                self.assertIs(result, False)
                self.assertEqual(method, 'event-sequence-rule')
                self.assertNotEqual(evidence['reference_sequence'], evidence['candidate_sequence'])

    def test_real_correct_paraphrased_sequence_previously_rejected(self):
        key = 'IWWlNWAT::task0'
        answer = self.old['sft_lora_epoch3'][key]['final_answer']
        self.assertIs(scoring.deterministic_rule(self.labels[key], answer)[0], True)

    def test_real_annotated_wrong_sequences_previously_accepted(self):
        for key in ('heOrfYdb::task0', 'yMhAdUta::task1'):
            with self.subTest(key=key):
                answer = self.old['sft_lora_epoch3'][key]['final_answer']
                self.assertIs(scoring.deterministic_rule(self.labels[key], answer)[0], False)

    def test_event_rule_needs_question_definitions(self):
        label = {'question': 'What is the order in the video?', 'gold_answer_text': '(a)(b)(c)'}
        self.assertIsNone(scoring.deterministic_rule(label, '(a)(b)(c)')[0])

    def test_actual_undefined_event_reference_is_not_a_model_error(self):
        problem = scoring.reference_insufficiency(self.labels['xpALqvCU::task1'])
        self.assertIsNotNone(problem)
        self.assertEqual(set(problem['missing_event_codes']), set('eakdfbchg'))
        flagged = [key for key, label in self.labels.items() if scoring.reference_insufficiency(label)]
        self.assertEqual(flagged, ['xpALqvCU::task1'])

    def test_reference_audit_accepts_defined_events_and_intentional_unknown(self):
        label = {'question_type': 'Event Sorting', 'question': '(a) First event. (b) Second event. What is the order?',
                 'gold_answer_text': '(a) (b).'}
        self.assertIsNone(scoring.reference_insufficiency(label))
        self.assertIsNone(scoring.reference_insufficiency(self.labels['PdpWrLsy::task0']))

    def test_ambiguous_or_partial_event_answer_needs_judge(self):
        label = self.labels['IWWlNWAT::task0']
        for answer in ('a, c, b', 'a,c,b,d,e or a,b,c,d,e', 'a,c,b,d,d',
                       'The wrong sequence is: a. Throwing something at a hat. b. Handing a book to the other hand. c. Putting on a hat. d. Writing on the book. e. Bumping fists.'):
            with self.subTest(answer=answer):
                self.assertIsNone(scoring.deterministic_rule(label, answer)[0])

    def test_letter_order_does_not_override_conflicting_event_text(self):
        label = self.labels['IWWlNWAT::task0']
        answer = 'a. Putting on a hat. c. Throwing something at a hat. b. Handing a book to the other hand. d. Writing on the book. e. Bumping fists.'
        self.assertIsNone(scoring.deterministic_rule(label, answer)[0])

    def test_only_final_answer_is_used(self):
        answer, analysis, valid, _ = scoring.extract('<analysis>There were six people.</analysis>')
        self.assertEqual(answer, '')
        self.assertFalse(valid)
        self.assertIn('six', analysis)
        answer, _, valid, _ = scoring.extract('<analysis>Six.</analysis><answer>Five.')
        self.assertEqual(answer, 'Five.')
        self.assertTrue(valid)

    def test_short_count_equivalence_and_wrong_number(self):
        label = {'question': 'How many times did applause occur?', 'gold_answer_text': 'Four times.'}
        self.assertIs(scoring.deterministic_rule(label, '4')[0], True)
        self.assertIs(scoring.deterministic_rule(label, 'Four times.')[0], True)
        self.assertIs(scoring.deterministic_rule(label, 'Five times.')[0], False)

    def test_only_bare_absent_choice_references_are_rejected(self):
        label = {'question': 'Are there any blue race cars?', 'gold_answer_text': 'No, there are not.'}
        for answer in ('None of the above.', 'All of the above', 'None of above.'):
            self.assertIs(scoring.route_answer(label, answer)[0], False)
            self.assertEqual(scoring.route_answer(label, answer)[1], 'option-dependent-answer-rule')
        self.assertIsNone(scoring.route_answer(label, 'None of the above; there are no blue race cars.')[0])
        self.assertIsNone(scoring.route_answer(label, 'None.')[0])

    def test_calibration_uses_same_rule_before_semantic_judge(self):
        key = 'IWWlNWAT::task0'
        result, route, _ = scoring.route_answer(self.labels[key], 'c, b, d, a, e')
        self.assertIs(result, False)
        self.assertEqual(route, 'event-sequence-rule')

    def test_numeric_rule_does_not_grab_arbitrary_number_or_wrong_unit(self):
        label = {'question': 'How many types of sports events were shown?', 'gold_answer_text': 'Six.'}
        for answer in ('At 00:06 five events appeared.', 'Six or seven.', 'Six times.',
                       'Six sports events.', 'There were six types but five in total.', '-6'):
            with self.subTest(answer=answer):
                self.assertIsNone(scoring.deterministic_rule(label, answer)[0])
        self.assertIs(scoring.deterministic_rule(label, '6 types.')[0], True)

    def test_integer_parser(self):
        for text, expected in [('twenty-three', 23), ('one hundred and twenty three', 123),
                               ('two thousand and five', 2005), ('zero', 0), ('13', 13)]:
            self.assertEqual(scoring.integer_words(text), expected)
        for text in ('two three', 'twenty twenty', 'one and two', 'four or five', '-5', '2.5'):
            self.assertIsNone(scoring.integer_words(text))

    def test_malformed_judge_output_is_uncertain_not_zero(self):
        for output in ('YES', '{"verdict":"YES"}', 'truncated {', '[]'):
            self.assertEqual(scoring.parse_judgement(output)['verdict'], 'UNCERTAIN')

    def test_dual_reviews_must_agree(self):
        yes = {'verdict': 'YES'}
        no = {'verdict': 'NO'}
        unknown = {'verdict': 'UNCERTAIN'}
        self.assertIs(scoring.combine_reviews([yes, yes]), True)
        self.assertIs(scoring.combine_reviews([no, no]), False)
        self.assertIsNone(scoring.combine_reviews([yes, no]))
        self.assertIsNone(scoring.combine_reviews([yes, unknown]))

    def test_empty_swift_think_wrapper_is_accepted_nonempty_is_not(self):
        obj = {'verdict': 'YES', 'reference_facts': ['red'], 'candidate_facts': ['red'], 'reason': 'Same color.'}
        self.assertEqual(scoring.parse_judgement('<think>\n\n</think>\n\n' + json.dumps(obj)), obj)
        self.assertEqual(scoring.parse_judgement('<think>Some reasoning</think>' + json.dumps(obj))['verdict'], 'UNCERTAIN')

    def test_metric_uncertainty_is_explicit(self):
        rows = [{'semantic_correct': v, 'normalized_exact_match': False, 'format_ok': True}
                for v in (True, False, None)]
        metrics = scoring.metrics(rows)
        self.assertIsNone(metrics['accuracy'])
        self.assertEqual(metrics['uncertain'], 1)
        self.assertAlmostEqual(metrics['accuracy_lower_bound'], 1 / 3)
        self.assertAlmostEqual(metrics['accuracy_upper_bound'], 2 / 3)

    def test_display_order_swap_never_swaps_reference_roles(self):
        label = {'question': 'What color?', 'gold_answer_text': 'Red.'}
        first = scoring.judge_source('case', label, 'Blue.', 0)['messages'][0]['content']
        second = scoring.judge_source('case', label, 'Blue.', 1)['messages'][0]['content']
        for prompt in (first, second):
            data = json.loads(prompt.split('DATA:\n', 1)[1])
            self.assertEqual(data['reference_answer'], 'Red.')
            self.assertEqual(data['candidate_final_answer'], 'Blue.')
        self.assertNotEqual(first, second)

    def test_calibration_covers_real_scope_and_partial_answer_misgrades(self):
        cases = {c['case_id']: c for c in scoring.calibration_cases(list(self.labels.values()))}
        expectations = {
            'calibration-real-scope-yes-without-unasked-reason': True,
            'calibration-question-why-needs-reason': False,
            'calibration-real-scope-partial-left': False,
            'calibration-real-scope-partial-rear': False,
            'calibration-real-scope-rear-left-paraphrase': True,
            'calibration-real-scope-bent-paraphrase': True,
            'calibration-real-scope-reversed-change': False,
        }
        for key, expected in expectations.items():
            self.assertIs(cases[key]['expected'], expected)

    def test_cpu_end_to_end_disagreement_retains_null_and_interval(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            labels, sources, predictions = [], [], []
            for i, (tier, response) in enumerate([
                    ('A', '<analysis>A car.</analysis><answer>Red.</answer>'),
                    ('B', '<analysis>A car.</analysis>'),
                    ('D', '<analysis>A car.</analysis><answer>Crimson.</answer>')]):
                key = f'test-{i}'
                labels.append({'sample_id': key, 'video_id': key, 'tier': tier,
                               'question_type': 'Attribute Recognition',
                               'question': f'What color is car {i}?', 'gold_answer_text': 'Red.',
                               'choices': ['Red.', 'Blue.'], 'options_dependent_wording': False})
                source = {'case_id': key, 'messages': [{'role': 'user', 'content': f'What color is car {i}?'}]}
                sources.append(source)
                predictions.append({'messages': source['messages'], 'response': response})
            for name, rows in [('labels', labels), ('dataset', sources), ('results', predictions)]:
                scoring.write(temp / (name + '.jsonl'), rows)
            command = [sys.executable, str(REPO / 'training_code/scripts/score_worldsense_openqa_v2.py'),
                       '--results', str(temp / 'results.jsonl'), '--labels', str(temp / 'labels.jsonl'),
                       '--dataset', str(temp / 'dataset.jsonl'), '--output', str(temp / 'out/summary.json'),
                       '--arm', 'test']
            subprocess.run(command + ['--prepare-only'], check=True, stdout=subprocess.DEVNULL)
            inputs = scoring.read(temp / 'out/judge_v2_inputs.jsonl')
            expected = {c['case_id']: c['expected'] for c in scoring.read(temp / 'out/judge_v2_calibration_cases.jsonl')}
            outputs = []
            for source in inputs:
                key, pass_number = source['case_id'].rsplit('::judge', 1)
                verdict = ('YES' if expected[key] else 'NO') if key in expected else ('YES' if pass_number == '0' else 'NO')
                evidence = {'verdict': verdict, 'reference_facts': ['red'], 'candidate_facts': ['color'], 'reason': 'Fixture comparison.'}
                outputs.append({'messages': source['messages'], 'response': json.dumps(evidence)})
            scoring.write(temp / 'judge_results.jsonl', outputs)
            subprocess.run(command + ['--judge-results', str(temp / 'judge_results.jsonl')],
                           check=True, stdout=subprocess.DEVNULL)
            summary = json.loads((temp / 'out/summary.json').read_text())
            self.assertEqual(summary['judge_calibration']['accuracy'], 1)
            self.assertEqual(summary['uncertain'], 1)
            self.assertIsNone(summary['accuracy'])
            self.assertAlmostEqual(summary['accuracy_lower_bound'], 1 / 3)
            self.assertAlmostEqual(summary['accuracy_upper_bound'], 2 / 3)
            scored = {r['sample_id']: r for r in scoring.read(temp / 'out/scored.jsonl')}
            self.assertIsNone(scored['test-2']['semantic_correct'])
            self.assertTrue(scored['test-2']['needs_review'])


if __name__ == '__main__':
    unittest.main()

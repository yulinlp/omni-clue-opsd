import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from score_generalization_analysis_mcq_v3 import parse_analysis_option


class SentenceFormatsTest(unittest.TestCase):
    def test_explicit_answer_sentences(self):
        for body in ["The correct answer is D because it matches.",
                     "Given the reference, the correct answer is 'D'.",
                     "Therefore, the correct answer is D, as it matches.",
                     'D is the correct response because the scene confirms it.',
                     "The sequence presented as the answer 'D' is correct.",
                     'Answer should be: D.', 'I will choose D.',
                     '<p> D. </p>', "The correct answer is 'D.'"]:
            with self.subTest(body=body):
                self.assertEqual(parse_analysis_option('<answer>'+body+'</answer>')['prediction'], 'D')

    def test_misplaced_final_fields(self):
        for text in ['<analysis>Evidence.\n<answer>C.</answer>\n</analysis>',
                     '<analysis>Evidence.\nAnswer: C.</analysis>',
                     '<analysis>Evidence.\nAnswer is C.\n</analysis>']:
            with self.subTest(text=text):
                self.assertEqual(parse_analysis_option(text)['prediction'], 'C')

    def test_do_not_infer_from_natural_language_or_reasoning(self):
        for text in ['<answer>A dog.</answer>', '<answer>Angry.</answer>',
                     '<analysis>The correct answer is D because it matches.</analysis>',
                     '<answer>The answer is D or C.</answer>',
                     '<answer>The answer is A dog.</answer>',
                     '<answer>B is wrong and C is not clear.</answer>',
                     '<answer>AC</answer>', '<answer>E.</answer>']:
            with self.subTest(text=text):
                self.assertIsNone(parse_analysis_option(text)['prediction'])

    def test_conflicting_final_statements(self):
        for text in ['<answer>The answer is A. The correct answer is B.</answer>',
                     '<answer>B</answer><answer>The answer is A because...</answer>',
                     '<answer>A. Text.\nB. Other text.</answer>']:
            with self.subTest(text=text):
                self.assertIsNone(parse_analysis_option(text)['prediction'])


if __name__ == '__main__':
    unittest.main()

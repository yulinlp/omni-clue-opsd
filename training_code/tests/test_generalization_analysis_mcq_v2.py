import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from score_generalization_analysis_mcq_v2 import parse_analysis_option


class ExplicitFormatsTest(unittest.TestCase):
    def test_explicit_recoveries(self):
        cases = {
            '<analysis>Evidence.</analysis>\n<B>': 'B',
            '<analysis>Evidence.</analysis>\n<D>\n</D>': 'D',
            '<analysis>Evidence.</analysis>\n<D>...</D>': 'D',
            '<analysis>Evidence.</analysis>\n<A>\nExplanation.\n</answer>': 'A',
            'Some evidence.\n<analysis>D</analysis>': 'D',
            'Some evidence.\n<analysis>\nC.\n</analysis>.': 'C',
            '<analysis>Evidence.</analysis><solution>B</solution>': 'B',
            'A. is unlikely.\nC. also unlikely.\n\nB': 'B',
            '<analysis>Evidence.</analysis>\nOption B.': 'B',
            '<analysis>Evidence, still open.\nAnswer: C': 'C',
            '<analysis>Evidence, still open.\n<B>': 'B',
        }
        for text, letter in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_analysis_option(text)['prediction'], letter)

    def test_reasoning_and_malformed_text_not_answers(self):
        cases = [
            '<analysis>Option B is plausible.</analysis>',
            '<analysis>The answer is C',
            '<analysis>The answer is C.</analysis>',
            '<analysis><B> is merely an example.</analysis>',
            '<analysis>B</analysis>\nMore reasoning without a final answer.',
            '<C5></C5>', '<C13B944D-1C21-4483-A774-3207A45AC644>',
            '<answer></answer>', '<answer>A dog</answer>',
            'A or B', 'The answer is the red cup.',
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertIsNone(parse_analysis_option(text)['prediction'])

    def test_conflicting_explicit_choices(self):
        for text in ['<B>C</B>', '<c>D</c>', '<B></C>',
                     'Therefore, the answer is D.\n\n<analysis>B</analysis>',
                     '<solution>A</solution><answer>B</answer>',
                     '<B>\n<C>', '<answer>A</answer><answer>B</answer>\nD']:
            with self.subTest(text=text):
                self.assertIsNone(parse_analysis_option(text)['prediction'])

    def test_existing_explicit_answers(self):
        for text in ['<answer>B</answer>', '<answer>B.</answer>',
                     '<answer>Option: B</answer>', '<option>B</option>',
                     '<analysis>A or C.</analysis><answer>B</answer>',
                     'A. Wrong.\nC. Wrong.<answer>B</answer>',
                     '<C>\n<answer>B</answer>',
                     '<answer>B</answer>\n<analysis>A</analysis>',
                     '<answer>Choice is B']:
            with self.subTest(text=text):
                self.assertEqual(parse_analysis_option(text)['prediction'], 'B')


if __name__ == '__main__':
    unittest.main()

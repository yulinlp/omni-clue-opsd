import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from score_generalization_analysis_mcq import parse_analysis_option

class AnalysisChoiceTest(unittest.TestCase):
    def test_actual_option_prefix(self):
        self.assertEqual(parse_analysis_option('<analysis>Answer A is unlikely.</analysis><answer>Option: B</answer>')['prediction'],'B')

    def test_option_tag_and_unfinished_tag(self):
        self.assertEqual(parse_analysis_option('<analysis>Consider C.</analysis><option>D</option>')['prediction'],'D')
        self.assertEqual(parse_analysis_option('<analysis>Evidence.</analysis><answer>Choice is A')['prediction'],'A')

    def test_no_answer_inferred_from_analysis(self):
        self.assertIsNone(parse_analysis_option('<analysis>Option: B is plausible.</analysis>')['prediction'])
        self.assertIsNone(parse_analysis_option('<analysis>The answer is C')['prediction'])

    def test_article_and_conflicts_rejected(self):
        self.assertIsNone(parse_analysis_option('<answer>Option: A dog</answer>')['prediction'])
        self.assertIsNone(parse_analysis_option('<answer>Option: A</answer><option>B</option>')['prediction'])

    def test_plain_reasoning_does_not_compete_with_final_field(self):
        self.assertEqual(parse_analysis_option('B. Circle is round.\nC. Rhombus is different.\nD. Square is wrong. <analysis></analysis><answer>A</answer>.')['prediction'],'A')
        self.assertEqual(parse_analysis_option('The correct answer is B:\n<answer>D</answer>')['prediction'],'D')

if __name__=='__main__':unittest.main()

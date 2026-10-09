import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from score_generalization_analysis_mcq_v4 import parse_analysis_option


class FinalFormatTest(unittest.TestCase):
    choices=['Happy','Angry','Sad','Indifferent']

    def check(self,text,expected,choices=None):
        self.assertEqual(parse_analysis_option(text,choices or self.choices)['prediction'],expected,text)

    def test_damaged_final_tags(self):
        for text in ['<analysis>Evidence.</analysis> B </answer>',
                     '<analysis>Evidence.</analysis></answer>B.</answer>',
                     '<analysis>Evidence.</analysis>\\</answer\\>B\\</answer\\>',
                     '<analysis>Evidence.</analysis> B </answer>B',
                     '<analysis>Evidence.</analysis><TOPUP>B</TOPUP>',
                     '<answer>B,</answer>', '<answer>B - Angry.</answer>']:
            self.check(text,'B')

    def test_option_text(self):
        for text in ['<answer>Angry.</answer>','<answer>**ANGRY**</answer>',
                     '<solution>The final answer is Angry.</solution>',
                     '<analysis>He mentions C.</analysis>Angry.',
                     '<answertopic>Angry.</answertopic>']:
            self.check(text,'B')
        self.check('<answer>A van.</answer>','C',['Bus','Taxi','Van','APC'])
        self.check('<answer>25,000.</answer>','B',['25001','25000','5000','None of the above'])
        self.check('<answer>The third position.</answer>','C',['the 1st','the 2nd','the 3rd','the 4th'])
        self.check('<answer>Three.</answer>','B',['2','3','4','5'])

    def test_do_not_invent_decisions(self):
        for text in ['<analysis>The answer is B.</analysis>',
                     '<analysis>Angry.</analysis><answer></answer>',
                     '<answer>Not angry.</answer>',
                     '<answer>The person might be angry or sad.</answer>',
                     '<answer>Grateful.</answer>',
                     '<answer>E</answer>',
                     '<analysis>Evidence.\nB',
                     '<answer>A dog.</answer>']:
            self.check(text,None)

    def test_conflicts(self):
        for text in ['<answer>B</answer><answer>C</answer>',
                     '<answer>Angry</answer><answer>Sad</answer>',
                     '<answer>A</answer><answer>Angry</answer>',
                     '<analysis>Evidence.</analysis><D>C</D>',
                     '<analysis>Evidence.</analysis><D>\nC\n</D>',
                     '<analysis>Evidence.</analysis><D>\nC',
                     '<analysis>Evidence.</analysis>, B, C, or D.',
                     '<answer>B, C, A, D</answer>',
                     '<answer>A, B, C, or D (Given the reference...)</answer>',
                     '<analysis>Evidence.</analysis>B</answer>C']:
            self.check(text,None)
        self.check('<answer>Angry.</answer>',None,['Angry','Happy','ANGRY.','Sad'])

    def test_nested_reasoning_is_removed(self):
        self.check('<analysis>A. One thought. <analysis>B. Another.</analysis>C. More.</analysis>\nB','B')

    def test_mapping_uses_input_options(self):
        self.check('<answer>Angry</answer>','B')
        self.check('<answer>Angry</answer>','D',['Sad','Happy','Indifferent','Angry'])

    def test_letter_priority_over_musical_note_text(self):
        self.check('<answer>B</answer>','B',['E','F','G','B'])

    def test_untagged_reasoning_does_not_override_final(self):
        self.check('A and B could fit. <analysis>Evidence.</analysis>\nC</answer>','C')
        self.check('<analysis>Evidence.</analysis><answer>B</answer>\nA. Compared option\nC. Another comparison','B')
        self.check('<analysis>Evidence.</analysis>\nA. Compared option\nC. Another comparison\nB','B')

    def test_repeated_analysis_with_two_final_decisions(self):
        self.check('<analysis>Evidence.</analysis>A<analysis>More evidence.</analysis>B',None)


if __name__=='__main__':unittest.main()

#!/usr/bin/env python3
"""Use the frozen WorldSense semantic rubric with predeclared external QA issues."""
from pathlib import Path
import sys

REPO=Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd')
sys.path.insert(0,str(REPO/'training_code/src'))
sys.path.insert(0,str(REPO/'training_code/scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import score_worldsense_openqa_v2 as scorer

original=scorer.reference_insufficiency
def reference_insufficiency(label):
    if label.get('openqa_reference_issues'):
        return {'problem':'; '.join(label['openqa_reference_issues']),
                'policy':'predeclared before generation; unresolved for every model'}
    return original(label)

scorer.reference_insufficiency=reference_insufficiency

if __name__=='__main__':scorer.main()

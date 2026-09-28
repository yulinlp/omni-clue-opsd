"""ms-swift plugin for VideoOdyssey multiple-choice GRPO."""

from omni_opsd.rewards import mcq_exact_rewards
from swift.rewards import ORM, orms


class VideoOdysseyMCQAccuracy(ORM):

    def __call__(self, completions, solution, **kwargs):
        return mcq_exact_rewards(completions, solution)


orms["video_odyssey_mcq_accuracy"] = VideoOdysseyMCQAccuracy

"""Mark checkpoints resumable only after every rank and the EMA writer finish."""
import json
from pathlib import Path


def install_checkpoint_markers(cls):
    original = cls._save_checkpoint
    def save(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self.accelerator.wait_for_everyone()
        if self.args.should_save:
            checkpoint = Path(self.state.last_model_checkpoint)
            value = dict(global_step=self.state.global_step, epoch=self.state.epoch,
                         world_size=self.accelerator.num_processes,
                         has_ema_teacher=(checkpoint/'ema_teacher.safetensors').is_file(),
                         save_only_model=self.args.save_only_model)
            temp = checkpoint/'MCQ_CHECKPOINT_COMPLETE.json.tmp'
            temp.write_text(json.dumps(value, indent=2)+'\n')
            temp.replace(checkpoint/'MCQ_CHECKPOINT_COMPLETE.json')
        self.accelerator.wait_for_everyone()
        return result
    cls._save_checkpoint = save

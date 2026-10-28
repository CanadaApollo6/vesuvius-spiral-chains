#!/usr/bin/env python3
"""fit_ckpt.py - the pinned headless fit_spiral.py CLI, plus a checkpoint every N iterations so a long fit can resume.

The pinned CLI saves checkpoint_fitted.ckpt only at the end. This wrapper imports the pinned module unchanged, wraps
FitContext.step so that after every FIT_CKPT_EVERY completed iterations it calls the module's own _save_model
('autosave', completed) (atomic write; the checkpoint carries model, optimizer, scheduler, RNG and
completed_iterations), then runs the module's own __main__ block. Resume: set FIT_SPIRAL_RESUME_PATH to
<run>/checkpoint_autosave.ckpt (apply_checkpoint restores completed_iterations, so FIT_SPIRAL_RESUME_STEP is only a
fallback). Same arguments and FIT_SPIRAL_* env as fit_spiral.py.

    cd $SPIRAL_DIR && FIT_CKPT_EVERY=2500 $VILLA_PY /path/to/fit_ckpt.py --dataset <ds>
"""
import os
import sys
import textwrap

SPIRAL_DIR = os.environ.get("SPIRAL_DIR") or os.getcwd()
sys.path.insert(0, SPIRAL_DIR)
import fit_spiral  # noqa: E402  (pinned)

EVERY = int(os.environ.get("FIT_CKPT_EVERY", "2500"))
_step = fit_spiral.FitContext.step


def step(self, iteration):
    out = _step(self, iteration)
    done = iteration + 1
    if EVERY > 0 and done % EVERY == 0 and done < self.num_training_steps and self.dist.is_main_process:
        path = self._save_model("autosave", done)
        print(f"[fit_ckpt] autosave at {done} -> {path}", flush=True)
    return out


fit_spiral.FitContext.step = step
src = open(fit_spiral.__file__).read()
marker = "if __name__ == '__main__':\n"
assert src.count(marker) == 1, "pinned fit_spiral.py layout changed"
main_block = textwrap.dedent(src.split(marker, 1)[1])
sys.argv[0] = fit_spiral.__file__
exec(compile(main_block, fit_spiral.__file__ + ":__main__", "exec"), fit_spiral.__dict__)

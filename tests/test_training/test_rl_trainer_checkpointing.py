"""Regression test: the RL trainers must persist progress, not only the final model."""

import inspect

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

import thinkrl.training.grpo_trainer as grpo_trainer
import thinkrl.training.reinforce_pp_trainer as reinforce_pp_trainer
import thinkrl.training.star_trainer as star_trainer
from thinkrl.training.grpo_trainer import GRPOTrainer
from thinkrl.utils.checkpoint import CheckpointManager, load_training_checkpoint, save_training_checkpoint


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 16, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return {"logits": self.head(self.emb(input_ids))}


TRAINERS = (grpo_trainer.GRPOTrainer, reinforce_pp_trainer.ReinforcePPTrainer, star_trainer.STaRTrainer)


def test_every_rl_trainer_accepts_a_checkpoint_directory():
    """The regression: none of these took a checkpoint argument at all."""
    for trainer in TRAINERS:
        params = inspect.signature(trainer.train).parameters
        assert "checkpoint_dir" in params, f"{trainer.__name__}.train has no checkpoint_dir"
        assert "save_every" in params, f"{trainer.__name__}.train has no save_every"


def test_every_rl_trainer_reaches_the_checkpoint_manager():
    for module in (grpo_trainer, reinforce_pp_trainer, star_trainer):
        source = inspect.getsource(module)
        assert "CheckpointManager" in source, f"{module.__name__} never constructs a CheckpointManager"
        assert "save_training_checkpoint" in source, f"{module.__name__} never saves"


def test_helper_is_a_no_op_without_a_manager():
    assert save_training_checkpoint(None, model=_TinyLM(), step=1) is None


def test_helper_writes_a_loadable_checkpoint(tmp_path):
    manager = CheckpointManager(tmp_path, max_checkpoints=5)
    model = _TinyLM()

    path = save_training_checkpoint(manager, model=model, epoch=0, step=7, metrics={"loss": 1.5})

    assert path is not None and path.exists()
    assert manager.checkpoints, "checkpoint was not registered with the manager"


def test_helper_coerces_tensor_metrics(tmp_path):
    """Trainers collect metrics as a mix of floats and 0-dim tensors."""
    manager = CheckpointManager(tmp_path, max_checkpoints=5)

    path = save_training_checkpoint(
        manager,
        model=_TinyLM(),
        step=1,
        metrics={"loss": torch.tensor(0.5), "reward": 2.0, "grad": torch.ones(3), "name": "ignored"},
    )

    assert path is not None
    saved = manager.checkpoints[-1]["metrics"]
    assert saved == {"loss": 0.5, "reward": 2.0}


def test_rotation_keeps_max_checkpoints(tmp_path):
    manager = CheckpointManager(tmp_path, max_checkpoints=2)
    model = _TinyLM()

    for step in range(4):
        save_training_checkpoint(manager, model=model, step=step)

    assert len(manager.checkpoints) <= 2


def test_grpo_trainer_accepts_resume_from():
    """GRPO first: resume lands here before the other two RL trainers get it.

    A crash loses the run anyway if nothing can load the checkpoint back in.
    """
    params = inspect.signature(GRPOTrainer.train).parameters
    assert "resume_from" in params
    assert params["resume_from"].default is None, "must stay opt-in"


def test_resume_without_a_checkpoint_dir_is_rejected():
    """resume_from names a checkpoint under checkpoint_dir; without one there is nowhere
    to look, and silently training from scratch would be the wrong kind of quiet."""
    trainer = GRPOTrainer.__new__(GRPOTrainer)  # avoid constructing a real model/tokenizer

    with pytest.raises(ValueError, match="checkpoint_dir"):
        GRPOTrainer.train(trainer, resume_from="latest", checkpoint_dir=None)


def test_load_training_checkpoint_is_a_no_op_without_a_manager():
    assert load_training_checkpoint(None, "latest", model=_TinyLM()) == (0, 0)


def test_load_training_checkpoint_is_a_no_op_without_resume_from(tmp_path):
    """A manager existing is not consent to resume; resume_from is the opt-in."""
    manager = CheckpointManager(tmp_path, max_checkpoints=5)
    save_training_checkpoint(manager, model=_TinyLM(), epoch=3, step=70, metrics={"loss": 0.1})

    assert load_training_checkpoint(manager, None, model=_TinyLM()) == (0, 0)


def test_load_training_checkpoint_restores_latest(tmp_path):
    manager = CheckpointManager(tmp_path, max_checkpoints=5)
    saved = _TinyLM()
    with torch.no_grad():
        saved.head.weight.fill_(1.23)
    save_training_checkpoint(manager, model=saved, epoch=3, step=70, metrics={"loss": 0.1})

    restored = _TinyLM()
    optimizer = optim.SGD(restored.parameters(), lr=0.01)
    epoch, step = load_training_checkpoint(manager, "latest", model=restored, optimizer=optimizer)

    assert (epoch, step) == (3, 70)
    assert torch.allclose(restored.head.weight, saved.head.weight)


def test_load_training_checkpoint_restores_best(tmp_path):
    manager = CheckpointManager(tmp_path, max_checkpoints=5, metric_name="loss", mode="min")
    save_training_checkpoint(manager, model=_TinyLM(), epoch=0, step=10, metrics={"loss": 0.9})
    best = _TinyLM()
    with torch.no_grad():
        best.head.weight.fill_(7.0)
    save_training_checkpoint(manager, model=best, epoch=1, step=20, metrics={"loss": 0.1})

    restored = _TinyLM()
    epoch, step = load_training_checkpoint(manager, "best", model=restored)

    assert (epoch, step) == (1, 20)
    assert torch.allclose(restored.head.weight, best.head.weight)


def test_load_training_checkpoint_accepts_an_explicit_path(tmp_path):
    manager = CheckpointManager(tmp_path, max_checkpoints=5)
    save_training_checkpoint(manager, model=_TinyLM(), epoch=0, step=5, metrics={})
    explicit_path = manager.checkpoints[0]["path"]

    epoch, step = load_training_checkpoint(manager, str(explicit_path), model=_TinyLM())

    assert (epoch, step) == (0, 5)

from types import SimpleNamespace

import torch

from kdflow.algorithms import vanilla_kd
from kdflow.backend.fsdp.fsdp_strategy import FSDP2Strategy


class _Student:
    def __init__(self):
        self.model = SimpleNamespace(lm_head=object())

    def __call__(self, input_ids, **kwargs):
        hidden = torch.ones((*input_ids.shape, 2), requires_grad=True)
        return {"hidden_states": [hidden]}


def _micro_batch(collect_metrics):
    return {
        "stu_input_ids": torch.tensor([[1, 2, 3]]),
        "stu_attn_mask": torch.ones((1, 3), dtype=torch.long),
        "stu_loss_mask": torch.tensor([[False, True, True]]),
        "tea_input_ids": torch.tensor([[1, 2, 3]]),
        "tea_attn_mask": torch.ones((1, 3), dtype=torch.long),
        "tea_loss_mask": torch.tensor([[False, True, True]]),
        "teacher_hiddens": torch.ones((2, 2)),
        "avg_micro_batch_token_num": 2,
        "rollout_log_probs": torch.zeros((1, 3)),
        "_collect_metrics": collect_metrics,
    }


def test_vanilla_kd_skips_diagnostics_without_changing_loss(monkeypatch):
    seen_metric_fns = []

    def fake_chunked_loss(*args, metric_fns, **kwargs):
        seen_metric_fns.append(metric_fns)
        return torch.tensor(2.0, requires_grad=True), {}

    monkeypatch.setattr(vanilla_kd, "chunked_loss", fake_chunked_loss)
    algorithm = vanilla_kd.VanillaKD.__new__(vanilla_kd.VanillaKD)
    algorithm.args = SimpleNamespace(
        train=SimpleNamespace(chunked_loss_size=1024),
        kd=SimpleNamespace(kd_ratio=1.0),
    )
    algorithm.strategy = SimpleNamespace(ring_attn_group=None)
    algorithm.student = _Student()
    algorithm.teacher_lm_head = SimpleNamespace(
        weight=torch.ones((2, 2))
    )
    algorithm.loss_fn = object()
    algorithm.suppress_token_ids = ()
    algorithm.metric_fns = [object()]

    result = algorithm.training_step(_micro_batch(collect_metrics=False))

    assert result["train/loss"].item() == 1.0
    assert seen_metric_fns == [[]]


def test_optimizer_step_returns_the_single_clipped_gradient_norm():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.ones((1, 2))).sum().backward()

    strategy = FSDP2Strategy.__new__(FSDP2Strategy)
    strategy.step = 0
    strategy.max_norm = 1.0
    grad_norm = strategy.optimizer_step(optimizer, model, scheduler=None)

    assert grad_norm is not None
    assert grad_norm.item() > 0
    assert all(parameter.grad is None for parameter in model.parameters())

    strategy.step = 1
    assert strategy.optimizer_step(optimizer, model, scheduler=None) is None


def test_chunked_kd_loss_suppresses_configured_logits():
    from kdflow.loss.chunked_loss import chunked_loss

    captured = {}

    class _Head(torch.nn.Linear):
        def forward(self, hidden, skip=False):
            return super().forward(hidden)

    def capture_loss(student_logits, teacher_logits, reduction, **kwargs):
        captured["student"] = student_logits.detach().clone()
        captured["teacher"] = teacher_logits.detach().clone()
        return student_logits[:, 0] * 0

    student_head = _Head(2, 4, bias=False)
    teacher_head = torch.nn.Linear(2, 4, bias=False)
    chunked_loss(
        torch.ones((1, 2)),
        student_head,
        capture_loss,
        teacher_hidden=torch.ones((1, 2)),
        teacher_head=teacher_head,
        reduction="sum",
        suppress_token_ids=(1, 3),
    )

    assert captured["student"][0, 1] == torch.finfo(torch.float32).min
    assert captured["student"][0, 3] == torch.finfo(torch.float32).min
    assert captured["teacher"][0, 1] == torch.finfo(torch.float32).min
    assert captured["teacher"][0, 3] == torch.finfo(torch.float32).min

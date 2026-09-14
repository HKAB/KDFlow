import torch
import torch.distributed.checkpoint as dcp
from torch.utils.data import DistributedSampler, TensorDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from kdflow.backend.fsdp.fsdp_strategy import _TrainingAppState


def test_distributed_checkpoint_restores_training_state(tmp_path):
    torch.manual_seed(7)
    model = torch.nn.ModuleDict({"student": torch.nn.Linear(4, 2)})
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    loss = model["student"](torch.ones(1, 4)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    saved_weight = model["student"].weight.detach().clone()
    saved_lr = scheduler.get_last_lr()

    app_state = _TrainingAppState(model, optimizer, scheduler)
    dcp.save({"app": app_state}, checkpoint_id=tmp_path)
    with torch.no_grad():
        model["student"].weight.zero_()
    optimizer.param_groups[0]["lr"] = 9.0

    dcp.load({"app": app_state}, checkpoint_id=tmp_path)

    torch.testing.assert_close(model["student"].weight, saved_weight)
    assert scheduler.get_last_lr() == saved_lr
    assert optimizer.state


def test_stateful_dataloader_resumes_inside_epoch():
    dataset = TensorDataset(torch.arange(12))
    sampler = DistributedSampler(
        dataset, num_replicas=1, rank=0, shuffle=True, seed=17
    )
    sampler.set_epoch(2)
    loader = StatefulDataLoader(dataset, batch_size=3, sampler=sampler)
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    state = loader.state_dict()
    expected_remaining = [batch[0].tolist() for batch in iterator]

    resumed_sampler = DistributedSampler(
        dataset, num_replicas=1, rank=0, shuffle=True, seed=17
    )
    resumed_sampler.set_epoch(2)
    resumed_loader = StatefulDataLoader(
        dataset, batch_size=3, sampler=resumed_sampler
    )
    resumed_loader.load_state_dict(state)

    assert [batch[0].tolist() for batch in resumed_loader] == expected_remaining

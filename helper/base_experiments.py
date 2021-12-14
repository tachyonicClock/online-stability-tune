"""
Base Experiments Defines all control variables that should not be varied
"""
import avalanche.benchmarks.classic as avl_benchmarks
import avalanche.models as avl_models
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms

from . import strategy_builder

# Pytorch expects models to be normalized in the same way
# https://pytorch.org/vision/stable/models.html
# The origins of these values is interesting https://github.com/pytorch/vision/issues/1439
# which is so wild! Apparently they are from a random sample of imagenet
NORMALIZE=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def new_mnist(dataset_root:str, seed):
    """
    new_mnist constructs a StrategyRunner and a StrategyBuilder for the mnist
    dataset.
    """
    scenario = avl_benchmarks.SplitMNIST(
        n_experiences=10,
        dataset_root=dataset_root,
        seed=seed
    )

    builder = strategy_builder.StrategyBuilder()

    builder.use_model(
        avl_models.simple_mlp.SimpleMLP,
        num_classes=scenario.n_classes)

    builder.use_optimizer(
        torch.optim.SGD,
        lr=0.001)

    builder.use_training_args(
        train_mb_size=64,
        train_epochs=1,
        eval_mb_size=256,
        device="cuda")

    builder.use_loss_function(CrossEntropyLoss())

    runner = strategy_builder.StrategyRunner(scenario)
    builder.add_runner(runner)
    return builder, runner


def new_cifar100(dataset_root:str, seed):
    """
    new_cifar100 constructs a StrategyRunner and a StrategyBuilder for the cifar100
    dataset.
    """

    # https://pytorch.org/hub/pytorch_vision_resnet/
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        NORMALIZE,
    ])

    scenario = avl_benchmarks.ccifar100.SplitCIFAR100(
        n_experiences=10, # 10 per 10 experiences
        return_task_id=False,
        dataset_root=dataset_root,
        seed=seed,
        eval_transform=transform,
        train_transform=transform
    )

    builder = strategy_builder.StrategyBuilder()

    builder.use_model(ptcv_get_model, name="resnet18", pretrained=True)
    builder.use_optimizer(torch.optim.SGD, lr=0.005)
    builder.use_training_args(train_mb_size=64, train_epochs=1, eval_mb_size=64, device="cuda")
    builder.use_loss_function(CrossEntropyLoss())

    runner = strategy_builder.StrategyRunner(scenario)
    builder.add_runner(runner)
    return builder, runner

def load_custom_model(network):
    """Load a network using torch.load"""
    model = torch.load(network)
    assert model, f"Failed to load network {network}"
    return model

def new_core50(dataset_root: str, run):
    """
    new_core50 constructs a StrategyRunner and a StrategyBuilder for the cifar100
    dataset.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        NORMALIZE,
    ])

    scenario = avl_benchmarks.core50.CORe50(
        scenario="nc",
        run=run,
        dataset_root=dataset_root,
        train_transform=transform,
        eval_transform=transform
    )

    builder = strategy_builder.StrategyBuilder()

    builder.use_model(ptcv_get_model, name="resnet18", pretrained=True)
    builder.use_optimizer(torch.optim.SGD, lr=0.001)
    builder.use_training_args(train_mb_size=64, train_epochs=1, eval_mb_size=128, device="cuda")
    builder.use_loss_function(CrossEntropyLoss())
    builder.fix_cm(50)

    runner = strategy_builder.StrategyRunner(scenario)
    builder.add_runner(runner)
    return builder, runner

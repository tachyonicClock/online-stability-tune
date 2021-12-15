#!/usr/bin/env python
"""
CLI for experiments to aid in reproducibility
"""
import pprint
import random
import warnings
from dataclasses import dataclass
from datetime import datetime

import click
import torch
from avalanche.training import strategies
from algorithms import cost, ost, ost_tuners
from helper.base_experiments import new_cifar100, new_core50, new_mnist
from helper.strategy_builder import (StrategyBuilder, StrategyRunner,
                                           build_tuning_strategy)
from strategies.ewc import EWC
from strategies.lwf import LwF

SCENARIOS = {
    "cifar": (
        """
Cifar100 split into 10 experiences each with 10 classes.
For more information: https://www.cs.toronto.edu/~kriz/cifar.html
        """,
        new_cifar100),
    "core50": (
        """
Core50 new classes variant. As described in
Lomonaco, V., & Maltoni, D. (2017). CORe50: A New Dataset and Benchmark
for Continuous Object Recognition. Proceedings of the 1st Annual
Conference on Robot Learning, 17-26.
https://proceedings.mlr.press/v78/lomonaco17a.html
        """,
        new_core50),
    "mnist": (
        """
Modified Nist split into 10 experiences each with 1 class. For more information
http://yann.lecun.com/exdb/mnist/
        """,
        new_mnist),
}

TUNERS = {
    "ewc": ost_tuners.EWCTuner,
    "si": ost_tuners.SITuner,
    "lwf": ost_tuners.LWFTuner
}

# Cybernetic Tuners
COST_TUNERS = {
    "ewc": cost.EWCCyberneticOST,
    "lwf": cost.LwFCyberneticOST,
    "si": cost.SICyberneticOST,
    "naive": cost.LRCyberneticOST 
}

@dataclass
class MyContext():
    """Holds the context that is threaded through click"""
    builder: StrategyBuilder = None
    runner: StrategyRunner = None
    tb_logdir: str = None

def pretty_print_strategy(parameters: dict, name):
    """Nicely echo back parameters"""
    click.secho("⎯"*80, fg="blue")
    click.secho(f"Running {name}", fg="green", bold=True)
    click.secho("Parameters:", fg="green")
    pprint.pprint(parameters)
    click.secho("⎯"*80, fg="blue")

@click.group()
@click.argument("scenario_name")
@click.option("--device", default="cuda")
@click.option("--lr", default=0.001)
@click.option("--epochs", default=1)
@click.option("--train-mb-size", default=64)
@click.option("--eval-mb-size", default=64)
@click.option("--dataset", default="./data_sets", help="Path to where datasets will be downloaded and used")
@click.option("--seed", default=0, help="Seed to use for reproducibility")
@click.option("--rand", default=False, help="Overide the seed with the timestamp")
@click.option("--core-run", default=0, help="Core has multiple pre-defined runs. This is ignored for non-core scenarios")
@click.option("--tb-logdir", default=None, help="Tensorboard logdir")
@click.pass_context
def scenario(ctx, scenario_name, device, lr, epochs, train_mb_size, eval_mb_size, seed, dataset, tb_logdir, rand, core_run):
    """SCENARIO_NAME is one of [cifar, core50, mnist]"""
    assert scenario_name in SCENARIOS.keys(), "Scenario not found"
    description, benchmark = SCENARIOS[scenario_name]

    if rand:
        print("Overiding seed")
        seed = datetime.now().timestamp()

    # Setup randomness for torch & python
    random.seed(seed)
    torch.manual_seed(seed)

    dataset_root = ""
    if scenario_name == "core50":
        builder, runner = benchmark(
            dataset_root=dataset,
            run=core_run)
    else:
        builder, runner = benchmark(
            dataset_root=dataset,
            seed=seed)  # Seed used to order classes in experiences
    
    assert isinstance(builder, StrategyBuilder), str(builder)

    builder.use_optimizer(torch.optim.SGD, lr=lr)
    builder.use_training_args(
        train_mb_size=train_mb_size,
        train_epochs=epochs,
        eval_mb_size=eval_mb_size,
        device=device)

    if tb_logdir:
        builder.use_logger(tb_logdir)
        ctx.obj.tb_logdir = tb_logdir

    ctx.obj.builder = builder
    ctx.obj.runner = runner

    click.secho("About scenario:", fg="green")
    click.echo(description)


@scenario.group()
@click.option("--ewc-lambda", default=0.0, show_default=True,
              help="Stability-plasticity tradeoff")
@click.option("--clipping-threshold", default=0.0001, show_default=True,
              help="Helps protect ewc from exploding gradients")
@click.option("--decay-factor", default=0.0, show_default=True,
              help="If not 0 use online ewc")
@click.pass_context
def ewc(ctx, ewc_lambda, clipping_threshold, decay_factor):
    """Elastic Weight Consolidation"""
    builder = ctx.obj.builder
    assert isinstance(builder, StrategyBuilder)

    online = {}
    if decay_factor != 0:
        online["mode"] = "online"
        online["decay_factor"] = decay_factor

    if clipping_threshold == 0.0:
        clipping_threshold = None

    builder.use_strategy(EWC,
                         ewc_lambda=ewc_lambda,
                         **online,
                         clipping_threshold=clipping_threshold)\
        .use_label("ewc")

@scenario.group()
@click.option("--si-lambda", default=2500, show_default=True,
              help="Stability-plasticity tradeoff")
@click.pass_context
def si(ctx, si_lambda):
    """Synaptic Intelligence"""
    builder = ctx.obj.builder
    assert isinstance(builder, StrategyBuilder)
    builder.use_strategy(strategies.SynapticIntelligence, si_lambda=si_lambda) \
           .use_label("si")


@scenario.group()
@click.option("-a", "--lwf-lambda", default=10.0, show_default=True,
              help="Stability-plasticity tradeoff")
@click.option("-T", "--temperature", default=2.0, show_default=True,
              help="Softmax temperature in knowledge distilation")
@click.pass_context
def lwf(ctx, lwf_lambda, temperature):
    """Learning without Forgetting"""
    builder = ctx.obj.builder
    assert isinstance(builder, StrategyBuilder)
    builder.use_strategy(LwF, alpha=lwf_lambda, temperature=temperature) \
           .use_label("lwf")


@scenario.group()
@click.pass_context
def naive(ctx):
    """Naively use fine tuning"""
    builder = ctx.obj.builder
    assert isinstance(builder, StrategyBuilder)
    builder.use_strategy(strategies.Naive).use_label("naive")


@naive.command()
@click.pass_context
def constant(ctx):
    """Keep stability constant"""
    builder, runner = ctx.obj.builder, ctx.obj.runner
    assert isinstance(builder, StrategyBuilder)
    strategy = builder.build()
    pretty_print_strategy(builder.get_hyper_parameters(), strategy.label)
    runner.run(strategy)


@ewc.command()
@click.option("-df", "--decay-factor", default=0.8, show_default=True,
              help="Speed of exponential decay, controls the precision of the search")
@click.option("-dm", "--drop-margin", default=0.1, show_default=True,
              help="Governs how much intransigence is acceptable")
@click.pass_context
def decay(ctx, decay_factor, drop_margin):
    """
    Decay stability

    Uses my semi-Online Stability Decay algorithm (OSD) which is a simiplication
    of Continual Hyperparameter Framework by (Delange et al.)

    Delange, M., Aljundi, R., Masana, M., Parisot, S., Jia, X., Leonardis, A.,
    Slabaugh, G., & Tuytelaars, T. (2021). A continual learning survey: Defying
    forgetting in classification tasks. IEEE Transactions on Pattern Analysis
    and Machine Intelligence, 1-1. https://doi.org/10.1109/TPAMI.2021.3057446
    """
    builder = ctx.obj.builder
    assert isinstance(builder, StrategyBuilder)

    reference = builder.copy().use_strategy(strategies.Naive).use_label("ref") \
                       .without_logger()

    tuning_policy = ost.StabilityDecay(
        decay_factor=decay_factor, drop_margin=drop_margin)

    strategy = build_tuning_strategy(
        TUNERS[builder.get_label()], builder, reference, tuning_policy, ctx.obj.tb_logdir)

    pretty_print_strategy({
        "underlying": builder.get_hyper_parameters(),
        "reference": reference.get_hyper_parameters(),
        "decay_factor": decay_factor,
        "drop_margin": drop_margin}, strategy.label)

    ctx.obj.runner.run(strategy)

@ewc.command()
@click.option("-cf", "--change-factor", default=0.2, show_default=True,
              help="Speed of exponential decay/growth, controls the precision of"
                   "the search. 0.2 coressponds to a decay_factor=0.8")
@click.option("-dm", "--drop-margin", default=0.1, show_default=True,
              help="Governs how much intransigence is acceptable")
@click.option("-lr", "--tuned-lr", default=None, help="Set the learning rate of the tuned strategy")
@click.pass_context
def tune(ctx, change_factor, drop_margin, tuned_lr):
    """
    Grow and decay stability

    Uses my semi-Online Stability Tuning algorithm (OST)
    """
    builder = ctx.obj.builder
    assert isinstance(builder, StrategyBuilder)

    reference = builder.copy().use_strategy(strategies.Naive).use_label("ref") \
                   .without_logger()

    tuning_policy = ost.StabilityTune(
        change_factor=change_factor, drop_margin=drop_margin)


    strategy = build_tuning_strategy(
        TUNERS[builder.get_label()], builder, reference, tuning_policy, ctx.obj.tb_logdir)

    if tuned_lr:
        assert builder.get_label() == "naive", "Learning rate can only be set for learning rate decay"
        strategy.tuned_init_lr = float(tuned_lr)

    pretty_print_strategy({
        "underlying": builder.get_hyper_parameters(),
        "reference": reference.get_hyper_parameters(),
        "change_factor": change_factor,
        "drop_margin": drop_margin}, strategy.label)

    ctx.obj.runner.run(strategy)


@ewc.command()
@click.option("-p", "--p-gain", required=True,
              help="Proporitonal gain factor: how much should stability be "
                   "changed to counteract error")
@click.option("-i", "--i-gain", default=0.0, show_default=True,
              help="Integral gain factor: how much should stability be changed "
                   "to counteract sustained error")
@click.option("-d", "--d-gain", default=0.0, show_default=True,
              help="Derivative gain factor: how much should stability be "
                   "changed to counteract an increasing error")
@click.option("-sp", "--setpoint", default=0.1, show_default=True,
              help="A target level of intransigence to maintain")
@click.pass_context
def cybernetic_tune(ctx, p_gain, i_gain, d_gain, setpoint):
    """
    Grow and decay stability online

    Uses my Cybernetic Online Stability Tuning algorithm (COST). We found changin
    proportional gain was sufficient. but the other gain factors are supplied
    for completeness.

    I recommend the following parameters (by no means the best):
        - EWC --p_gain=5000
        - SI  --p_gain=1000
        - LWF --p_gain=1.0
    """
    builder = ctx.obj.builder
    assert isinstance(builder, StrategyBuilder)

    reference = builder.copy().use_strategy(strategies.Naive).use_label("ref") \
                       .without_logger()

    builder.use_strategy(
            COST_TUNERS[builder.get_label()],
            reference_strategy=reference.build(),
            **builder._strategy_kwargs,
            cost_params=dict(kp=float(p_gain), ki=float(i_gain), kd=float(d_gain), setpoint=setpoint)
        ).use_label(f"cost_{builder.get_label()}")

    strategy = builder.build()
    pretty_print_strategy(builder.get_hyper_parameters(), strategy.label)
    ctx.obj.runner.run(strategy)


si.add_command(tune)
si.add_command(decay)
si.add_command(cybernetic_tune)
si.add_command(constant)
lwf.add_command(tune)
lwf.add_command(decay)
lwf.add_command(cybernetic_tune)
lwf.add_command(constant)
ewc.add_command(constant)
naive.add_command(cybernetic_tune)
naive.add_command(decay)
naive.add_command(tune)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scenario(obj=MyContext())



from typing import Sequence

import click
import experiment_cli as cli

LOGDIR = "tb_data"

# Constants
DECAY_FACTOR  = 0.8
CHANGE_FACTOR = 0.2
MNIST_LR      = 0.001
CIFAR_LR      = 0.005
CORE_LR       = 0.001

# Grid Search Definition
EWC_LAMBDAS = [2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000]
SI_LAMBDAS  = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000]
LWF_LAMBDAS = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]

EWC_P_GAINS     =  [5_000, 10_000] 
SI_P_GAINS      =  [2_000, 2_000]
LWF_P_GAINS     =  [1, 2]
COST_LR_P_GAINS =  [-1e-3, -1e-4, -1e-5]

DROP_MARGINS = [0.1, 0.2, 0.3, 0.4]
SET_POINTS   = [0.1, 0.2, 0.3, 0.4]

def mnist_multirun():
    """
    Run MNIST experiments. Parameters are picked from the best results in grid search
    """
    batch = [
        # Naive
        "naive constant",

        # Hold stability constant
        "ewc --ewc-lambda 128000 constant",
        "si  --si-lambda   64000 constant",
        "lwf --lwf-lambda     16 constant",

        # Cybernetic Online Stability Tuning
        "ewc --ewc-lambda 0 cybernetic-tune -p 5000 -sp 0.3",
        "si  --si-lambda  0 cybernetic-tune -p 1000 -sp 0.3",
        "lwf --lwf-lambda 0 cybernetic-tune -p 1     -sp 0.4",

        # Decay
        f"ewc --ewc-lambda 500000 decay -df {DECAY_FACTOR} -dm 0.4",
        f"si   --si-lambda 100000 decay -df {DECAY_FACTOR} -dm 0.3",
        f"lwf --lwf-lambda 50     decay -df {DECAY_FACTOR} -dm 0.4",

        # Tune
        f"ewc --ewc-lambda 10000 tune -cf {CHANGE_FACTOR} -dm 0.3",
        f"si   --si-lambda 10000 tune -cf {CHANGE_FACTOR} -dm 0.4",
        f"lwf --lwf-lambda 10    tune -cf {CHANGE_FACTOR} -dm 0.4",

        # COST-LR
        "naive cybernetic-tune -p -1e-5 -sp 0.4",
    ]

    for i in range(10):
        run_batch(f"--lr {MNIST_LR} --rand True --tb-logdir {LOGDIR}/mnist mnist", batch, f"Completed mnist {i}/10")

def cifar_multirun():
    """
    Run x10 CIFAR experiments. Parameters are picked from the best results in grid search
    """
    batch = [
        # Naive
        "naive constant",
        # Hold stability constant
        "ewc --ewc-lambda   2000 constant",
        "si  --si-lambda    8000 constant",
        "lwf --lwf-lambda     32 constant",
        # Cybernetic Online Stability Tuning
        "ewc --ewc-lambda 0 cybernetic-tune -p 5000 -sp 0.2",
        "si  --si-lambda 0  cybernetic-tune -p 2000 -sp 0.2",
        "lwf --lwf-lambda 0 cybernetic-tune -p 1    -sp 0.4",
        # Decay
        f"ewc --ewc-lambda 100000 decay -df {DECAY_FACTOR} -dm 0.2",
        f"si   --si-lambda 100000  decay -df {DECAY_FACTOR} -dm 0.4",
        f"lwf --lwf-lambda 50      decay -df {DECAY_FACTOR} -dm 0.4",
        # Tune
        f"ewc --ewc-lambda 10000 tune -cf {CHANGE_FACTOR} -dm 0.2",
        f"si   --si-lambda 10000 tune -cf {CHANGE_FACTOR} -dm 0.2",
        f"lwf --lwf-lambda 10    tune -cf {CHANGE_FACTOR} -dm 0.4",

        # COST-LR
        "naive cybernetic-tune -p -1e-4 -sp 0.1",
    ]

    for i in range(10):
        run_batch(f"--lr {CIFAR_LR} --rand True --tb-logdir {LOGDIR}/cifar cifar", batch, f"Completed CIFAR {i}/10")

def core_multirun():
    """
    Run CORE experiments. Parameters are picked from the best results in grid search
    """
    batch = [
        # Naive
        "naive constant",
        # Hold stability constant
        "ewc --ewc-lambda  16000 constant",
        "si  --si-lambda   16000 constant",
        "lwf --lwf-lambda     32 constant",
        # Cybernetic Online Stability Tuning
        "ewc --ewc-lambda 0 cybernetic-tune -p 5000 -sp 0.1",
        "si  --si-lambda 0  cybernetic-tune -p 1000 -sp 0.1",
        "lwf --lwf-lambda 0 cybernetic-tune -p 2     -sp 0.2",
        # Decay
        f"ewc --ewc-lambda 100000 decay -df {DECAY_FACTOR} -dm 0.3",
        f"si   --si-lambda 100000  decay -df {DECAY_FACTOR} -dm 0.2",
        f"lwf --lwf-lambda 50      decay -df {DECAY_FACTOR} -dm 0.4",
        # Tune
        f"ewc --ewc-lambda 10000 tune -cf {CHANGE_FACTOR} -dm 0.1",
        f"si   --si-lambda 10000 tune -cf {CHANGE_FACTOR} -dm 0.2",
        f"lwf --lwf-lambda 10    tune -cf {CHANGE_FACTOR} -dm 0.2",
        # COST-LR
        "naive cybernetic-tune -p -1e-5 -sp 0.4",
    ]

    for i in range(10):
        run_batch(f"--lr {CORE_LR} --core-run {i} --rand True --tb-logdir {LOGDIR}/core core50", batch, f"Completed CORE {i}/10")


def parameter_search():
    """
    Run grid-search
    """

    naive = []
    for ewc_lambda in EWC_LAMBDAS:
        naive.append(f"ewc --ewc-lambda {ewc_lambda} constant")
    for si_lambda in SI_LAMBDAS:
        naive.append(f"si --si-lambda {si_lambda} constant")
    for lwf_lambda in LWF_LAMBDAS:
        naive.append(f"lwf --lwf-lambda {lwf_lambda} constant")

    cost = []
    for sp in SET_POINTS:
        for p_gain in EWC_P_GAINS:
            cost.append(f"ewc --ewc-lambda 0 cybernetic-tune -p {p_gain} -sp {sp}")
        for p_gain in LWF_P_GAINS:
            cost.append(f"lwf --lwf-lambda 0 cybernetic-tune -p {p_gain} -sp {sp}")
        for p_gain in SI_P_GAINS:
            cost.append(f"si  --si-lambda  0 cybernetic-tune -p  {p_gain} -sp {sp}")
        for p_gain in COST_LR_P_GAINS:
            cost.append(f"naive cybernetic-tune -p {p_gain} -sp {sp}")

    decay = []
    for dm in DROP_MARGINS:
        decay.append(f"ewc --ewc-lambda 500000  decay -df {DECAY_FACTOR} -dm {dm}")
        decay.append(f"si   --si-lambda 100000  decay -df {DECAY_FACTOR} -dm {dm}")
        decay.append(f"lwf --lwf-lambda 50      decay -df {DECAY_FACTOR} -dm {dm}")

    tune = []
    for dm in DROP_MARGINS:
        tune.append(f"ewc --ewc-lambda 10000  tune -cf {CHANGE_FACTOR} -dm {dm}")
        tune.append(f"si  --si-lambda  10000  tune -cf {CHANGE_FACTOR} -dm {dm}")
        tune.append(f"lwf --lwf-lambda 10      tune -cf {CHANGE_FACTOR} -dm {dm}")

    # NAIVE
    run_batch(f"--lr {MNIST_LR} --tb-logdir {LOGDIR}/mnist/naive mnist", naive, "MNIST NAIVE GRID")
    run_batch(f"--lr {CORE_LR}  --tb-logdir {LOGDIR}/core/naive core50", naive, "CORE NAIVE GRID")
    run_batch(f"--lr {CIFAR_LR} --tb-logdir {LOGDIR}/cifar/naive cifar", naive, "CIFAR NAIVE GRID")

    # COST Experiments
    run_batch(f"--lr {MNIST_LR} --tb-logdir {LOGDIR}/mnist/cost mnist", cost, "MNIST COST GRID")
    run_batch(f"--lr {CORE_LR}  --tb-logdir {LOGDIR}/core/cost core50", cost, "CORE  COST GRID")
    run_batch(f"--lr {CIFAR_LR} --tb-logdir {LOGDIR}/cifar/cost cifar", cost, "CIFAR COST GRID")

    # Decay Experiments
    run_batch(f"--lr {MNIST_LR} --tb-logdir {LOGDIR}/mnist/decay mnist", decay, "MNIST DECAY GRID")
    run_batch(f"--lr {CORE_LR}  --tb-logdir {LOGDIR}/core/decay core50", decay, "CORE  DECAY GRID")
    run_batch(f"--lr {CIFAR_LR} --tb-logdir {LOGDIR}/cifar/decay cifar", decay, "CIFAR  DECAY GRID")

    # Tune Experiments
    run_batch(f"--lr {MNIST_LR} --tb-logdir {LOGDIR}/mnist/tune mnist", tune, "MNIST TUNE GRID")
    run_batch(f"--lr {CORE_LR}  --tb-logdir {LOGDIR}/core/tune core50", tune, "CORE  TUNE GRID")
    run_batch(f"--lr {CIFAR_LR} --tb-logdir {LOGDIR}/cifar/tune cifar", tune, "CIFAR TUNE GRID")

def run_batch(base_cmd: str, commands: Sequence[str], msg:str=""):
    """Run a batch of commands"""
    for cmd in commands:
        run(base_cmd + " " + cmd)

def run(command: str):
    """Run a cli command"""
    print(f"\n{command}\n")
    cli.scenario(command.split(), obj=cli.MyContext(), standalone_mode=False)

@click.command()
@click.option('--grid-search', default=False)
@click.option('--mnist', default=False)
@click.option('--core', default=False)
@click.option('--cifar', default=False)
@click.option('--logdir', required=True)
def main(logdir: str, grid_search: bool, mnist: bool, cifar: bool, core: bool):
    global LOGDIR 
    LOGDIR = logdir
    if grid_search:
        parameter_search()
    if mnist:
        mnist_multirun()
    if core:
        core_multirun()
    if cifar:
        cifar_multirun()
    if not any([grid_search, mnist, cifar, core]):
        raise click.exceptions.UsageError("Nothing todo at least one option should be true")


if __name__ == '__main__':
    main()
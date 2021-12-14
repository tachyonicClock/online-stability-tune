## Install
1. Anaconda or Miniconda (Instructions: https://www.anaconda.com/)
2. Avalanche
 (**Requires specific version**)

    1. Download avalanche code
        ```sh
        git clone https://github.com/ContinualAI/avalanche.git
        cd avalanche
        ```
    2. Use specific version
        ```sh
        git checkout 658228d8b01df6f8eb0b18e148cce827028b483c 
        ```
    3. Create and install environment
        ```sh
        conda create -n avalanche-env python=3.8 -c conda-forge
        conda activate avalanche-env
        ```
    4. Install pytorch (instructions: https://pytorch.org/)
    5. Install dependencies
        ```
        conda env update --file environment.yml
        ```

> (Optional) **CORe50** by default has a very large test-set we sub-sampled this for speed. This can be conducted by running the following on each `test_filelist.txt` in `batches_filelists/NC_inc/run*`.
> ```sh
> cp test_filelist.txt test_filelist.txt.old && awk 'NR%20==0' test_filelist.> txt.old > test_filelist.txt
> ```


## Usage

- Run naive fine tuning on the mnist dataset
    ```
    ./experiment_cli.py mnist naive constant
    ```

- Run and output tensorboard logs
    ```
    ./experiment_cli.py --tb-logdir tb_data/mnist mnist naive constant
    ```
- Run with a regularization strategy.
    ```
    ./experiment_cli.py mnist ewc --ewc-lambda 128000 control
    ```
- Run with tuned regularization strategy
    - **Semi-Online Stability Decay** - Decrease stability when it is too high
        ```
        ./experiment_cli.py mnist \
            ewc   --ewc-lambda  500000 \
            decay --drop-margin 0.4 --decay-factor 0.8
        ```
    - **Semi-Online Stability Tune** - Increase and decrease stability to find a *"goldilocks"* zone
        ```
        ./experiment_cli.py mnist \
            ewc   --ewc-lambda 10000 \
            tune  --drop-margin 0.3 --change-factor 0.2
        ```
    - **Cybernetic Online Stability Tuning** - Continuously increase and decrease stability to find a *goldilocks* zone fully online.
        ```
        ./experiment_cli.py mnist \
            ewc              --ewc-lambda 0 \
            cybernetic-tune  --p-gain 1000 --set-point 0.3
        ```
    > **Any tuning algorithm works with any underlying regularization strategy**
- Run COST-LR.
    ```
    ./experiment_cli.py mnist \
        naive cybernetic-tune --p-gain -1e-4 --set-point 0.3
    ```
    > Negative p-gain is important since lower learning rates are more stable

- Randomize experience
    ```
    ./experiment_cli.py --rand True mnist naive constant
    ./experiment_cli.py --rand True cifar naive constant
    ./experiment_cli.py --core-run $(($RANDOM % 10)) --rand True core50  naive constant
    ```
    > CORe50 has pre-defined random runs 0-9

## Reproduce Experiments

### Grid Search
To run our grid search.
```
python ./run_experiments.py --logdir tb_data --grid-search True
```
**Parameters Held Constant**
|       | Learning Rate | Batch Size | Epochs |
|-------|---------------|------------|--------|
| MNIST |     0.001     |     64     |    1   |
| CORE  |     0.001     |      ^     |    ^   |
| CIFAR |     0.005     |      ^     |    ^   |

**Constant Stability Grid**
| Strategy  | Stability |
|---|:---:|
| EWC | 2k, 4k, 8k, 16k, 32k, 64k, 128k, 256k | 
| SI | 1k, 2k, 4k, 8k, 16k, 32k, 64k | 
| LWF | 0.5, 1, 2, 4, 8, 16, 32, 64, 128 |

**Semi-Online Stability Decay Grid (aka Decay/OSD)**
| Strategy | Initial Stability | Decay Factor | Drop Margin |
|---|:---:|:---:|:---:|
| EWC | 100k (500k for MNIST) | 0.8 | 0.1, 0.2, 0.3, 0.4 |
| SI | 100000 | ^  | ^ |
| LWF | 50 | ^ | ^ |

**Semi-Online Stability Tune Grid (aka Tune/OST)**

| Strategy | Initial Stability | Change Factor | Drop Margin |
|---|:---:|:---:|:---:|
| EWC | 10000 | 0.2 | 0.1, 0.2, 0.3, 0.4 |
| SI | 10000 | ^ | ^ |
| LWF | 10 | ^ | ^ |

**Cybernetic Online Stability Tuning (aka COST)**
|  | p-gain | setpoints |
|---|:---:|:---:|
| EWC | 5000, 10000 | 0.1, 0.2, 0.3, 0.4 |
| SI | 1000, 2000 | ^ |
| LWF | 1, 2 | ^ |
| COST | -1e-3, -1e-4, -1e-5 | ^ |

### Run 10x runs
To run our 10 runs with our parameters
```
python ./run_experiments.py --logdir tb_data --mnist True --cifar True --core True
```
## Learning Exploration Strategies for Model Agnostic Meta-Reinforcement Learning

Switch to branch another_sparse_branch_ppo to run the experiments on our model. The code in master is used to run experiments on the vanilla maml-trpo model.

## Getting started
To avoid any conflict with your existing Python setup, and to keep this project self-contained, it is suggested to work in a virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/). To install `virtualenv`:
```
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
You can use the [`main.py`](main.py) script in order to run reinforcement learning experiments. This script was tested with Python 3.5. 
```
python main.py --env-name HalfCheetahRandVelEnv-v1 --fast-batch-size 20 --meta-batch-size 40 --output-folder hcv-1 --num-workers 16 --embed-size 32  --exp-lr 7e-4 --baseline-type nn --nonlinearity tanh --num-layers-pre 1 --hidden-size 64 --seed 0
```

## References
A huge part of this implementation is borrowed from the MAML implementation of [tristandeleu/pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl) in Pytorch. A huge thanks to them for open-sourcing their implementation.

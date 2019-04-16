import argparse
import multiprocessing as mp


def get_args():
    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')

    # General
    parser.add_argument('--env-name', type=str,
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.95,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--embed-size', type=int, default=4,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers-pre', type=int, default=2,
        help='number of hidden layers-pre')
    parser.add_argument('--num-layers-post', type=int, default=1,
        help='number of hidden layers-post')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=1,
        help='batch size for each individual task')
    parser.add_argument('--fast-lr', type=float, default=0.5,
        help='learning rate for the 1-step gradient update of MAML')
    parser.add_argument('--exp-lr', type=float, default=7e-4,
        help='learning rate for exploration network')
    parser.add_argument('--exp-eps', type=float, default=1e-5,
        help='epsilon for exploration network optimizer')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=200,
        help='number of batches')
    parser.add_argument('--meta-batch-size', type=int, default=40,
        help='number of tasks per batch')
    parser.add_argument('--max-kl', type=float, default=1e-2,
        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-5,
        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=15,
        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
        help='maximum number of iterations for line search')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--load-dir', type=str, default=None,
        help='name of the directory to load model')
    parser.add_argument('--num-plots', type=int, default=1,
        help='number of plots to save per iteration')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cuda',
        help='device type')
    parser.add_argument('--num-updates', type=int, default=1,
        help='number of gradient steps to be taken')
    
    # logging
    parser.add_argument('--logdir', default='logs')
    parser.add_argument('--savedir', default='saves')
    parser.add_argument('--save-every', default=20, type=int)

    args = parser.parse_args()

    return args
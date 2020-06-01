import argparse
import ipdb

def get_args():
    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')

    # General
    parser.add_argument('--env-name', type=str,
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--embed-size', type=int, default=32,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers-pre', type=int, default=2,
        help='number of hidden layers-pre')
    parser.add_argument('--num-layers-post', type=int, default=1,
        help='number of hidden layers-post')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=20,
        help='batch size for each individual task')
    parser.add_argument('--fast-lr', type=float, default=0.1,
        help='learning rate for the 1-step gradient update of MAML')
    parser.add_argument('--exp-lr', type=float, default=7e-4,
        help='learning rate for exploration network')
    parser.add_argument('--exp-eps', type=float, default=1e-5,
        help='epsilon for exploration network optimizer')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=1000,
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
    parser.add_argument('--num-workers', type=int, default=16,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cuda',
        help='device type')
    parser.add_argument('--num-updates', type=int, default=1,
        help='number of gradient steps to be taken')
    parser.add_argument('--num-updates-outer', type=int, default=5, help='number outer loop updates')
    # logging
    parser.add_argument('--logdir', default='logs')
    parser.add_argument('--savedir', default='saves')
    parser.add_argument('--save-every', default=20, type=int)
    parser.add_argument('--test', action='store_true', help='do not log anything')
    parser.add_argument('--dontplot', action='store_true', help='dont plot anything')
    parser.add_argument('--algo', default='ppo', help='{ppo, a2c, trpo}')
    parser.add_argument('--emaml', action='store_true', help='run emaml')
    parser.add_argument('--n-exp', default=5, type=int, help='number of exploration traj to plot')
    parser.add_argument('--baseline-type', default='lin', type=str, help='Exploration baseline : {lin, nn}')
    parser.add_argument('--reward-net-type', default='input_latent', type=str, help='{input_latent, output_latent}')
    parser.add_argument('--nonlinearity', default='relu', type=str, help='{relu, tanh, sigmoid}')
    parser.add_argument('--seed', default=0, type=int, help='seed for numpy and torch')
    parser.add_argument('--M-type', default='returns', type=str, help='{rewards, returns, next-state}')
    parser.add_argument('--separate-actions', action='store_true', help='use a separate branch in reward net to read actions')
    args = parser.parse_args()
    args.n_exp = min(args.n_exp, args.fast_batch_size)
    args.num_plots = min(args.num_plots, args.meta_batch_size)
    return args
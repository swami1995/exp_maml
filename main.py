import maml_rl.envs
import gym
import numpy as np
import torch
import json
import ipdb
import os
import sys
import matplotlib.pyplot as plt

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy, RewardNetMLP
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler
from maml_rl.envs.point_envs.point_env_2d_corner import MetaPointEnvCorner

from tensorboardX import SummaryWriter

def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

def plotting(episodes, batch, save_folder,n):
    for i in range(n):
        train_ep = episodes[i][0]
        val_ep = episodes[i][1]
        task = episodes[i][0].task
        train_obs = train_ep.observations[:,0].cpu().numpy()
        val_obs = val_ep.observations[:,0].cpu().numpy()
        corners = np.array([np.array([-2,-2]), np.array([2,-2]), np.array([-2,2]), np.array([2, 2])])
        plt.plot(train_obs[:,0], train_obs[:,1], 'b', label='exploring agent')
        plt.plot(val_obs[:,0], val_obs[:,1], 'k', label='trained agent')
        plt.scatter(corners[:,0], corners[:,1], s=10, color='g')
        plt.scatter(task[None,0], task[None,1], s=10, color='r')
        plt.savefig(os.path.join(save_folder,'plot-{0}-{1}.png'.format(batch,i)))
        plt.clf()
    return None


def main(args):
    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
        '2DNavigation-v0', '2DPointEnvCorner-v0'])

    save_folder = './saves/{0}'.format(args.env_name+'/'+args.output_folder)
    if args.output_folder!='maml-trial' and args.output_folder!='trial':
        # i=0
        # while os.path.exists(save_folder):
        #     args.output_folder=str(i+1)
        #     i+=1
        #     save_folder = './saves/{0}'.format(args.env_name+'/'+args.output_folder)
        log_directory = './logs/{0}'.format(args.env_name+'/'+args.output_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    writer = SummaryWriter('./logs/{0}'.format(args.env_name+'/'+args.output_folder))

    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)
    os.system("mkdir "+save_folder+'/code')
    os.system("cp -r *.py "+save_folder+'/code/')
    os.system("cp -r maml_rl "+save_folder+'/code/')
    print("Models saved in :", save_folder)

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
        num_workers=args.num_workers)
    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            args.embed_size,
            hidden_sizes_pre_embedding=(args.hidden_size,) * args.num_layers_pre,
            hidden_sizes_post_embedding=(args.hidden_size,) * args.num_layers_post)
        exp_policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            args.embed_size,
            hidden_sizes_pre_embedding=(args.hidden_size,) * args.num_layers_pre,
            hidden_sizes_post_embedding=(args.hidden_size,) * args.num_layers_post)
    else:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            args.embed_size,
            hidden_sizes_pre_embedding=(args.hidden_size,) * args.num_layers_pre,
            hidden_sizes_post_embedding=(args.hidden_size,) * args.num_layers_post)
        exp_policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            args.embed_size,
            hidden_sizes_pre_embedding=(args.hidden_size,) * args.num_layers_pre,
            hidden_sizes_post_embedding=(args.hidden_size,) * args.num_layers_post)
    reward_net = RewardNetMLP(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.shape[0],
            args.embed_size,
            hidden_sizes_pre_embedding=(args.hidden_size,) * args.num_layers_pre,
            hidden_sizes_post_embedding=(args.hidden_size,) * args.num_layers_post)

    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape)))

    exp_baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape)))

    if args.load_dir is not None:
        folder_idx = args.load_dir.split('.')
        load_folder = './saves/{0}'.format(args.env_name+'/'+folder_idx[0])
        policy.load_state_dict(torch.load(load_folder+'policy-{0}.pt'.format(folder_idx[1])))
        exp_policy.load_state_dict(torch.load(load_folder+'policy-{0}-exp.pt'.format(folder_idx[1])))
        reward_net.load_state_dict(torch.load(load_folder+'reward-{0}.pt'.format(folder_idx[1])))

    metalearner = MetaLearner(sampler, policy, exp_policy, baseline, exp_baseline, reward_net, embed_size=args.embed_size, gamma=args.gamma,
        fast_lr=args.fast_lr, tau=args.tau, lr=args.exp_lr, eps=args.exp_eps, device=args.device)

    best_reward_after=-400
    for batch in range(args.num_batches):
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        # if batch==0:
        episodes = metalearner.sample(tasks, first_order=args.first_order)
        r_loss, pg_loss = metalearner.step(episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
            cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
            ls_backtrack_ratio=args.ls_backtrack_ratio)

        reward_after = total_rewards([ep.rewards for _, ep in episodes])
        print('total_rewards/before_update', total_rewards([ep.rewards for ep, _ in episodes]), batch)
        print('total_rewards/after_update', total_rewards([ep.rewards for _, ep in episodes]), batch)
        print('reward_loss/before_update', r_loss[0], batch)
        print('reward_loss/after_update', r_loss[1], batch)
        print('pg_loss/after_update', pg_loss, batch)

        if args.load_dir is not None:
            sys.exit(0)
            
        # Tensorboard
        writer.add_scalar('total_rewards/before_update',
            total_rewards([ep.rewards for ep, _ in episodes]), batch)
        writer.add_scalar('total_rewards/after_update',
            total_rewards([ep.rewards for _, ep in episodes]), batch)
        writer.add_scalar('reward_loss/before_update', r_loss[0], batch)
        writer.add_scalar('reward_loss/after_update', r_loss[1], batch)
        writer.add_scalar('pg_loss/after_update', pg_loss, batch)

        ## Save policy network
        if batch%20==0 or reward_after > best_reward_after:
            with open(os.path.join(save_folder,
                    'policy-{0}.pt'.format(batch)), 'wb') as f:
                torch.save(policy.state_dict(), f)
            with open(os.path.join(save_folder,
                    'policy-{0}-exp.pt'.format(batch)), 'wb') as f:
                torch.save(exp_policy.state_dict(), f)
            with open(os.path.join(save_folder,
                    'reward-{0}.pt'.format(batch)), 'wb') as f:
                torch.save(reward_net.state_dict(), f)
            best_reward_after = reward_after
            # Plotting figure
            plotting(episodes, batch, save_folder,args.num_plots)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

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
    parser.add_argument('--embed-size', type=int, default=100,
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
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda)')

    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)

# import maml_rl.envs
# import gym
import numpy as np
import torch
import json
import os
import sys

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy, RewardNetMLP
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler
from maml_rl.envs.point_envs.point_env_2d_corner import MetaPointEnvCorner

from tensorboardX import SummaryWriter
from arguments import get_args
from utils import plotting


def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()


def main(args):
    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
        '2DNavigation-v0', '2DPointEnvCorner-v0'])

    # do not save anything
    if not args.test:
        save_folder = os.path.join(args.savedir, args.env_name, args.output_folder)
        if args.output_folder!='maml-trial' and args.output_folder!='trial':
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
        logdir = os.path.join(args.logdir, args.env_name, args.output_folder)
        writer = SummaryWriter(logdir)

        with open(os.path.join(save_folder, 'config.json'), 'w') as f:
            config = {k: v for (k, v) in vars(args).items() if k != 'device'}
            config.update(device=args.device.type)
            json.dump(config, f, indent=2)

        # save script
        os.system("mkdir "+save_folder+'/code')
        os.system("cp -r *.py "+save_folder+'/code/')
        os.system("cp -r maml_rl "+save_folder+'/code/')
        print("Models saved in :", save_folder)

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size, num_workers=args.num_workers)
    
    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            args.embed_size,
            hidden_sizes_pre_embedding=(args.hidden_size,) * args.num_layers_pre,
            hidden_sizes_post_embedding=(args.hidden_size,) * args.num_layers_post)
        # exploration policy
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
        # exploration policy
        exp_policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            args.embed_size,
            hidden_sizes_pre_embedding=(args.hidden_size,) * args.num_layers_pre,
            hidden_sizes_post_embedding=(args.hidden_size,) * args.num_layers_post)
        
    # reward prediction network
    reward_net = RewardNetMLP(
        int(np.prod(sampler.envs.observation_space.shape)),
        sampler.envs.action_space.shape[0],
        args.embed_size,
        hidden_sizes_pre_embedding=(args.hidden_size,) * args.num_layers_pre,
        hidden_sizes_post_embedding=(args.hidden_size,) * args.num_layers_post)
    reward_net_outer = RewardNetMLP(
        int(np.prod(sampler.envs.observation_space.shape)),
        sampler.envs.action_space.shape[0],
        args.embed_size,
        hidden_sizes_pre_embedding=(args.hidden_size,) * args.num_layers_pre,
        hidden_sizes_post_embedding=(args.hidden_size,) * args.num_layers_post)

    baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))
    exp_baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))

    metalearner = MetaLearner(sampler, policy, exp_policy, baseline, exp_baseline, reward_net, reward_net_outer, 
                              embed_size=args.embed_size, gamma=args.gamma, fast_lr=args.fast_lr, tau=args.tau,
                              lr=args.exp_lr, eps=args.exp_eps, device=args.device)

    if args.load_dir is not None:
        folder_idx = args.load_dir.split('.')
        load_folder = './saves/{0}'.format(args.env_name+'/'+folder_idx[0])
        policy.load_state_dict(torch.load(load_folder+'policy-{0}.pt'.format(folder_idx[1])))
        exp_policy.load_state_dict(torch.load(load_folder+'policy-{0}-exp.pt'.format(folder_idx[1])))
        reward_net.load_state_dict(torch.load(load_folder+'reward-{0}.pt'.format(folder_idx[1])))
        metalearner.z_old.copy_(torch.load(load_folder+'z_old-{0}.pt'.format(folder_idx[1]))['z_old'])
        reward_net_outer.load_state_dict(torch.load(load_folder+'reward_outer-{0}.pt'.format(folder_idx[1])))

    best_reward_after = -400
    for batch in range(args.num_batches):
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        episodes = metalearner.sample(tasks, first_order=args.first_order)
        r_loss, pg_loss, grad_vals = metalearner.step(episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
                                                      cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
                                                      ls_backtrack_ratio=args.ls_backtrack_ratio)

        reward_after = total_rewards([ep.rewards for _, ep in episodes])
        print('Batch {:d}/{:d}'.format(batch+1, args.num_batches))
        rew_before_update = total_rewards([ep.rewards for ep, _ in episodes])
        rew_after_update = total_rewards([ep.rewards for _, ep in episodes])
        rew_loss_before = r_loss[0].item()
        rew_loss_after = r_loss[1].item()
        print('Total Rewards \t Before update {:4f} \t After Update {:.4f}'.format(rew_before_update, rew_after_update))
        print('Reward Loss \t Before update {:4f} \t After Update {:.4f}'.format(rew_loss_before, rew_loss_after))
        print('PG Loss After Update\n'.format(pg_loss.item()))

        if args.load_dir is not None:
            sys.exit(0)
            
        # Tensorboard
        if not args.test:
            for i in range(args.num_updates):
                writer.add_scalar('total_rewards/before_update_'+str(i), rew_before_update, batch)
                writer.add_scalar('reward_loss/before_update_'+str(i), rew_loss_before, batch)
            writer.add_scalar('total_rewards/after_update', rew_after_update, batch)
            writer.add_scalar('reward_loss/after_update', rew_loss_after, batch)
            writer.add_scalar('pg_loss/after_update', pg_loss, batch)
            writer.add_scalar('grad_vals/z', grad_vals[0], batch)  
            writer.add_scalar('grad_vals/policy', grad_vals[1], batch)  
            writer.add_scalar('grad_vals/exp_policy', grad_vals[2], batch)  
            writer.add_scalar('grad_vals/reward_net', grad_vals[3], batch)  
            writer.add_scalar('grad_vals/reward_net_outer', grad_vals[4], batch)

            # Save policy network
            if batch%args.save_every==0 or reward_after > best_reward_after:
                with open(os.path.join(save_folder, 'policy-{0}.pt'.format(batch)), 'wb') as f:
                    torch.save(policy.state_dict(), f)
                with open(os.path.join(save_folder, 'policy-{0}-exp.pt'.format(batch)), 'wb') as f:
                    torch.save(exp_policy.state_dict(), f)
                with open(os.path.join(save_folder, 'reward-{0}.pt'.format(batch)), 'wb') as f:
                    torch.save(reward_net.state_dict(), f)
                with open(os.path.join(save_folder, 'reward_outer-{0}.pt'.format(batch)), 'wb') as f:
                    torch.save(reward_net_outer.state_dict(), f)
                with open(os.path.join(save_folder, 'z_old-{0}.pt'.format(batch)), 'wb') as f:
                    torch.save({'z_old':metalearner.z_old}, f)

                best_reward_after = reward_after
                # Plotting figure
                if args.env_name in ['2DNavigation-v0', '2DPointEnvCorner-v0']:
                    plotting(episodes, batch, save_folder, args.num_plots)


if __name__ == '__main__':
    args = get_args()

    # Create logs and saves folder if they don't exist
    if not args.test:
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)

    # Device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)

import numpy as np
import torch
import torch.nn.functional as F
import json
import os
import sys
import matplotlib.pyplot as plt
import shutil 
import datetime
import ipdb

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy, RewardNetMLP, ValueNetMLP, RewardNetMLP_shared
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler
from maml_rl.envs.point_envs.point_env_2d_corner import MetaPointEnvCorner
from maml_rl.utils.documentation import modify_excel

from tensorboardX import SummaryWriter
from arguments import get_args


def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()


def plotting(episodes, batch, save_folder, n, n_exp):
    for i in range(n):
        train_ep = [episodes[i][0]]
        val_ep = episodes[i][1]
        task = episodes[i][0].task
        val_obs = val_ep.observations[:,0].cpu().numpy()
        corners = np.array([np.array([-2,-2]), np.array([2,-2]), np.array([-2,2]), np.array([2, 2])])
        # cmap = cm.get_cmap('PiYG', len(train_ep)+1)
        for j in range(len(train_ep)):
            for k in range(n_exp):
                train_obs = train_ep[j].observations[:,k].cpu().numpy()
                # train_obs = np.maximum(train_obs, -4)
                # train_obs = np.minimum(train_obs, 4)
                plt.plot(train_obs[:,0], train_obs[:,1], label='exploring agent'+str(j)+str(k))
        # val_obs = np.maximum(val_obs, -4)
        # val_obs = np.minimum(val_obs, 4)
        plt.plot(val_obs[:,0], val_obs[:,1], label='trained agent')
        plt.legend()
        plt.scatter(corners[:,0], corners[:,1], s=10, color='g')
        plt.scatter(task[None,0], task[None,1], s=10, color='r')
        plt.savefig(os.path.join(save_folder,'plot-{0}-{1}.png'.format(batch,i)))
        plt.clf()
    return None


def main(args):
    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
        '2DNavigation-v0', '2DPointEnvCorner-v0', '2DPointEnvCorner-v1', '2DPointEnvCustom-v1',
        'AntRandDirecEnv-v1', 'AntRandDirec2DEnv-v1', 'AntRandGoalEnv-v1', 'HalfCheetahRandDirecEnv-v1',
        'HalfCheetahRandVelEnv-v1', 'HumanoidRandDirecEnv-v1', 'HumanoidRandDirec2DEnv-v1', 
        'Walker2DRandDirecEnv-v1', 'Walker2DRandVelEnv-v1'])

    if not args.test:

        save_folder = './saves/{0}'.format(args.env_name+'/'+args.output_folder)
        log_folder = './logs/{0}'.format(args.env_name+'/'+args.output_folder)
        if os.path.exists(save_folder):
            print('Save folder already exists! Enter')
            text = 'c (rename the existing directory with _old and continue)\n' + \
                   's (stop)!\ndel (delete existing dir): \n' + \
                   'i (ignore and overwrite)'
            ch = input(text)
                        
            if ch == 's':
                sys.exit(0)
            elif ch == 'c':
                os.rename(save_folder, save_folder+'_old')
                os.rename(log_folder, log_folder+'_old')
            elif ch == 'del':
                shutil.rmtree(save_folder)
                shutil.rmtree(log_folder)
            elif ch=='i':
                pass
            else:
                raise NotImplementedError('Unknown input')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        writer = SummaryWriter(log_folder)

        comments_text = 'Do you want to add any comments to the logs?\n'
        with open(os.path.join(save_folder, 'config.json'), 'w') as f:
            config = {k: v for (k, v) in vars(args).items() if k != 'device'}
            config['comments'] = input(comments_text)
            config.update(device=args.device.type)
            json.dump(config, f, indent=2)

        modify_excel(config)

        # save script
        os.system("mkdir "+save_folder+'/code')
        os.system("cp -r *.py "+save_folder+'/code/')
        os.system("cp -r maml_rl "+save_folder+'/code/')
        print("Models saved in :", save_folder)

    start = datetime.datetime.now()
    if args.nonlinearity=='relu':
        nonlinearity = F.relu
    elif args.nonlinearity=='tanh':
        nonlinearity = torch.tanh
    elif args.nonlinearity=='sigmoid':
        nonlinearity = F.sigmoid

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size, num_workers=args.num_workers)
    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            args.embed_size,
            hidden_sizes_pre_embedding=(args.hidden_size,) * args.num_layers_pre,
            hidden_sizes_post_embedding=(args.hidden_size,) * args.num_layers_post, 
            nonlinearity=nonlinearity)
        # exploration policy
        exp_policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            args.embed_size,
            hidden_sizes_pre_embedding=(args.hidden_size,) * args.num_layers_pre,
            hidden_sizes_post_embedding=(args.hidden_size,) * args.num_layers_post, 
            nonlinearity=nonlinearity)
    else:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            args.embed_size,
            hidden_sizes_pre_embedding=(args.hidden_size,) * args.num_layers_pre,
            hidden_sizes_post_embedding=(args.hidden_size,) * args.num_layers_post, 
            nonlinearity=nonlinearity)
        # exploration policy
        exp_policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            args.embed_size,
            hidden_sizes_pre_embedding=(args.hidden_size,) * args.num_layers_pre,
            hidden_sizes_post_embedding=(args.hidden_size,) * args.num_layers_post, 
            nonlinearity=nonlinearity)
    
    # reward prediction network
    if args.reward_net_type == 'input_latent':
        reward_net = RewardNetMLP(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.shape[0],
            args.embed_size,
            hidden_sizes_pre_embedding=(args.hidden_size,) * args.num_layers_pre,
            hidden_sizes_post_embedding=(args.hidden_size,) * args.num_layers_post, 
            nonlinearity=nonlinearity)
        reward_net_outer = RewardNetMLP(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.shape[0],
            args.embed_size,
            hidden_sizes_pre_embedding=(args.hidden_size,) * args.num_layers_pre,
            hidden_sizes_post_embedding=(args.hidden_size,) * args.num_layers_post, 
            nonlinearity=nonlinearity)
    elif args.reward_net_type=='output_latent':
        reward_net = RewardNetMLP_shared(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.shape[0],
            args.embed_size,
            hidden_sizes=(args.hidden_size,) * args.num_layers_pre, 
            nonlinearity=nonlinearity)
        reward_net_outer = RewardNetMLP_shared(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.shape[0],
            args.embed_size,
            hidden_sizes=(args.hidden_size,) * args.num_layers_pre, 
            nonlinearity=nonlinearity)
    if args.baseline_type=='nn':
        exp_baseline = ValueNetMLP(int(np.prod(sampler.envs.observation_space.shape)),
            hidden_sizes_pre_embedding=(args.hidden_size,) * args.num_layers_pre, 
            nonlinearity=nonlinearity)
        exp_baseline_targ = ValueNetMLP(int(np.prod(sampler.envs.observation_space.shape)),
            hidden_sizes_pre_embedding=(args.hidden_size,) * args.num_layers_pre, 
            nonlinearity=nonlinearity)
    elif args.baseline_type=='lin':
        exp_baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))
        exp_baseline_targ = exp_baseline

    baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))
    
    if args.load_dir is not None:
        z_old = torch.zeros((1,args.embed_size))
        folder_idx = args.load_dir.split('.')
        start_batch = int(folder_idx[1])
        load_folder = './saves/{0}'.format(args.env_name+'/'+folder_idx[0]+'/')
        policy.load_state_dict(torch.load(load_folder+'policy-{0}.pt'.format(folder_idx[1])))
        exp_policy.load_state_dict(torch.load(load_folder+'policy-{0}-exp.pt'.format(folder_idx[1])))
        reward_net.load_state_dict(torch.load(load_folder+'reward-{0}.pt'.format(folder_idx[1])))
        z_old.copy_(torch.load(load_folder+'z_old-{0}.pt'.format(folder_idx[1]))['z_old'])
        reward_net_outer.load_state_dict(torch.load(load_folder+'reward_outer-{0}.pt'.format(folder_idx[1])))
        moving_params_normalize = torch.tensor(np.load('moving_params_normalize-{0}.npy'.format(folder_idx[1])))
        if args.baseline_type=='nn':
            exp_baseline.load_state_dict(torch.load(load_folder+'value_net-{0}.pt'.format(folder_idx[1])))
            exp_baseline_targ.load_state_dict(torch.load(load_folder+'value_net_targ-{0}.pt'.format(folder_idx[1])))
    else:
        start_batch = 0
        z_old = None
        moving_params_normalize=torch.tensor([0.,1.])
    metalearner = MetaLearner(sampler, policy, exp_policy, baseline, exp_baseline, reward_net, reward_net_outer, exp_baseline_targ, z_old, 
                              args.baseline_type, embed_size=args.embed_size, gamma=args.gamma, fast_lr=args.fast_lr, tau=args.tau, lr=args.exp_lr, 
                              eps=args.exp_eps, device=args.device, algo=args.algo, use_emaml=args.emaml, num_updates_outer=args.num_updates_outer, 
                              moving_params_normalize=moving_params_normalize)
    
    best_reward_after = -40000
    for batch in range(start_batch+1,args.num_batches):
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        episodes = metalearner.sample(tasks, first_order=args.first_order)
        r_loss, pg_loss, grad_vals = metalearner.step(episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
            cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
            ls_backtrack_ratio=args.ls_backtrack_ratio)
        metalearner.update_targs_value()

        before_update_reward = total_rewards([ep.rewards for ep, _ in episodes])
        after_update_reward = total_rewards([ep.rewards for _, ep in episodes])
        end = datetime.datetime.now()

        print('Batch {:d}/{:d}'.format(batch+1, args.num_batches))
        print('Rewards')
        print('Before update {:.3f} After update {:.3f}'.format(before_update_reward, after_update_reward))
        print('Reward loss')
        print('Before update {:.3f} After update {:.3f}'.format(r_loss[0], r_loss[1]))
        print('PG Loss After Update {:.3f}'.format(pg_loss))
        print('Time {}\n'.format(end-start))

        # if args.load_dir is not None:
        #     sys.exit(0)
            
        if not args.test:
            writer.add_scalar('total_rewards/before_update', before_update_reward, batch)
            writer.add_scalar('total_rewards/after_update', after_update_reward, batch)
            writer.add_scalar('reward_loss/before_update', r_loss[0], batch)
            writer.add_scalar('reward_loss/after_update', r_loss[1], batch)
            writer.add_scalar('pg_loss/after_update', pg_loss, batch)
            writer.add_scalar('grad_vals/z', grad_vals[0], batch)
            writer.add_scalar('grad_vals/policy', grad_vals[1], batch)
            writer.add_scalar('grad_vals/exp_policy', grad_vals[2], batch)
            writer.add_scalar('grad_vals/reward_net', grad_vals[3], batch)
            writer.add_scalar('grad_vals/reward_net_outer', grad_vals[4], batch)
            writer.add_scalar('grad_vals/kl_grads', grad_vals[5], batch)

            # Save policy network
            if batch%args.save_every==0 or after_update_reward > best_reward_after:
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
                np.save('moving_params_normalize-{0}.npy'.format(batch), metalearner.moving_params_normalize.cpu().numpy())
                if args.baseline_type=='nn':
                    with open(os.path.join(save_folder, 'value_net-{0}.pt'.format(batch)), 'wb') as f:
                        torch.save(exp_baseline.state_dict(), f)
                    with open(os.path.join(save_folder, 'value_net_targ-{0}.pt'.format(batch)), 'wb') as f:
                        torch.save(exp_baseline_targ.state_dict(), f)
                best_reward_after = after_update_reward
                # Plotting figure
                if args.env_name in ['2DNavigation-v0', '2DPointEnvCorner-v0', '2DPointEnvCorner-v1', '2DPointEnvCustom-v1']:
                    plotting(episodes, batch, save_folder, args.num_plots, args.n_exp)


if __name__ == '__main__':
    args = get_args()

    # Create logs and saves folder if they don't exist
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

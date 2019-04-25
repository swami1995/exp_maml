import numpy as np
import matplotlib.pyplot as plt
import os


def plotting(episodes, batch, save_folder, n):
    for i in range(n):
        train_ep = [episodes[i][0]]
        val_ep = episodes[i][1]
        task = episodes[i][0].task
        val_obs = val_ep.observations[:,0].cpu().numpy()
        corners = np.array([np.array([-2,-2]), np.array([2,-2]), np.array([-2,2]), np.array([2, 2])])
        # cmap = cm.get_cmap('PiYG', len(train_ep)+1)
        for j in range(len(train_ep)):
            for k in range(5):
                train_obs = train_ep[j].observations[:,k].cpu().numpy()
                train_obs = np.maximum(train_obs, -4)
                train_obs = np.minimum(train_obs, 4)
                plt.plot(train_obs[:,0], train_obs[:,1], label='exploring agent'+str(j)+str(k))
        val_obs = np.maximum(val_obs, -4)
        val_obs = np.minimum(val_obs, 4)
        plt.plot(val_obs[:,0], val_obs[:,1], label='trained agent')
        plt.legend()
        plt.scatter(corners[:,0], corners[:,1], s=10, color='g')
        plt.scatter(task[None,0], task[None,1], s=10, color='r')
        plt.savefig(os.path.join(save_folder,'plot-{0}-{1}.png'.format(batch,i)))
        plt.clf()
    return None






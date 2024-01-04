import os
import numpy as np
import matplotlib.pyplot as plt

FIGSIZE = (20, 20)


def plot_tracklet_position(x, z, path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, z, color='blue')
    ax.scatter(x[0], z[0], color='green', label='start')
    ax.scatter(x[-1], z[-1], color='red', label='end')
    ax.scatter(0, 0, color='black', label='ego_car')
    for i in range(len(x)-1):
        dx = x[i+1]-x[i]
        dz = z[i+1]-z[i]
        ax.arrow(x[i], z[i], dx, dz, length_includes_head=True)
    ax.legend()
    ax.grid()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title('position map')
    plt.tight_layout()
    file_name = os.path.join(path, 'position_map.png')
    plt.savefig(file_name)
    plt.close()


def plot_kinematics(age, axis_dict: dict, file_path):
    axes = axis_dict.keys()
    N = len(axes)
    M = len(axis_dict[list(axes)[0]])
    fig, ax = plt.subplots(M, N, figsize=FIGSIZE)
    for i in range(N):
        axis = axis_dict[list(axes)[i]]
        for j, (key, values) in enumerate(axis.items()):
            vec = values['vector']
            units = values['units']
            plot_variable(ax[j, i], age, vec, key, units)

    plt.savefig(file_path)
    plt.close()

def plot_bbox_params(age, params_dict: dict, file_path):
    fig, ax = plt.subplots(2, 1, figsize=FIGSIZE)
    for i,(key, values) in enumerate(params_dict.items()):
        vec = values['vector']
        units = values['units']
        plot_variable(ax[i], age, vec, key, units)

    plt.savefig(file_path)
    plt.close()



def plot_variable(ax, x, y, title, units):
    N = len(x)
    mean, std = np.mean(y), np.std(y)
    ax.plot(x, y, label=title, color='blue')
    ax.plot(x, np.full(N, mean), label='mean', color='green')
    ax.plot(x, np.full(N, 2*std + mean), label=f'2*std',
            color='red', linestyle='dashed')
    ax.plot(x, np.full(N, -2*std + mean), label=f'2*std',
            color='red', linestyle='dashed')
    ax.legend()
    ax.grid()
    ax.set_ylabel(f'[{units}]')

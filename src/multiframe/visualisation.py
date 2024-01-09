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
    ax.plot(x, y,'--bo', label=title)
    ax.plot(x, np.full(N, mean), label='mean', color='green')
    ax.plot(x, np.full(N, 2*std + mean), label=f'2*std',
            color='red', linestyle='dashed')
    ax.plot(x, np.full(N, -2*std + mean), label=f'2*std',
            color='red', linestyle='dashed')
    ax.legend()
    ax.grid()
    ax.set_ylabel(f'[{units}]')

def plot_derivatives(age, x, title, unit, file_path):
    first_derivative = np.diff(x, n=1)
    avg1, std1 = np.mean(first_derivative), np.std(first_derivative)
    second_derivative = np.diff(x, n=2)
    avg2, std2 = np.mean(second_derivative), np.std(second_derivative)
    fig, ax = plt.subplots(3, 1, figsize=FIGSIZE)
    ax[0].plot(age, x, '--bo', label=f'{title}')
    ax[0].set_xlabel('age')
    ax[0].set_ylabel(unit)
    ax[0].legend()
    ax[0].grid()
    ax[0].set_xticks(age)
    
    ax[1].plot(age[1:], first_derivative, '--co', label='first derivative')
    ax[1].plot(age[1:], np.full(len(age[1:]), avg1), label='mean', color='green')
    ax[1].plot(age[1:], np.full(len(age[1:]), 2*std1 + avg1), label=f'2*std',color='red', linestyle='dashed')
    ax[1].plot(age[1:], np.full(len(age[1:]), -2*std1 + avg1), label=f'2*std',color='red', linestyle='dashed')
    ax[1].plot(age[1:], np.full(len(age[1:]), 3*std1 + avg1), label=f'3*std',color='red', linestyle='dashdot')
    ax[1].plot(age[1:], np.full(len(age[1:]), -3*std1 + avg1), label=f'3*std',color='red', linestyle='dashdot')
    ax[1].set_xlabel('age')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_xticks(age)
    
    ax[2].plot(age[2:], second_derivative, '--mo', label=f'second derivative')
    ax[2].plot(age[2:], np.full(len(age[2:]), avg2), label='mean', color='green')
    ax[2].plot(age[2:], np.full(len(age[2:]), 2*std2 + avg2), label=f'2*std',color='red', linestyle='dashed')
    ax[2].plot(age[2:], np.full(len(age[2:]), -2*std2 + avg2), label=f'2*std',color='red', linestyle='dashed')
    ax[2].plot(age[2:], np.full(len(age[2:]), 3*std2 + avg2), label=f'3*std',color='red', linestyle='dashdot')
    ax[2].plot(age[2:], np.full(len(age[2:]), -3*std2 + avg2), label=f'3*std',color='red', linestyle='dashdot')
    ax[2].set_xlabel('age')
    ax[2].legend()
    ax[2].grid()
    ax[2].set_xticks(age)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
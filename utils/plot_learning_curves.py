"""
Run an evaluation script on the saved models to get average performance
"""

import os
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch

sns.set(style="darkgrid")
sns.set_context("paper")

cols_deep = sns.color_palette("deep", 10)
cols_bright = sns.color_palette("bright", 10)
cols_dark = sns.color_palette("dark", 10)


def moving_average(array, num_points, only_past=False):
    if not only_past:
        ma = []
        for i in range(num_points, array.shape[0]):
            ma.append(np.mean(array[i - num_points:i]))
        ma = np.array(ma)
    else:
        ma = []
        for i in range(1, array.shape[0] + 1):
            ma.append(np.mean(array[max(0, i - num_points):i]))
        ma = np.array(ma)

    return ma


def get_array_from_event(event_path, tag, m):
    arr = []
    steps = []
    try:
        for event in summary_iterator(event_path):
            if hasattr(event.summary, 'value') and len(event.summary.value) > 0:
                if event.summary.value[0].tag == tag:
                    arr.append(event.summary.value[0].simple_value)
                    steps.append(event.step)
    except:
        pass

    steps = np.array(steps)
    arr = moving_average(np.array(arr), m, only_past=True)
    return arr, steps


def get_array_from_event_multi_episode(event_path, tag, rollout_indices, m):
    num_rollouts = len(rollout_indices)
    r1 = [[] for _ in range(num_rollouts)]
    steps = []

    try:
        for event in summary_iterator(event_path):
            if hasattr(event.summary, 'value') and len(event.summary.value) > 0:
                for i, n in enumerate(rollout_indices):
                    if event.summary.value[0].tag == tag + '{}'.format(n):
                        r1[i].append(event.summary.value[0].simple_value)
                        if i == 0:
                            steps.append(event.step)
    except:
        pass

    if len(np.unique([len(r) for r in r1])) > 1:
        print('warning: different lengths found')
    min_len = min([len(r) for r in r1])
    arr = np.array([np.array(r)[:min_len] for r in r1]).sum(axis=0)  # sum over all rollouts
    steps = np.array(steps)

    arr = moving_average(arr, m, only_past=True)

    return arr, steps


###################################################################################


def plot_learning_curve(x, y, label, mode='std', **kwargs):
    """
    Takes as input an x-value (number of frames)
    and a matrix of y-values (rows: runs, columns: results)
    """

    y = y[:, :len(x)]

    # get the mean (only where we have data) and compute moving average
    mean = np.sum(y, axis=0) / (np.sum(y != 0, axis=0) + 1e-6)
    p = plt.plot(x, mean, linewidth=2, label=label,
                 c=kwargs['color'] if 'color' in kwargs else cols_deep[0])

    if mode == 'std':
        # compute standard deviation
        std = np.std(y, axis=0)
        # compute confidence intervals
        cis = [mean - std, mean + std]

        plt.gca().fill_between(x, cis[0], cis[1], facecolor=p[0].get_color(), alpha=0.1)
    elif mode == 'all':
        plt.plot(x, y.T, linewidth=2, alpha=0.3, c=p[0].get_color())
    else:
        raise NotImplementedError


def plot_tb_results(env_name, exp_name, tag, m, **kwargs):
    """

    :param env_name:            name of the environment
    :param exp_name:            in env_name folder, which experiment
    :param m:                   parameter for temporally smoothing the curve
    :return:
    """

    results_directory = os.path.join(os.getcwd(), '../logs/{}'.format(env_name))
    exp_ids = [folder for folder in os.listdir(results_directory) if
               folder.startswith(exp_name + '__')]

    arrays = []
    for exp_id in exp_ids:
        exp_dir = os.path.join(results_directory, exp_id)
        tf_event = [event for event in os.listdir(exp_dir) if event.startswith('event')][0]

        if kwargs['multi_episode'] == True:
            arr, steps = get_array_from_event_multi_episode(os.path.join(exp_dir, tf_event), tag=tag,
                                                            rollout_indices=kwargs['rollout_indices'], m=m)
        else:
            arr, steps = get_array_from_event(os.path.join(exp_dir, tf_event), tag=tag, m=m)

        arrays.append(arr)

    arr_lens = np.array([len(array) for array in arrays])
    if len(np.unique(arr_lens)) > 1:
        min_len = min(arr_lens)
        arrays = [array[:min_len] for array in arrays]
        steps = steps[:min_len]
    arrays = np.vstack(arrays)

    plot_learning_curve(steps, arrays, label=kwargs['label'] if 'label' in kwargs else exp_name,
                        color=kwargs['color'])


def plot_results_from_dir(exp_dir, tag, m, **kwargs):
    tf_event = [event for event in os.listdir(exp_dir) if event.startswith('event')][0]

    arr, steps = get_array_from_event(os.path.join(exp_dir, tf_event), tag=tag, m=m)

    plt.plot(steps, arr)
    plt.show()


def get_eval_results_from_dir(dir, m=1):
    tf_event = [event for event in os.listdir(dir) if event.startswith('event')][0]

    arr, steps = get_array_from_event(os.path.join(dir, tf_event), tag='returns_multi_episode/sum_eval', m=m)

    return arr, steps


def get_times_of_momentum_change(dir):
    tf_event = [event for event in os.listdir(dir) if event.startswith('event')][0]
    momentum_arr, steps = get_array_from_event(os.path.join(dir, tf_event), tag='momentum', m=1)
    steps_of_momentum_change = []
    momentum_before = momentum_arr[0]
    for i, step in enumerate(steps):
        if momentum_arr[i] != momentum_before:
            steps_of_momentum_change.append(step)
            momentum_before = momentum_arr[i]
    return steps_of_momentum_change


def plot_results_from_dirs_together(dir_list=[], labels_list=[], m=1, is_changing_momentum=False):
    assert len(dir_list) == len(labels_list), "lengths don't match"
    arr_list = []
    steps_list = []
    for dir in dir_list:
        arr, steps = get_eval_results_from_dir(dir, m)
        arr_list.append(arr)
        steps_list.append(steps)
        plt.plot(steps, arr)
    plt.legend(labels_list)
    plt.xlabel("Training steps")
    plt.ylabel("Evaluation reward")
    plt.grid()
    if is_changing_momentum:
        steps_momentum_change = get_times_of_momentum_change(dir_list[0])
        for step in steps_momentum_change:
            plt.axvline(step, linestyle='--', color='black')
    plt.show()


def compare(env_names, exp_names, tags, m, ylabel, save_path=None, **kwargs):
    for i in range(len(env_names)):
        plot_tb_results(env_names[i], exp_names[i], tags[i], m,
                        multi_episode=kwargs['multi_episode'][i],
                        rollout_indices=kwargs['rollout_indices'][i],
                        label=kwargs['labels'][i], color=kwargs['colors'][i])

    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel('Frames', fontsize=15)
    # plt.xlabel('Train Steps', fontsize=15)
    # plt.title('Gridworld', fontsize=20)
    plt.title('Semi-circle', fontsize=20)
    from matplotlib.ticker import ScalarFormatter
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    tx = plt.gca().xaxis.get_offset_text()
    tx.set_fontsize(15)

    if 'truncate_at' in kwargs:
        plt.xlim([0., kwargs['truncate_at']])

    # plt.legend(fontsize=18, loc='lower right', prop={'size': 16})
    plt.legend(fontsize=18, loc='upper right', prop={'size': 16})
    plt.tight_layout()
    plt.gca().tick_params(axis='both', which='major', labelsize=15)
    plt.gca().tick_params(axis='both', which='minor', labelsize=15)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == '__main__':
    # # First try - compare with and without vae
    # dirnames = [
    #     "C:\\Users\\matan\\Documents\\OMRL\\logs\\PointRobotSparse-v0\\sac__73__29_01_16_48_03",
    #     "C:\\Users\\matan\\Documents\\OMRL\\logs\\PointRobotSparse-v0\\only_sac__73__31_01_23_09_04",
    #     "C:\\Users\\matan\\Documents\\OMRL\\logs\\PointRobotSparse-v0\\sacmer__73__02_02_16_19_34"
    # ]
    # labels = [
    #     "OMRL",
    #     "SAC (Policy only, no VAE)",
    #     "SAC+MER (Policy only, no VAE)"
    # ]
    # plot_results_from_dirs_together(dirnames, labels)
    #
    # # OMRL only - sequential learning comparison
    # dirnames = [
    #     "C:\\Users\\matan\\Documents\\OMRL\\logs\\PointRobotSparse-v0
    #     \\omrl_momentum__73__03_02_13_41_16_momentum_from_0.1",
    #     "C:\\Users\\matan\\Documents\\OMRL\\logs\\PointRobotSparse-v0
    #     \\omrl_momentum__73__16_02_22_42_09_momentum_from_0.82",
    #     "C:\\Users\\matan\\Documents\\OMRL\\logs\\PointRobotSparse-v0
    #     \\omrl_momentum__73__13_02_17_24_18_momentum_1_const"
    # ]
    # labels = [
    #     "momentum going from 0.1 to 1",
    #     "momentum going from 0.82 to 1",
    #     "momentum = 1 (const)",
    # ]
    #
    # plot_results_from_dirs_together(dirnames, labels, is_changing_momentum=True)
    #
    # # Momentum from 0.82 comparison
    # dirnames = [
    #     "C:\\Users\\matan\\Documents\\OMRL\\logs\\PointRobotSparse-v0
    #     \\omrl_momentum__73__16_02_22_42_09_momentum_from_0.82",
    #     "C:\\Users\\matan\\Documents\\OMRL\\logs\\PointRobotSparse-v0
    #     \\omrl_momentum_mer__73__18_02_09_25_13_momentum_from_0.82_gamma_0.3both_s_500policy_15vae"
    # ]
    # labels = [
    #     "OMRL",
    #     "OMRL + MER",
    # ]
    #
    # plot_results_from_dirs_together(dirnames, labels, is_changing_momentum=True)

    # Momentum from 0.1 comparison
    dirnames = [
        "C:\\Users\\matan\\Documents\\OMRL\\logs\\PointRobotSparse-v0"
        "\\omrl_momentum__73__03_02_13_41_16_momentum_from_0.1",
        "C:\\Users\\matan\\Documents\\OMRL\\logs\\PointRobotSparse-v0"
        "\\omrl_momentum_mer__73__08_02_09_34_58_momentum_from_0.1_gamma_0.5both_s_50policy_5vae",
        "C:\\Users\\matan\\Documents\\OMRL\\logs\\PointRobotSparse-v0"
        "\\omrl_momentum_mer__73__09_02_23_32_47_momentum_from_0.1_gamma_0.1policy_0.3vae_s_500policy_12vae",
        "C:\\Users\\matan\\Documents\\OMRL\\logs\\PointRobotSparse-v0"
        "\\omrl_momentum_mer__73__12_02_01_31_43_momentum_from_0.1_gamma_0.1both_s_1000policy_25vae",
    ]
    labels = [
        "OMRL",
        "OMRL + MER (1)",
        "OMRL + MER (2)",
        "OMRL + MER (3)",
    ]

    plot_results_from_dirs_together(dirnames, labels, is_changing_momentum=True)

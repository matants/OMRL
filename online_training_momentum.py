import os
import glob
import torch
import argparse
import multiprocessing

from torchkit.pytorch_utils import set_gpu_mode
from metalearner_momentum import MetaLearnerMomentum
from metalearner_momentum_mer import MetaLearnerMomentumMer
from online_config import args_gridworld, args_point_robot, args_point_robot_sparse, \
    args_cheetah_vel, args_ant_semicircle, args_ant_semicircle_sparse


def main(args_list: list = []):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env-type', default='cheetah_vel')
    parser.add_argument('--env-type', default='point_robot_sparse')
    # parser.add_argument('--env-type', default='gridworld')
    args, rest_args = parser.parse_known_args(args_list)
    env = args.env_type

    # --- GridWorld ---
    if env == 'gridworld':
        args = args_gridworld.get_args(rest_args)
    # --- PointRobot ---
    elif env == 'point_robot':
        args = args_point_robot.get_args(rest_args)
    elif env == 'point_robot_sparse':
        args = args_point_robot_sparse.get_args(rest_args)
    # --- Mujoco ---
    elif env == 'cheetah_vel':
        args = args_cheetah_vel.get_args(rest_args)
    elif env == 'ant_semicircle':
        args = args_ant_semicircle.get_args(rest_args)
    elif env == 'ant_semicircle_sparse':
        args = args_ant_semicircle_sparse.get_args(rest_args)

    # make sure we have log directories
    try:
        os.makedirs(args.agent_log_dir)
    except OSError:
        files = glob.glob(os.path.join(args.agent_log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
    eval_log_dir = args.agent_log_dir + "_eval"
    try:
        os.makedirs(eval_log_dir)
    except OSError:
        files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    # set gpu
    set_gpu_mode(torch.cuda.is_available() and args.use_gpu)

    # start training
    if args.output_file_prefix == 'omrl_momentum':
        learner = MetaLearnerMomentum(args)
    elif args.output_file_prefix == 'omrl_momentum_mer':
        learner = MetaLearnerMomentumMer(args)
    else:
        raise NotImplementedError

    learner.train()


def main_proc(args):
    p = multiprocessing.Process(target=main, args=(args,))
    p.start()
    p.join()


if __name__ == '__main__':
    for i in range(73, 73 + 5):
        main_proc(['--seed', f'{i}',
                   '--batch-size', str(256),
                   '--dqn-layers', str(64), str(32),
                   '--policy-layers', str(64), str(32),
                   '--aggregator-hidden-size', str(64),
                   '--reward-decoder-layers', str(32), str(16),
                   '--output-file-prefix', 'omrl_momentum_mer',
                   ])

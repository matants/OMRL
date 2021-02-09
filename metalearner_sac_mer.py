import os
import time

import gym
import numpy as np
import torch
import copy

from algorithms.dqn import DQN, DoubleDQN
from algorithms.sac import SAC
from environments.make_env import make_env
from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from torchkit.networks import FlattenMlp
from data_management.storage_policy import MultiTaskPolicyStorage
from utils.tb_logger import TBLogger
from models.policy import TanhGaussianPolicy


class MetaLearnerSACMER:
    """
    Meta-Learner class.
    """

    def __init__(self, args):
        """
        Seeds everything.
        Initialises: logger, environments, policy (+storage +optimiser).
        """

        self.args = args

        # make sure everything has the same seed
        utl.seed(self.args.seed)

        # initialize tensorboard logger
        if self.args.log_tensorboard:
            self.tb_logger = TBLogger(self.args)

        # initialise environment
        self.env = make_env(self.args.env_name,
                            self.args.max_rollouts_per_task,
                            seed=self.args.seed,
                            n_tasks=self.args.num_tasks)

        # unwrapped env to get some info about the environment
        unwrapped_env = self.env.unwrapped
        # split to train/eval tasks
        shuffled_tasks = np.random.permutation(unwrapped_env.get_all_task_idx())
        self.train_tasks = shuffled_tasks[:self.args.num_train_tasks]
        if self.args.num_eval_tasks > 0:
            self.eval_tasks = shuffled_tasks[-self.args.num_eval_tasks:]
        else:
            self.eval_tasks = []
        # calculate what the maximum length of the trajectories is
        args.max_trajectory_len = unwrapped_env._max_episode_steps
        args.max_trajectory_len *= self.args.max_rollouts_per_task
        self.args.max_trajectory_len = args.max_trajectory_len

        # get action / observation dimensions
        if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self.args.action_dim = 1
        else:
            self.args.action_dim = self.env.action_space.shape[0]
        self.args.obs_dim = self.env.observation_space.shape[0]
        self.args.num_states = unwrapped_env.num_states if hasattr(unwrapped_env, 'num_states') else None
        self.args.act_space = self.env.action_space

        # initialize policy
        self.initialize_policy()
        # initialize buffer for RL updates
        self.policy_storage = MultiTaskPolicyStorage(
            max_replay_buffer_size=int(self.args.policy_buffer_size),
            obs_dim=self._get_augmented_obs_dim(),
            action_space=self.env.action_space,
            tasks=self.train_tasks,
            trajectory_len=args.max_trajectory_len,
        )
        self.current_experience_storage = None

        self.args.belief_reward = False  # initialize arg to not use belief rewards

    def initialize_current_experience_storage(self):
        self.current_experience_storage = MultiTaskPolicyStorage(
            max_replay_buffer_size=int(
                self.args.num_tasks_sample * self.args.num_rollouts_per_iter * self.args.max_trajectory_len),
            obs_dim=self._get_augmented_obs_dim(),
            action_space=self.env.action_space,
            tasks=self.train_tasks,
            trajectory_len=self.args.max_trajectory_len,
        )

    def initialize_policy(self):

        if self.args.policy == 'dqn':
            assert self.args.act_space.__class__.__name__ == "Discrete", (
                "Can't train DQN with continuous action space!")
            q_network = FlattenMlp(input_size=self._get_augmented_obs_dim(),
                                   output_size=self.args.act_space.n,
                                   hidden_sizes=self.args.dqn_layers)
            self.agent = DQN(
                q_network,
                # optimiser_vae=self.optimizer_vae,
                lr=self.args.policy_lr,
                eps_optim=self.args.dqn_eps,
                alpha_optim=self.args.dqn_alpha,
                gamma=self.args.gamma,
                eps_init=self.args.dqn_epsilon_init,
                eps_final=self.args.dqn_epsilon_final,
                exploration_iters=self.args.dqn_exploration_iters,
                tau=self.args.soft_target_tau,
            ).to(ptu.device)
        elif self.args.policy == 'ddqn':
            assert self.args.act_space.__class__.__name__ == "Discrete", (
                "Can't train DDQN with continuous action space!")
            q_network = FlattenMlp(input_size=self._get_augmented_obs_dim(),
                                   output_size=self.args.act_space.n,
                                   hidden_sizes=self.args.dqn_layers)
            self.agent = DoubleDQN(
                q_network,
                # optimiser_vae=self.optimizer_vae,
                lr=self.args.policy_lr,
                eps_optim=self.args.dqn_eps,
                alpha_optim=self.args.dqn_alpha,
                gamma=self.args.gamma,
                eps_init=self.args.dqn_epsilon_init,
                eps_final=self.args.dqn_epsilon_final,
                exploration_iters=self.args.dqn_exploration_iters,
                tau=self.args.soft_target_tau,
            ).to(ptu.device)
        elif self.args.policy == 'sac':
            assert self.args.act_space.__class__.__name__ == "Box", (
                "Can't train SAC with discrete action space!")
            q1_network = FlattenMlp(input_size=self._get_augmented_obs_dim() + self.args.action_dim,
                                    output_size=1,
                                    hidden_sizes=self.args.dqn_layers)
            q2_network = FlattenMlp(input_size=self._get_augmented_obs_dim() + self.args.action_dim,
                                    output_size=1,
                                    hidden_sizes=self.args.dqn_layers)
            policy = TanhGaussianPolicy(obs_dim=self._get_augmented_obs_dim(),
                                        action_dim=self.args.action_dim,
                                        hidden_sizes=self.args.policy_layers)
            self.agent = SAC(
                policy,
                q1_network,
                q2_network,

                actor_lr=self.args.actor_lr,
                critic_lr=self.args.critic_lr,
                gamma=self.args.gamma,
                tau=self.args.soft_target_tau,

                entropy_alpha=self.args.entropy_alpha,
                automatic_entropy_tuning=self.args.automatic_entropy_tuning,
                alpha_lr=self.args.alpha_lr
            ).to(ptu.device)
        else:
            raise NotImplementedError

    def train(self):
        """
        meta-training loop
        """

        self._start_training()
        for iter_ in range(self.args.num_iters):
            self.training_mode(True)
            # switch to belief reward
            if self.args.switch_to_belief_reward is not None and iter_ >= self.args.switch_to_belief_reward:
                self.args.belief_reward = True
            if iter_ == 0:
                print('Collecting initial pool of data..')
                for task in self.train_tasks:
                    self.task_idx = task
                    self.env.reset_task(idx=task)
                    # self.collect_rollouts(num_rollouts=self.args.num_init_rollouts_pool)
                    self.collect_rollouts(num_rollouts=self.args.num_init_rollouts_pool, random_actions=True)
                print('Done!')

            # collect data from subset of train tasks
            tasks_to_collect = [self.train_tasks[np.random.randint(len(self.train_tasks))] for _ in
                                range(self.args.num_tasks_sample)]
            self.initialize_current_experience_storage()
            for i in range(self.args.num_tasks_sample):
                task = tasks_to_collect[i]
                self.task_idx = task
                self.env.reset_task(idx=task)
                self.collect_rollouts(num_rollouts=self.args.num_rollouts_per_iter)
            # update
            indices = np.random.choice(self.train_tasks, self.args.meta_batch)
            train_stats = self.update(indices, current_task_indices=tasks_to_collect)
            self.training_mode(False)

            if self.args.policy == 'dqn':
                self.agent.set_exploration_parameter(iter_ + 1)
            # evaluate and log
            if (iter_ + 1) % self.args.log_interval == 0:
                self.log(iter_ + 1, train_stats)

    def update(self, tasks, current_task_indices=[]):
        '''
        Meta-update
        :param tasks: list/array of task indices. perform update based on the tasks
        :param current_task_indices: indices that the last rollout collected
        :return:
        '''

        # --- RL TRAINING ---
        rl_losses_agg = {}
        prev_qf1 = copy.deepcopy(self.agent.qf1).to(ptu.device)
        prev_qf2 = copy.deepcopy(self.agent.qf2).to(ptu.device)
        prev_qf1_target = copy.deepcopy(self.agent.qf1_target).to(ptu.device)
        prev_qf2_target = copy.deepcopy(self.agent.qf2_target).to(ptu.device)
        prev_policy = copy.deepcopy(self.agent.policy).to(ptu.device)
        # prev_alpha = copy.deepcopy(self.agent.log_alpha_entropy)  # not relevant when alpha is const
        prev_nets = [prev_qf1, prev_qf2, prev_qf1_target, prev_qf2_target, prev_policy]

        updates_from_current_experience = np.random.choice(self.args.rl_updates_per_iter, self.args.mer_s,
                                                           replace=False)

        for update in range(self.args.rl_updates_per_iter):
            # sample random RL batch
            obs, actions, rewards, next_obs, terms = self.sample_rl_batch(
                tasks if update not in updates_from_current_experience else current_task_indices, self.args.batch_size,
                is_sample_from_current_experience=(update in updates_from_current_experience))
            # flatten out task dimension
            t, b, _ = obs.size()
            obs = obs.view(t * b, -1)
            actions = actions.view(t * b, -1)
            rewards = rewards.view(t * b, -1)
            next_obs = next_obs.view(t * b, -1)
            terms = terms.view(t * b, -1)

            # RL update
            rl_losses = self.agent.update(obs, actions, rewards, next_obs, terms)

            for k, v in rl_losses.items():
                if update == 0:  # first iterate - create list
                    rl_losses_agg[k] = [v]
                else:  # append values
                    rl_losses_agg[k].append(v)

        # reptile update
        new_nets = [self.agent.qf1, self.agent.qf2, self.agent.qf1_target, self.agent.qf2_target, self.agent.policy]
        for new_net, prev_net in zip(new_nets, prev_nets):
            ptu.soft_update_from_to(new_net, prev_net, self.args.mer_gamma_policy)
            ptu.copy_model_params_from_to(prev_net, new_net)


        # take mean
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        self._n_rl_update_steps_total += self.args.rl_updates_per_iter

        train_stats = {**rl_losses_agg, }

        return train_stats

    def evaluate(self, tasks):
        num_episodes = self.args.max_rollouts_per_task
        num_steps_per_episode = self.env.unwrapped._max_episode_steps

        returns_per_episode = np.zeros((len(tasks), num_episodes))
        success_rate = np.zeros(len(tasks))

        if self.args.policy == 'dqn':
            values = np.zeros((len(tasks), self.args.max_trajectory_len))
        else:
            rewards = np.zeros((len(tasks), self.args.max_trajectory_len))
            obs_size = self.env.unwrapped.observation_space.shape[0]
            observations = np.zeros((len(tasks), self.args.max_trajectory_len + 1, obs_size))

        for task_idx, task in enumerate(tasks):
            obs = ptu.from_numpy(self.env.reset(task))
            obs = obs.reshape(-1, obs.shape[-1])
            step = 0

            if self.args.policy == 'dqn':
                raise NotImplementedError
            else:
                observations[task_idx, step, :] = ptu.get_numpy(obs[0, :obs_size])

            for episode_idx in range(num_episodes):
                running_reward = 0.
                for step_idx in range(num_steps_per_episode):
                    # add distribution parameters to observation - policy is conditioned on posterior
                    augmented_obs = self.get_augmented_obs(obs=obs)
                    if self.args.policy == 'dqn':
                        action, value = self.agent.act(obs=augmented_obs, deterministic=True)
                    else:
                        action, _, _, log_prob = self.agent.act(obs=augmented_obs,
                                                                deterministic=self.args.eval_deterministic,
                                                                return_log_prob=True)
                    # observe reward and next obs
                    next_obs, reward, done, info = utl.env_step(self.env, action.squeeze(dim=0))
                    running_reward += reward.item()

                    if self.args.policy == 'dqn':
                        values[task_idx, step] = value.item()
                    else:
                        rewards[task_idx, step] = reward.item()
                        observations[task_idx, step + 1, :] = ptu.get_numpy(next_obs[0, :obs_size])

                    if "is_goal_state" in dir(self.env.unwrapped) and self.env.unwrapped.is_goal_state():
                        success_rate[task_idx] = 1.
                    # set: obs <- next_obs
                    obs = next_obs.clone()
                    step += 1

                returns_per_episode[task_idx, episode_idx] = running_reward

        if self.args.policy == 'dqn':
            return returns_per_episode, success_rate, values,
        else:
            return returns_per_episode, success_rate, observations, rewards,

    def log(self, iteration, train_stats):
        # --- save models ---
        if iteration % self.args.save_interval == 0:
            save_path = os.path.join(self.tb_logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(self.agent.state_dict(), os.path.join(save_path, "agent{0}.pt".format(iteration)))

        # evaluate to get more stats
        if self.args.policy == 'dqn':
            # get stats on train tasks
            returns_train, success_rate_train, values = self.evaluate(self.train_tasks)
        else:
            # get stats on train tasks
            returns_train, success_rate_train, observations, rewards_train = self.evaluate(
                self.train_tasks[:len(self.eval_tasks)])
            returns_eval, success_rate_eval, observations_eval, rewards_eval = self.evaluate(self.eval_tasks)

        if self.args.log_tensorboard:
            # some metrics
            self.tb_logger.writer.add_scalar('metrics/successes_in_buffer',
                                             self._successes_in_buffer / self._n_env_steps_total,
                                             self._n_env_steps_total)

            if self.args.max_rollouts_per_task > 1:
                for episode_idx in range(self.args.max_rollouts_per_task):
                    self.tb_logger.writer.add_scalar('returns_multi_episode/episode_{}'.
                                                     format(episode_idx + 1),
                                                     np.mean(returns_train[:, episode_idx]),
                                                     self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('returns_multi_episode/sum',
                                                 np.mean(np.sum(returns_train, axis=-1)),
                                                 self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('returns_multi_episode/success_rate',
                                                 np.mean(success_rate_train),
                                                 self._n_env_steps_total)
                if self.args.policy != 'dqn':
                    self.tb_logger.writer.add_scalar('returns_multi_episode/sum_eval',
                                                     np.mean(np.sum(returns_eval, axis=-1)),
                                                     self._n_env_steps_total)
                    self.tb_logger.writer.add_scalar('returns_multi_episode/success_rate_eval',
                                                     np.mean(success_rate_eval),
                                                     self._n_env_steps_total)
            else:
                # self.tb_logger.writer.add_scalar('returns/returns_mean', np.mean(returns),
                #                                  self._n_env_steps_total)
                # self.tb_logger.writer.add_scalar('returns/returns_std', np.std(returns),
                #                                  self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('returns/returns_mean_train', np.mean(returns_train),
                                                 self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('returns/returns_std_train', np.std(returns_train),
                                                 self._n_env_steps_total)
                # self.tb_logger.writer.add_scalar('returns/success_rate', np.mean(success_rate),
                #                                  self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('returns/success_rate_train', np.mean(success_rate_train),
                                                 self._n_env_steps_total)

            # policy
            if self.args.policy == 'dqn':
                self.tb_logger.writer.add_scalar('policy/value_init', np.mean(values[:, 0]), self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('policy/value_halfway', np.mean(values[:, int(values.shape[-1] / 2)]),
                                                 self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('policy/value_final', np.mean(values[:, -1]), self._n_env_steps_total)

                self.tb_logger.writer.add_scalar('policy/exploration_epsilon', self.agent.eps, self._n_env_steps_total)
                # RL losses
                self.tb_logger.writer.add_scalar('rl_losses/qf_loss_vs_n_updates', train_stats['qf_loss'],
                                                 self._n_rl_update_steps_total)
                self.tb_logger.writer.add_scalar('rl_losses/qf_loss_vs_n_env_steps', train_stats['qf_loss'],
                                                 self._n_env_steps_total)
            else:
                self.tb_logger.writer.add_scalar('rl_losses/qf1_loss', train_stats['qf1_loss'],
                                                 self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('rl_losses/qf2_loss', train_stats['qf2_loss'],
                                                 self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('rl_losses/policy_loss', train_stats['policy_loss'],
                                                 self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('rl_losses/alpha_loss', train_stats['alpha_entropy_loss'],
                                                 self._n_env_steps_total)

            # weights and gradients
            if self.args.policy == 'dqn':
                self.tb_logger.writer.add_scalar('weights/q_network',
                                                 list(self.agent.qf.parameters())[0].mean(),
                                                 self._n_env_steps_total)
                if list(self.agent.qf.parameters())[0].grad is not None:
                    param_list = list(self.agent.qf.parameters())
                    self.tb_logger.writer.add_scalar('gradients/q_network',
                                                     sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                     self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('weights/q_target',
                                                 list(self.agent.target_qf.parameters())[0].mean(),
                                                 self._n_env_steps_total)
                if list(self.agent.target_qf.parameters())[0].grad is not None:
                    param_list = list(self.agent.target_qf.parameters())
                    self.tb_logger.writer.add_scalar('gradients/q_target',
                                                     sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                     self._n_env_steps_total)
            else:
                self.tb_logger.writer.add_scalar('weights/q1_network',
                                                 list(self.agent.qf1.parameters())[0].mean(),
                                                 self._n_env_steps_total)
                if list(self.agent.qf1.parameters())[0].grad is not None:
                    param_list = list(self.agent.qf1.parameters())
                    self.tb_logger.writer.add_scalar('gradients/q1_network',
                                                     sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                     self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('weights/q1_target',
                                                 list(self.agent.qf1_target.parameters())[0].mean(),
                                                 self._n_env_steps_total)
                if list(self.agent.qf1_target.parameters())[0].grad is not None:
                    param_list = list(self.agent.qf1_target.parameters())
                    self.tb_logger.writer.add_scalar('gradients/q1_target',
                                                     sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                     self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('weights/q2_network',
                                                 list(self.agent.qf2.parameters())[0].mean(),
                                                 self._n_env_steps_total)
                if list(self.agent.qf2.parameters())[0].grad is not None:
                    param_list = list(self.agent.qf2.parameters())
                    self.tb_logger.writer.add_scalar('gradients/q2_network',
                                                     sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                     self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('weights/q2_target',
                                                 list(self.agent.qf2_target.parameters())[0].mean(),
                                                 self._n_env_steps_total)
                if list(self.agent.qf2_target.parameters())[0].grad is not None:
                    param_list = list(self.agent.qf2_target.parameters())
                    self.tb_logger.writer.add_scalar('gradients/q2_target',
                                                     sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                     self._n_env_steps_total)
                self.tb_logger.writer.add_scalar('weights/policy',
                                                 list(self.agent.policy.parameters())[0].mean(),
                                                 self._n_env_steps_total)
                if list(self.agent.policy.parameters())[0].grad is not None:
                    param_list = list(self.agent.policy.parameters())
                    self.tb_logger.writer.add_scalar('gradients/policy',
                                                     sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                     self._n_env_steps_total)

            #log iteration
            self.tb_logger.writer.add_scalar('iteration', iteration, self._n_env_steps_total)

        # output to user
        # print("Iteration -- {:3d}, Num. RL updates -- {:6d}, Elapsed time {:5d}[s]".
        #       format(iteration,
        #              self._n_rl_update_steps_total,
        #              int(time.time() - self._start_time)))
        print("Iteration -- {}, Success rate train -- {:.3f}, Success rate eval.-- {:.3f}, "
              "Avg. return train -- {:.3f}, Avg. return eval. -- {:.3f}, Elapsed time {:5d}[s]"
              .format(iteration, np.mean(success_rate_train),
                      np.mean(success_rate_eval), np.mean(np.sum(returns_train, axis=-1)),
                      np.mean(np.sum(returns_eval, axis=-1)),
                      int(time.time() - self._start_time)))

    def training_mode(self, mode):
        # policy
        self.agent.train(mode)

    def collect_rollouts(self, num_rollouts, random_actions=False):
        '''

        :param num_rollouts:
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        :return:
        '''

        for rollout in range(num_rollouts):
            obs = ptu.from_numpy(self.env.reset(self.task_idx))
            obs = obs.reshape(-1, obs.shape[-1])
            done_rollout = False
            # self.policy_storage.reset_running_episode(self.task_idx)

            # if self.args.fixed_latent_params:
            #     assert 2 ** self.args.task_embedding_size >= self.args.num_tasks
            #     task_mean = ptu.FloatTensor(utl.vertices(self.args.task_embedding_size)[self.task_idx])
            #     task_logvar = -2. * ptu.ones_like(task_logvar)   # arbitrary negative enough number
            # add distribution parameters to observation - policy is conditioned on posterior
            augmented_obs = self.get_augmented_obs(obs=obs)

            while not done_rollout:
                if random_actions:
                    if self.args.policy == 'dqn':
                        action = ptu.FloatTensor([[self.env.action_space.sample()]]).type(
                            torch.long)  # Sample random action
                    else:
                        action = ptu.FloatTensor([self.env.action_space.sample()])
                else:
                    if self.args.policy == 'dqn':
                        action, _ = self.agent.act(obs=augmented_obs)  # DQN
                    else:
                        action, _, _, _ = self.agent.act(obs=augmented_obs)  # SAC
                # observe reward and next obs
                next_obs, reward, done, info = utl.env_step(self.env, action.squeeze(dim=0))
                done_rollout = False if ptu.get_numpy(done[0][0]) == 0. else True

                # get augmented next obs
                augmented_next_obs = self.get_augmented_obs(obs=next_obs)

                # add data to policy buffer - (s+, a, r, s'+, term)
                term = self.env.unwrapped.is_goal_state() if "is_goal_state" in dir(self.env.unwrapped) else False
                self.policy_storage.add_sample(task=self.task_idx,
                                               observation=ptu.get_numpy(augmented_obs.squeeze(dim=0)),
                                               action=ptu.get_numpy(action.squeeze(dim=0)),
                                               reward=ptu.get_numpy(reward.squeeze(dim=0)),
                                               terminal=np.array([term], dtype=float),
                                               next_observation=ptu.get_numpy(augmented_next_obs.squeeze(dim=0)))
                if not random_actions:
                    self.current_experience_storage.add_sample(task=self.task_idx,
                                                               observation=ptu.get_numpy(augmented_obs.squeeze(dim=0)),
                                                               action=ptu.get_numpy(action.squeeze(dim=0)),
                                                               reward=ptu.get_numpy(reward.squeeze(dim=0)),
                                                               terminal=np.array([term], dtype=float),
                                                               next_observation=ptu.get_numpy(
                                                                   augmented_next_obs.squeeze(dim=0)))

                # set: obs <- next_obs
                obs = next_obs.clone()
                augmented_obs = augmented_next_obs.clone()

                # update statistics
                self._n_env_steps_total += 1
                if "is_goal_state" in dir(self.env.unwrapped) and self.env.unwrapped.is_goal_state():  # count successes
                    self._successes_in_buffer += 1
            self._n_rollouts_total += 1

    def get_augmented_obs(self, obs):
        augmented_obs = obs.clone()
        return augmented_obs

    def _get_augmented_obs_dim(self):
        dim = utl.get_dim(self.env.observation_space)
        return dim

    def sample_rl_batch(self, tasks, batch_size, is_sample_from_current_experience=False):
        ''' sample batch of unordered rl training data from a list/array of tasks '''
        # this batch consists of transitions sampled randomly from replay buffer
        if is_sample_from_current_experience:
            batches = [ptu.np_to_pytorch_batch(
                self.current_experience_storage.random_batch(task, batch_size)) for task in tasks]
        else:
            batches = [ptu.np_to_pytorch_batch(
                self.policy_storage.random_batch(task, batch_size)) for task in tasks]
        unpacked = [utl.unpack_batch(batch) for batch in batches]
        # group elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def _start_training(self):
        self._n_env_steps_total = 0
        self._n_rl_update_steps_total = 0
        self._n_rollouts_total = 0
        self._successes_in_buffer = 0

        self._start_time = time.time()

    def load_model(self, device='cpu', **kwargs):
        if "agent_path" in kwargs:
            self.agent.load_state_dict(torch.load(kwargs["agent_path"], map_location=device))
        self.training_mode(False)

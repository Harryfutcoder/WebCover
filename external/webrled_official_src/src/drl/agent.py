import pickle
import collections

import lz4.frame as lz4f
import torch
import torch.nn.functional as F

from src.drl.buffer import EpisodeBuffer
from src.drl.model import EmbeddingNet, LifeLongNet, QNetwork, AutoEncoder, LongAutoEncoder
from src.drl.utils import (UCB, create_beta_list, create_gamma_list, inverse_rescaling, play_episode, rescaling,
                           segments2contents, transformed_retrace_operator)


class Agent:
    """
    collect experiments and get initial priority
    Attributes:
      pid                 (int): process id
      env              (Webenv): environment
      input_dim            (int): dim of state
      env          (gym object): environment
      action_space        (int): dim of action space
      in_q_network             : q network about intrinsic reward
      ex_q_network             : q network about extrinsic reward
      embedding_net            : embedding network to get episodic reward
      embedding_classifier     : classify action based on embedding representation
      original_lifelong_net    : lifelong network not to be trained
      trained_lifelong_net     : lifelong network to be trained
      ucb                      : object of UCB class which solve a multi-armed bandit problem
      betas              (list): list of beta which decide weights between intrinsic qvalues and extrinsic qvalues
      gammas             (list): list of gamma which is discount rate
      epsilon           (float): coefficient for epsilon greedy
      eta               (float): coefficient for priority caluclation
      lamda             (float): coefficient for retrace operation
      burnin_length       (int): length of burnin to calculate qvalues
      unroll_length       (int): length of unroll to calculate qvalues
      k                   (int): number of neighbors referenced when calculating episode reward
      L                   (int): upper limit of curiosity
      error_list               : list of errors to be accommodated when calculating lifelong reward
      agent_update_period (int): how often to update the target parameters
      num_rollout         (int): minimum segments to be got
      num_updates         (int): number of times to be updated
    """

    def __init__(self,
                 pid,
                 env,
                 input_dim,
                 epsilon,
                 eta,
                 lamda,
                 burnin_length,
                 unroll_length,
                 k,
                 L,
                 agent_update_period,
                 num_rollout,
                 num_arms,
                 window_size,
                 ucb_epsilon,
                 ucb_beta,
                 original_lifelong_weight):
        """
        Args:
          pid                 (int): process id
          env_name            (str): name of environment
          n_frames            (int): number of images to be stacked
          epsilon           (float): coefficient for epsilon soft-max
          eta               (float): coefficient for priority caluclation
          lamda             (float): coefficient for retrace operation
          burnin_length       (int): length of burnin to calculate qvalues
          unroll_length       (int): length of unroll to calculate qvalues
          k                   (int): number of neighbors referenced when calculating episode reward
          L                   (int): upper limit of curiosity
          agent_update_period (int): how often to update the target parameters
          num_rollout         (int): minimum segments to be got
          num_arms            (int): number of arms used in multi-armed bandit problem
          window_size         (int): size of window used in multi-armed bandit problem
          ucb_epsilon       (float): probability to select randomly used in multi-armed bandit problem
          ucb_beta          (float): weight between frequency and mean reward
          original_lifelong_weight : original weight of lifelong network
        """

        self.pid = pid
        self.env = env
        self.action_space = self.env.action_space_dim
        self.input_dim = input_dim

        self.in_q_network = QNetwork(self.action_space, input_dim)
        self.ex_q_network = QNetwork(self.action_space, input_dim)
        self.embedding_net = EmbeddingNet(input_dim)
        self.original_lifelong_net = LifeLongNet(input_dim)
        self.trained_lifelong_net = LifeLongNet(input_dim)
        self.short_auto_encoder = AutoEncoder(input_dim)
        self.long_auto_encoder = LongAutoEncoder(input_dim)

        self.ucb = UCB(num_arms, window_size, ucb_epsilon, ucb_beta)
        self.betas = create_beta_list(num_arms)
        self.gammas = create_gamma_list(num_arms)

        self.epsilon = epsilon
        self.eta = eta
        self.lamda = lamda

        self.burnin_len = burnin_length
        self.unroll_len = unroll_length

        self.k = k
        self.error_list = collections.deque(maxlen=int(1e4))
        self.short_ac_error_list = collections.deque(maxlen=int(1e4))
        self.long_ac_error_list = collections.deque(maxlen=int(1e4))
        self.L = L

        self.agent_update_period = agent_update_period
        self.num_rollout = num_rollout

        self.original_lifelong_net.load_state_dict(original_lifelong_weight)

        self.num_updates = 0

    def sync_weights_and_rollout(self, in_q_weight, ex_q_weight, embed_weight, lifelong_weight,
                                 short_auto_encoder_weight, long_auto_encoder_weight):
        """
        load weight and run rollout
        Args:
          in_q_weight     : weight of intrinsic q network
          ex_q_weight     : weight of extrinsic q network
          embed_weight    : weight of embedding network
          lifelong_weight : weight of lifelong network
        Returns:
          priority  (list): priority of segments when pulling segments from sum tree
          segments        : parts of expecimences
          self.pid        : process id
        """
        if self.num_updates % self.agent_update_period == 0:
            self.in_q_network.load_state_dict(in_q_weight)
            self.ex_q_network.load_state_dict(ex_q_weight)
            self.embedding_net.load_state_dict(embed_weight)
            self.trained_lifelong_net.load_state_dict(lifelong_weight)
            self.short_auto_encoder.load_state_dict(short_auto_encoder_weight)
            self.long_auto_encoder.load_state_dict(long_auto_encoder_weight)

        priorities, segments = [], []
        while len(segments) < self.num_rollout:
            _priorities, _segments = self._rollout()
            if not _segments:
                break
            priorities += _priorities
            segments += _segments

        self.num_updates += 1
        return priorities, segments, self.pid

    def _rollout(self):
        """
        get priority and segments from collected experiments
        Returns:
          priorities    (list): priorities of segments when pulling segments from sum tree
          compressed_segments : compressed segments in terms of  memory capacity
        """

        # get index from ucb
        j = self.ucb.pull_index()

        # get beta gamma
        beta, self.gamma = self.betas[j], self.gammas[j]

        episode_buffer = EpisodeBuffer(burnin_length=self.burnin_len, unroll_length=self.unroll_len)

        ucb_datas, transitions, self.error_list, self.short_ac_error_list, self.long_ac_error_list = play_episode(
            env_in=self.env,
            input_dim=self.input_dim,
            action_space=self.action_space,
            j=j,
            epsilon=self.epsilon,
            k=self.k,
            error_list=self.error_list,
            short_ac_error_list=self.short_ac_error_list,
            long_ac_error_list=self.long_ac_error_list,
            L=self.L,
            beta=beta,
            in_q_network=self.in_q_network,
            ex_q_network=self.ex_q_network,
            embedding_net=self.embedding_net,
            original_lifelong_net=self.original_lifelong_net,
            trained_lifelong_net=self.trained_lifelong_net,
            short_auto_encoder=self.short_auto_encoder,
            long_auto_encoder=self.long_auto_encoder
        )

        self.ucb.push_data(ucb_datas)

        for transition in transitions:
            episode_buffer.add(transition)

        segments = episode_buffer.pull_segments()
        if not segments:
            return [], []

        self.states, self.actions, self.in_rewards, self.ex_rewards, self.dones, self.j, self.next_states, \
            in_h0, in_c0, ex_h0, ex_c0, self.prev_in_rewards, self.prev_ex_rewards, self.prev_actions, self.time_step, \
            self.is_nearest_point = segments2contents(segments, self.burnin_len)

        # (unroll_len+1, batch_size, action_space)
        in_qvalues = self.get_qvalues(self.in_q_network, in_h0, in_c0)

        # (unroll_len+1, batch_size, action_space)
        ex_qvalues = self.get_qvalues(self.ex_q_network, ex_h0, ex_c0)

        # (unroll_len+1, batch_size)
        self.pi = torch.argmax(rescaling(inverse_rescaling(ex_qvalues) + beta * inverse_rescaling(in_qvalues)), dim=2)

        # (unroll_len, batch_size, action_space)
        self.actions_onehot = F.one_hot(self.actions[self.burnin_len:], num_classes=self.action_space)

        in_priorities = self.get_priorities(in_qvalues, self.in_rewards)
        ex_priorities = self.get_priorities(ex_qvalues, self.ex_rewards)

        priorities = in_priorities + ex_priorities
        compressed_segments = [lz4f.compress(pickle.dumps(seg)) for seg in segments]

        return priorities.detach().numpy().tolist(), compressed_segments

    def get_qvalues(self, q_network, h, c):
        """
        get qvalues from expeiences using q network
        Args:
          q_network             : network to get Q values
          h       (torch.tensor): LSTM hidden state
          c       (torch.tensor): LSTM cell state
        Returns:
          qvalues (torch.tensor): Q values [unroll_len+1, batch_size, action_space]
        """
        h_current, c_current = h, c
        h_next, c_next = h, c

        qvalues = []
        for t in range(self.burnin_len + self.unroll_len):
            if t > 0:
                for sample_index in range(len(self.time_step[t])):
                    last_t = t - 1
                    if self.time_step[last_t][sample_index] != self.time_step[t][sample_index]:
                        t_h_current, t_c_current = h_current.clone(), c_current.clone()
                        t_h_current[0][sample_index][:] = h_next[0][sample_index][:]
                        t_c_current[0][sample_index][:] = c_next[0][sample_index][:]
                        h_current, c_current = t_h_current, t_c_current

            lstm_states = (h_current, c_current)

            qvalue, (h, c) = q_network(self.states[t],
                                       states=lstm_states,
                                       prev_action=self.prev_actions[t],
                                       j=self.j,
                                       prev_in_rewards=self.prev_in_rewards[t],
                                       prev_ex_rewards=self.prev_ex_rewards[t])
            if t >= self.burnin_len:
                qvalues.append(qvalue)

            if t == 0:
                h_next, c_next = h, c
            else:
                for sample_index in range(len(self.time_step[t])):
                    last_t = max(t - 1, 0)
                    if self.time_step[last_t][sample_index] != self.time_step[t][sample_index]:
                        h_next[0][sample_index][:] = h[0][sample_index][:]
                        c_next[0][sample_index][:] = c[0][sample_index][:]
                    if self.is_nearest_point[t][sample_index]:
                        h_next[0][sample_index][:] = h[0][sample_index][:]
                        c_next[0][sample_index][:] = c[0][sample_index][:]

        # self.burnin_len + self.unroll_len - 1
        t = self.burnin_len + self.unroll_len - 1
        lstm_states = (h_current, c_current)
        qvalue, (h, c) = q_network(self.next_states[t],
                                   states=lstm_states,
                                   prev_action=F.one_hot(self.actions[t], num_classes=self.action_space),
                                   j=self.j,
                                   prev_in_rewards=self.in_rewards[t],
                                   prev_ex_rewards=self.ex_rewards[t])
        qvalues.append(qvalue)

        # (unroll_len+1, batch_size, action_space)
        qvalues = torch.stack(qvalues, dim=0)

        return qvalues

    def get_priorities(self, qvalues, rewards):
        """
        get priorities from q values and rewards
        Args:
          qvalues        (torch.tensor): Q values from Q network [unroll_len+1, batch_size, action_space]
          rewards        (torch.tensor): rewards from experiences [burnin_len+unroll_len, batch_size]
        Returns:
          priority       (torch.tensor): priority calculate based on td errors
        """

        # (unroll_len, batch_size)
        Q = torch.sum(qvalues[:-1] * self.actions_onehot, dim=2)

        # (unroll_len, batch_size)
        next_actions = torch.argmax(qvalues[1:], dim=2)

        # (unroll_len, batch_size, action_space)
        next_actions_onehot = F.one_hot(next_actions, self.action_space)

        # (unroll_len, batch_size)
        next_maxQ = torch.sum(qvalues[1:] * next_actions_onehot, dim=2)

        # (unroll_len, batch_size)
        TQ = rewards[self.burnin_len:] + self.gamma * (1 - self.dones) * inverse_rescaling(next_maxQ)

        # (unroll_len, batch_size)
        delta = TQ - inverse_rescaling(Q)

        # (unroll_len, batch_size)
        P = transformed_retrace_operator(delta,
                                         pi=self.pi[:-1],
                                         actions=self.actions[self.burnin_len:],
                                         lamda=self.lamda,
                                         gamma=self.gamma.repeat(delta.shape[1]),
                                         unroll_len=self.unroll_len)

        td_errors = rescaling(inverse_rescaling(Q) + P) - Q
        priorities = self.eta * torch.max(torch.abs(td_errors), dim=0).values + (1 - self.eta) * torch.mean(
            torch.abs(td_errors), dim=0)

        return priorities

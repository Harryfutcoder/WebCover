import collections
import numpy as np
import torch
import torch.nn.functional as F
from playwright._impl._errors import Error

import random
from src import settings


def rescaling(x):
    """
    rescaling function
    Args:
      x (torch.tensor): input
    Returns:
      rescaled value
    """

    eps = 0.001
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.) - 1.) + eps * x


def inverse_rescaling(x):
    """
    inverse rescaling function
    Args:
      x (torch.tensor): input
    Returns:
      inverse rescaled value
    """

    eps = 0.001
    return torch.sign(x) * (
            torch.square(((torch.sqrt(1. + 4. * eps * (torch.abs(x) + 1. + eps))) - 1.) / (2. * eps)) - 1.)


def seed_evrything(seed):
    """
    set seed
    Args:
      seed (int): value to set random number
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def create_beta_list(num_arms, beta=0.3):
    """
    create beta list for each arm
    Args:
      num_arms (int): number of multi arms
    Returns:
      betas   (list): list of beta which decide weights between intrinsic qvalues and extrinsic qvalues
    NOTE: Values differ from those in the paper of Agent57.
    """

    betas = [torch.tensor(0)]
    for i in range(1, num_arms - 1):
        betas.append(beta * torch.sigmoid(torch.tensor(10 * (2 * i / (num_arms - 2) - 1))))
    betas.append(torch.tensor(beta))
    return betas


def create_gamma_list(num_arms, gamma0=0.9999, gamma1=0.997, gamma2=0.99):
    """
    create gamma list for each arm
    Args:
      num_arms (int): number of multi arms
    Returns:
      gammas  (list): list of gamma which is discount rate
    NOTE: Values differ from those in the paper of Agent57.
    """

    gammas = [torch.tensor(gamma0)]
    for i in range(1, 7):
        gammas.append(gamma0 + (gamma1 - gamma0) * torch.sigmoid(torch.tensor(10 * (i - 3) / 3)))
    gammas.append(torch.tensor(gamma1))

    for i in range(8, num_arms):
        t = (num_arms - i - 1) * torch.log(torch.tensor(1 - gamma1)) + (i - 8) * torch.log(torch.tensor(1 - gamma2))
        gammas.append(1 - torch.exp(t / (num_arms - 9)))

    return gammas


class UCB:
    """
    Determine the index of the arms in terms of solving a multi-armed bandit problem
    Attributes:
      data           : list that stores the index and average reward of the arms
      num_arms  (int): number of arms used in multi-armed bandit problem
      epsilon (float): probability to select the index of the arms used in multi-armed bandit problem
      beta    (float): weight between frequency and mean reward
      count     (int): if count is less than num_arms, index is count because of trying to pick every arm at least once
    """

    def __init__(self, num_arms, window_size, epsilon, beta):
        """
        num_arms    (int): number of arms used in multi-armed bandit problem
        window_size (int): size of window used in multi-armed bandit problem
        epsilon   (float): probability to select the index of the arms used in multi-armed bandit problem
        beta      (float): weight between frequency and mean reward
        """

        self.data = collections.deque(maxlen=window_size)
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.beta = beta
        self.count = 0

    def pull_index(self):
        """
        pull index to determine value of betas and gammas
        Returns:
          index (float): index of arms
        """

        if self.count < self.num_arms:
            index = self.count
            self.count += 1

        else:
            if random.random() > self.epsilon:
                N = np.zeros(self.num_arms)
                mu = np.zeros(self.num_arms)

                for j, reward in self.data:
                    N[j] += 1
                    mu[j] += reward
                mu = mu / (N + 1e-10)
                index = np.argmax(mu + self.beta * np.sqrt(1 / (N + 1e-6)))

            else:
                index = np.random.choice(self.num_arms)
        return index

    def push_data(self, datas):
        """
        push datas to UCB's data list
        Args:
          datas :store index of arms and resulting reward
        """

        self.data += [(j, reward) for j, reward in datas]


def transformed_retrace_operator(delta, pi, actions, gamma, unroll_len, lamda, device=torch.device("cpu")):
    """
    transform retrace operator to get retraced q values
    retrace operator is done using following recurrence formula
    P_{s, b} = \delta_{s, b} + \gamma * C_{s+1, b} * P_{s+1, b}
    where P_{s, b} = \Sigma_{j=s}^{t+H-1} \gamma^{j-s} * (\Pi_{i=s+1}^{j} C_{i, b}) * \delta_{j, b}
          C_{i, b} = \lambda * min(1, \frac{\pi(a_{i}|x_{i}^{b})}{\mu_{i}})
          \delta_{j, b} = r_{j}^{b} + \gamma * \Sigma_{a \in A} {\pi(a|x_{j+1}^{b})}*h^{-1}(Q(x_{j+1}^{b}, a))-h^{-1}(Q(x_{j}^{b}, a_{j}^{b}))
    """

    # (unroll_len, batch_size)
    P_list = delta

    # (unroll_len, batch_size)
    C = torch.where(pi == actions, torch.tensor(lamda).to(device), torch.tensor(0.).to(device))

    for t in range(unroll_len - 2, -1, -1):
        P_list[t, :] += gamma * C[t + 1, :] * P_list[t + 1, :]

    return P_list


def get_episodic_reward(x, M, k, c=0.001, epsilon=0.0001, cluster_distance=0.008, max_similarity=8):
    """
    get episodic reward based on memory that store embedding representation
    Args:
      x               (np.ndarray): embeded representation
      M                           : memory that store embedding representation
      k                      (int): number of neighbors referenced when calculating episode reward
    Returns:
      episodic reward (np.ndarray): reward based on how different from neighbors
    """

    dist_list = [np.linalg.norm((m - x), ord=2) for m in M]

    topk_dist_list = np.sort(dist_list)[:k]
    dm = np.mean(topk_dist_list)

    if dm == 0:
        return 1e-10
    else:
        topk_dist_list = topk_dist_list / dm
        topk_dist_list = np.where(topk_dist_list - cluster_distance < 0, 0, topk_dist_list - cluster_distance)

    K = epsilon / (epsilon + topk_dist_list)
    s = np.sqrt(np.sum(K)) + c

    if s > max_similarity:
        return 1e-10
    else:
        return 1 / s


def play_episode(env_in,
                 input_dim,
                 action_space,
                 j,
                 epsilon,
                 k,
                 L,
                 error_list,
                 short_ac_error_list,
                 long_ac_error_list,
                 in_q_network,
                 ex_q_network,
                 embedding_net,
                 original_lifelong_net,
                 trained_lifelong_net,
                 short_auto_encoder,
                 long_auto_encoder,
                 beta=0.3,
                 is_test=False):
    """
    play episode
    Args:
      frame_process_func       : function to preprocess images
      env          (webenv): environment
      input_dim            (int): dim of state
      action_space        (int): dim of action space
      j                        : index of arms
      epsilon           (float): coefficient for epsilon greedy
      k                   (int): number of neighbors referenced when calculating episode reward
      L                   (int): upper limit of curiosity
      error_list               : list of errors to be accommodated when calculating lifelong reward
      in_q_network             : q network about intrinsic reward
      ex_q_network             : q network about extrinsic reward
      embedding_net            : embedding network to get episodic reward
      embedding_classifier     : classify action based on embedding representation
      original_lifelong_net    : lifelong network not to be trained
      trained_lifelong_net     : lifelong network to be trained
      beta              (float): coefficient to decide weights between intrinsic qvalues and extrinsic qvalues
      is_test            (bool): flag indicating whether it is a test or not
    """

    env = env_in
    # (768,)
    state, _ = env.reset()

    in_h = ex_h = torch.zeros(1, 1, in_q_network.lstm.hidden_size).float()
    in_c = ex_c = torch.zeros(1, 1, in_q_network.lstm.hidden_size).float()
    prev_action = F.one_hot(torch.tensor([np.random.choice(action_space)]),
                            num_classes=action_space)  # (1,400) todo a random elem?
    prev_ex_reward = 0
    prev_in_reward = 0
    episode_reward = 0
    done = False

    M = collections.deque(maxlen=int(1e3))
    ucb_datas = []
    transitions = []

    timestep = 1
    env_retry_times = 0
    while not done:

        # batching (1, 768)
        state = torch.tensor(np.array([state]), dtype=torch.float)

        # intrinsic Qvalues (1, action_space)
        in_qvalue, (next_in_h, next_in_c) = in_q_network(state,
                                                         states=(in_h, in_c),
                                                         prev_action=prev_action,
                                                         j=torch.tensor([j]),
                                                         prev_ex_rewards=torch.tensor([prev_ex_reward]).float(),
                                                         prev_in_rewards=torch.tensor([prev_in_reward]).float())
        # extrinsic Qvalues (1, action_space)
        ex_qvalue, (next_ex_h, next_ex_c) = ex_q_network(state,
                                                         states=(ex_h, ex_c),
                                                         prev_action=prev_action,
                                                         j=torch.tensor([j]),
                                                         prev_ex_rewards=torch.tensor([prev_ex_reward]).float(),
                                                         prev_in_rewards=torch.tensor([prev_in_reward]).float())

        # ε-greedy
        # try:
        #     if random.random() < epsilon:
        #         print('Random Action: ')
        #         next_state, ex_reward, done, info = env.random_step()
        #     else:
        #         print('Max Value Action: ')
        #         # concat with rescaling
        #         qvalue = rescaling(inverse_rescaling(ex_qvalue) + beta * inverse_rescaling(in_qvalue))
        #         # action = np.argmax(qvalue.detach().numpy())
        #         next_state, ex_reward, done, info = env.coordinate_step(qvalue)
        # except Exception as msg:
        #     print('agent play_episode env step Failed\n')
        #     state, _ = env.update_state_and_action()
        #     continue

        # New
        try:
            action_log = ''
            if random.random() < epsilon:
                action_log += 'Random Action: '
                action_info = env.select_action_by_random()
            else:
                action_log += 'Max Value Action: '
                qvalue = rescaling(inverse_rescaling(ex_qvalue) + beta * inverse_rescaling(in_qvalue))
                action_info = env.select_action_by_grid_only(qvalue)
            settings.logger.debug(action_log + 'perform_action Page: {}'.format(str(env.web_config.page)))
            # settings.logger.debug(action_log + action_info['outerHTML'].replace("\n", '\\n'))
            next_state, ex_reward, done, info = env.step(action_info)
            env_retry_times = 0
        except Error as e:
            msg = str(e).replace("\n", '\\n')
            # settings.logger.debug('Action Selection and Perform. ' + msg)
            settings.logger.debug('Action Selection and Perform.')
            env_retry_times += 1
            if env_retry_times >= 10:
                env.web_config.reset_page()
                env_retry_times = 0
            env.update_environment(flag_error=True)
            state = env.state
            continue

        covered_points, nearest_point = info['covered_points'], info['nearest_point']
        # batching (768,)
        next_state = np.array(next_state, dtype=float)

        control_state = embedding_net(state).squeeze(0).detach().numpy()
        error = np.square(original_lifelong_net(state).detach().numpy(),
                          trained_lifelong_net(state).detach().numpy()).mean()
        short_ac_error = np.square(short_auto_encoder(state).detach().numpy(),
                                   state.numpy()).mean()
        long_ac_error = np.square(long_auto_encoder(state).detach().numpy(),
                                  state.numpy()).mean()

        if len(M) < k:
            episodic_reward = 0
            std = 1
            avg = 1

            short_ac_std = 1
            short_ac_avg = 1

            long_ac_std = 1
            long_ac_avg = 1
        else:
            episodic_reward = get_episodic_reward(control_state, M, k)
            std = np.std(error_list)
            avg = np.mean(error_list)

            short_ac_std = np.std(short_ac_error_list)
            short_ac_avg = np.mean(short_ac_error_list)

            long_ac_std = np.std(long_ac_error_list)
            long_ac_avg = np.mean(long_ac_error_list)

        curiosity = 1 + (error - avg) / (std + 1e-10)
        r_short_ac = 1 + (short_ac_error - short_ac_avg) / (short_ac_std + 1e-10)
        r_long_ac = 1 + (long_ac_error - long_ac_avg) / (long_ac_std + 1e-10)

        # push data to Memory
        M.append(control_state)
        error_list.append(error)
        short_ac_error_list.append(short_ac_error)
        long_ac_error_list.append(long_ac_error)

        r_ac = max(r_short_ac, r_long_ac)
        in_reward = episodic_reward * np.clip(r_ac, 1, L)

        current_action = torch.zeros((1, 400))
        if is_test:
            episode_reward += ex_reward
        else:

            if done:  # done==True when lose life
                for action, ratio in covered_points.items():  # action : int
                    is_nearest_point = 1 if nearest_point == action else 0

                    action_ex_reward, action_in_reward = ex_reward * ratio, in_reward * ratio
                    current_action[0][action] = 1
                    transition = (prev_ex_reward, prev_in_reward, prev_action.float(),
                                  state, action, in_h, in_c, ex_h, ex_c, j,
                                  True, action_ex_reward, action_in_reward, next_state, timestep, is_nearest_point)
                    transitions.append(transition)
            else:
                for action, ratio in covered_points.items():
                    is_nearest_point = 1 if nearest_point == action else 0

                    action_ex_reward, action_in_reward = ex_reward * ratio, in_reward * ratio
                    current_action[0][action] = 1
                    transition = (prev_ex_reward, prev_in_reward, prev_action.float(),
                                  state, action, in_h, in_c, ex_h, ex_c, j,
                                  done, action_ex_reward, action_in_reward, next_state, timestep, is_nearest_point)
                    transitions.append(transition)

        ucb_datas.append((j, ex_reward))

        in_h, in_c, ex_h, ex_c = next_in_h, next_in_c, next_ex_h, next_ex_c
        prev_action, prev_ex_reward, prev_in_reward = current_action, ex_reward, in_reward

        state = next_state
        timestep += 1
    if is_test:
        return ucb_datas, episode_reward, error_list, short_ac_error_list, long_ac_error_list
    else:
        return ucb_datas, transitions, error_list, short_ac_error_list, long_ac_error_list


def segments2contents(segments, burnin_len, is_grad=False, device=torch.device("cpu")):
    """
    convert segments to contents
    Args:
      segments        : a coherent body of experience of some length
      burnin_len (int): burnin length to calculate qvalues
    Returns:
      each content
    """

    # (burnin_len+unroll_len, batch_size, 768)
    states = torch.stack([torch.tensor(np.vstack(seg.states), requires_grad=is_grad) for seg in segments],  # todo
                         dim=1).float().to(device)

    # (burnin_len+unroll_len, batch_size)
    actions = torch.stack([torch.tensor(seg.actions) for seg in segments], dim=1).to(device)

    # (burnin_len+unroll_len, batch_size)
    ex_rewards = torch.stack([torch.tensor(seg.ex_rewards, requires_grad=is_grad) for seg in segments],
                             dim=1).float().to(device)

    # (burnin_len+unroll_len, batch_size)
    in_rewards = torch.stack([torch.tensor(seg.in_rewards, requires_grad=is_grad) for seg in segments],
                             dim=1).float().to(device)

    # (unroll_len, batch_size)
    dones = torch.stack([torch.tensor(seg.dones[burnin_len:]) for seg in segments], dim=1).float().to(device)

    # (batch_size,)
    j = torch.stack([torch.tensor(seg.j) for seg in segments], dim=0).to(device)

    # (burnin_len+unroll_len, batch_size, n_frames, 84, 84)
    next_states = torch.stack([torch.tensor(np.vstack(seg.next_states), requires_grad=is_grad) for seg in segments],
                              # todo
                              dim=1).float().to(device)

    # (1, batch_size, hidden_size)
    in_h0 = torch.cat([seg.in_h_init for seg in segments], dim=1).float().to(device)

    # (1, batch_size, hidden_size)
    in_c0 = torch.cat([seg.in_c_init for seg in segments], dim=1).float().to(device)

    # (1, batch_size, hidden_size)
    ex_h0 = torch.cat([seg.ex_h_init for seg in segments], dim=1).float().to(device)

    # (1, batch_size, hidden_size)
    ex_c0 = torch.cat([seg.ex_c_init for seg in segments], dim=1).float().to(device)

    # (batch_size)
    in_reward0 = torch.tensor([seg.prev_in_reward_init for seg in segments]).float().to(device)

    # (batch_size)
    ex_reward0 = torch.tensor([seg.prev_ex_reward_init for seg in segments]).float().to(device)

    # (burnin+unroll_len, batch_size)
    prev_in_rewards = torch.cat([in_reward0[None, :], in_rewards], dim=0)[:-1]

    # (burnin+unroll_len, batch_size)
    prev_ex_rewards = torch.cat([ex_reward0[None, :], ex_rewards], dim=0)[:-1]

    # (batch_size)
    # a0 = torch.tensor([seg.prev_a_init for seg in segments]).to(device)

    # (burnin+unroll_len, batch_size)
    # prev_actions = torch.cat([a0[None, :], actions], dim=0)[:-1]
    # (burnin+unroll_len, batch_size, action_space)
    prev_actions = torch.stack([torch.tensor(np.vstack(seg.prev_a), requires_grad=True) for seg in segments],
                               dim=1).float().to(device)

    time_step = torch.stack([torch.tensor(seg.time_step) for seg in segments], dim=1).to(device)
    is_nearest_point = torch.stack([torch.tensor(seg.is_nearest_point) for seg in segments], dim=1).to(device)

    return states, actions, ex_rewards, in_rewards, dones, j, next_states, in_h0, in_c0, ex_h0, ex_c0, prev_in_rewards, \
        prev_ex_rewards, prev_actions, time_step, is_nearest_point

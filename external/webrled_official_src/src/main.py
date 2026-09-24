import argparse
import time

from src import settings
from settings import time_limit
from src.webenv.environment import WebEnvironment
from src.drl.agent import Agent
from src.drl.buffer import SegmentReplayBuffer
from src.drl.learner import Learner
from src.drl.utils import seed_evrything


def train(args, env):
    seed_evrything(args.seed)
    learner = Learner(env=env,
                      target_update_period=args.target_update_period,
                      eta=args.eta,
                      input_dim=env.state_dim,
                      num_arms=args.num_arms,
                      lamda=args.lamda,
                      burnin_length=args.burnin_length,
                      unroll_length=args.unroll_length,
                      in_q_lr=args.in_q_lr,
                      ex_q_lr=args.ex_q_lr,
                      embed_lr=args.embed_lr,
                      lifelong_lr=args.lifelong_lr,
                      in_q_clip_grad=args.in_q_clip_grad,
                      ex_q_clip_grad=args.ex_q_clip_grad,
                      embed_clip_grad=args.embed_clip_grad,
                      lifelong_clip_grad=args.lifelong_clip_grad)
    in_q_weight, ex_q_weight, embed_weight, trained_lifelong_weight, original_lifelong_weight, \
        short_auto_encoder_weight, long_auto_encoder_weight = learner.define_network()

    agent = Agent(pid=0,
                  env=env,
                  input_dim=args.input_dim,
                  epsilon=args.epsilon_l,
                  eta=args.eta,
                  lamda=args.lamda,
                  agent_update_period=args.agent_update_period,
                  num_rollout=args.num_rollout,
                  num_arms=args.num_arms,
                  k=args.k,
                  L=args.L,
                  burnin_length=args.burnin_length,
                  unroll_length=args.unroll_length,
                  window_size=args.window_size,
                  ucb_epsilon=args.ucb_epsilon,
                  ucb_beta=args.ucb_beta,
                  original_lifelong_weight=original_lifelong_weight)
    replay_buffer = SegmentReplayBuffer(buffer_size=args.buffer_size, weight_expo=args.weight_expo)

    def rollout_once():
        return agent.sync_weights_and_rollout(in_q_weight=in_q_weight,
                                              ex_q_weight=ex_q_weight,
                                              embed_weight=embed_weight,
                                              lifelong_weight=trained_lifelong_weight,
                                              short_auto_encoder_weight=short_auto_encoder_weight,
                                              long_auto_encoder_weight=long_auto_encoder_weight)

    wip_agent = rollout_once()
    if not wip_agent[1]:
        settings.logger.info("No rollout segments collected before time limit; stopping cleanly.")
        settings.logger.info("Web auto test end")
        return

    for _ in range(args.n_agent_burnin):
        priorities, segments, _pid = wip_agent
        if not segments:
            settings.logger.info("No rollout segments collected during burn-in; stopping cleanly.")
            settings.logger.info("Web auto test end")
            return
        replay_buffer.add(priorities, segments)
        wip_agent = rollout_once()

    minibatchs = [replay_buffer.sample_minibatch(batch_size=args.batch_size) for _ in range(args.update_iter)]
    wip_learner = learner.update_network(minibatchs)

    learner_cycles = 1
    agent_cycles = 0
    n_segment_added = 0

    while learner_cycles <= args.n_learner_cycle:
        if (time.time() - env.start_time) >= time_limit:
            settings.logger.info("Reached 1 hours.")
            break

        agent_cycles += 1
        priorities, segments, _pid = wip_agent
        if not segments:
            settings.logger.info("No rollout segments collected; stopping cleanly.")
            break
        replay_buffer.add(priorities, segments)
        wip_agent = rollout_once()

        n_segment_added += len(segments)

        if wip_learner:
            in_q_weight, ex_q_weight, embed_weight, trained_lifelong_weight, short_auto_encoder_weight, \
                long_auto_encoder_weight, indices, \
                priorities, in_q_loss, ex_q_loss, embed_loss, lifelong_loss, short_autoencoder_loss, \
                long_autoencoder_loss = wip_learner
            replay_buffer.update_priority(indices, priorities)
            minibatchs = [replay_buffer.sample_minibatch(batch_size=args.batch_size) for _ in range(args.update_iter)]
            wip_learner = learner.update_network(minibatchs)

            learner_cycles += 1
            agent_cycles = 0
            n_segment_added = 0

    settings.logger.info("Web auto test end")


def main(parser):
    args = parser.parse_args()
    if args.coverage:
        assert args.appname in settings.WEB_INFO
    if args.url == '' or args.domain == '':
        if args.appname == 'other':
            print("Please enter the URL and domain")
            assert args.url != '' and args.domain != ''
        args.url = settings.WEB_INFO[args.appname]['url']
        args.domain = settings.WEB_INFO[args.appname]['domain']
    args = settings.get_parameters_from_json(args)
    settings.update_path_for_new_app(args.appname)
    env = WebEnvironment(args.appname, args.url, args.domain, args.coverage, args.state_comparator)
    settings.logger.info(str(args.seed))
    train(args, env)
    env.__del__()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WebRLED')
    parser.add_argument(
        '--appname', type=str, default='splittypie',
        help='Name of web applications.'
    )
    parser.add_argument(
        '--url', type=str, default='',
        help='Enter the URL of the target application to be tested.'
    )
    parser.add_argument(
        '--domain', type=str, default='',
        help='Please enter the domain corresponding to the URL.'
    )
    parser.add_argument(
        '--coverage', action='store_true', default=False,
        help='Collect coverage statistics for supported applications.'
    )
    main(parser)

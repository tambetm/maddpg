import argparse
import numpy as np
import os
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg_ensemble import MADDPGAgentTrainer
from maddpg.trainer.replay_buffer_ensemble import ReplayBuffer
from policy import SheldonPolicy
import tensorflow.contrib.layers as layers

def parse_args(args=None):
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    parser.add_argument("--ensemble-size", type=int, default=3, help="ensemble size")
    parser.add_argument("--ensemble-choice", choices=['episode', 'timestep'], default='episode', help="shuffle agents at each step")
    parser.add_argument("--shared", action="store_true", default=False, help="use shared model for all agents")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    parser.add_argument("--restore", action="store_true", default=False, help="restore model from checkpoint")
    # Evaluation
    parser.add_argument("--display", action="store_true", default=False, help="render environment")
    parser.add_argument("--benchmark", action="store_true", default=False, help="run evaluation")
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    parser.add_argument("--save-replay", action="store_true", default=False, help="save replay memory contents along with benchmark data")
    parser.add_argument("--deterministic", action="store_true", default=False, help="use deterministic policy during benchmarking")
    parser.add_argument("--num-sheldons", type=int, default=0)
    parser.add_argument("--sheldon-ids", type=int, nargs='+')
    parser.add_argument("--sheldon-targets", type=int, nargs='+')
    return parser.parse_known_args(args)[0]

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        hidden1 = layers.fully_connected(input, num_outputs=num_units, activation_fn=tf.nn.relu)
        hidden2 = layers.fully_connected(hidden1, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(hidden2, num_outputs=num_outputs, activation_fn=None)
        return out, hidden1, hidden2

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        ensemble = []
        for j in range(arglist.ensemble_size):
            ensemble.append(trainer(
                "bad" if arglist.shared else "agent_%d_%d" % (i, j),
                model, obs_shape_n, env.action_space, i, arglist,
                local_q_func=(arglist.adv_policy=='ddpg'),
                reuse=tf.AUTO_REUSE if arglist.shared else False))
        trainers.append(ensemble)
    for i in range(num_adversaries, env.n):
        ensemble = []
        for j in range(arglist.ensemble_size):
            ensemble.append(trainer(
                "good" if arglist.shared else "agent_%d_%d" % (i, j),
                model, obs_shape_n, env.action_space, i, arglist,
                local_q_func=(arglist.good_policy=='ddpg'),
                reuse=tf.AUTO_REUSE if arglist.shared else False))
        trainers.append(ensemble)
    return trainers

def mark_sheldon_agents(env, arglist):
    colors = [
        np.array([0.85, 0.35, 0.35]),
        np.array([0.35, 0.85, 0.35]),
        np.array([0.35, 0.35, 0.85])
    ]
    for i in range(arglist.num_sheldons):
        agent_id = arglist.sheldon_ids[i]
        env.world.agents[agent_id].color = colors[i]
        landmark_id = arglist.sheldon_targets[i]
        env.world.landmarks[landmark_id].color = colors[i]

def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        
        # Create experience buffer
        replay_buffer = ReplayBuffer(arglist.num_episodes * arglist.max_episode_len if arglist.benchmark and arglist.save_replay else 1e6)
        min_replay_buffer_len = arglist.batch_size * arglist.max_episode_len

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        mark_sheldon_agents(env, arglist)
        if arglist.display:
            env.render()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        # pick random agent from ensemble for each episode
        if arglist.ensemble_choice == 'episode':
            agent_ids = np.random.randint(arglist.ensemble_size, size=len(trainers))
            agents = [trainers[i][agent_id] for i, agent_id in enumerate(agent_ids)]

        print('Starting iterations...')
        while True:
            # pick random agent from ensemble for each timestep
            if arglist.ensemble_choice == 'timestep':
                agent_ids = np.random.randint(arglist.ensemble_size, size=len(trainers))
                agents = [trainers[i][agent_id] for i, agent_id in enumerate(agent_ids)]
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(agents,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            replay_buffer.add(obs_n, action_n, rew_n, new_obs_n, done_n, agent_ids)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            for i, info in enumerate(info_n):
                agent_info[-1][i].append(info_n['n'])

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()

            if done or terminal:
                obs_n = env.reset()
                mark_sheldon_agents(env, arglist)
                if arglist.display:
                    env.render()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])
                # pick random agent from ensemble for each episode
                if arglist.ensemble_choice == 'episode':
                    agent_ids = np.random.randint(arglist.ensemble_size, size=len(trainers))
                    agents = [trainers[i][agent_id] for i, agent_id in enumerate(agent_ids)]

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                if train_step >= arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                        if arglist.save_replay:
                            pickle.dump(replay_buffer._storage, fp)
                    break
                continue

            # update all trainers, if not in display or benchmark mode
            # only update every 100 steps and if replay buffer is large enough
            if train_step % 100 == 0 and len(replay_buffer) >= min_replay_buffer_len:
                for i, ensemble in enumerate(trainers):
                    for agent in ensemble:
                        # sample different batch for each agent in ensemble
                        batch_obs_n, batch_act_n, batch_rew_n, batch_obs_next_n, batch_done_n, batch_agent_ids = replay_buffer.sample(arglist.batch_size)
                        batch_obs_n = [batch_obs_n[:, j] for j in range(batch_obs_n.shape[1])]
                        batch_act_n = [batch_act_n[:, j] for j in range(batch_act_n.shape[1])]
                        batch_obs_next_n = [batch_obs_next_n[:, j] for j in range(batch_obs_next_n.shape[1])]
                        # choose random agent from ensemble for target action
                        batch_agents = [random.choice(ensemble) for ensemble in trainers]
                        loss = agent.update(batch_agents, batch_obs_n, batch_act_n, batch_rew_n[:, i], batch_obs_next_n, batch_done_n[:, i])

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)

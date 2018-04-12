import pickle
import numpy as np
import argparse
import os.path

from train import parse_args, make_env, get_trainers
import maddpg.common.tf_util as U
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("--policy_file")
parser.add_argument("--output_file")
parser.add_argument("--num-units", type=int, default=128)
args = parser.parse_args()

with open(args.input_file, "rb") as fp:
    data = pickle.load(fp)
    agent1 = pickle.load(fp)
    agent2 = pickle.load(fp)
    agent3 = pickle.load(fp)

print("Number of episodes:", len(data))
#print("Something something:", len(data[0]))
#print("First episode length:", len(data[0][0]))
#print("Number of agents:", len(data[0][0][0]))
#print("Performance data items:", len(data[0][0][0][0]))
#print()

assert len(data[0]) == 1            # dummy
assert len(data[0][0]) == 25        # episode length
assert len(data[0][0][0]) == 3      # num agents
assert len(data[0][0][0][0]) == 4   # performance data items

assert len(agent1) == len(agent2) == len(agent3) == len(data) * 25
assert len(agent1[0]) == len(agent2[0]) == len(agent3[0]) == 5

states1 = np.array([agent1[i][0] for i in range(len(agent1))]).reshape((-1, 25, len(agent1[0][0])))
states2 = np.array([agent2[i][0] for i in range(len(agent2))]).reshape((-1, 25, len(agent2[0][0])))
states3 = np.array([agent3[i][0] for i in range(len(agent3))]).reshape((-1, 25, len(agent3[0][0])))
#print("State shapes:", states1.shape, states2.shape, states3.shape)

actions1 = np.array([agent1[i][1] for i in range(len(agent1))]).reshape((-1, 25, len(agent1[0][1])))
actions2 = np.array([agent2[i][1] for i in range(len(agent2))]).reshape((-1, 25, len(agent2[0][1])))
actions3 = np.array([agent3[i][1] for i in range(len(agent3))]).reshape((-1, 25, len(agent3[0][1])))
#print("Action shapes:", actions1.shape, actions2.shape, actions3.shape)

#rewards1 = np.array([agent1[i][2] for i in range(len(agent1))]).reshape((-1, 25))
#rewards2 = np.array([agent2[i][2] for i in range(len(agent2))]).reshape((-1, 25))
#rewards3 = np.array([agent3[i][2] for i in range(len(agent3))]).reshape((-1, 25))
#print("Reward shapes:", rewards1.shape, rewards2.shape, rewards3.shape)

# subtract one, because by default collision with agent itself is also counted
collisions1 = np.array([[data[i][0][t][0][1] for t in range(len(data[i][0]))] for i in range(len(data))]) - 1
collisions2 = np.array([[data[i][0][t][1][1] for t in range(len(data[i][0]))] for i in range(len(data))]) - 1
collisions3 = np.array([[data[i][0][t][2][1] for t in range(len(data[i][0]))] for i in range(len(data))]) - 1
#print("Collisions shapes:", collisions1.shape, collisions2.shape, collisions3.shape)

occupied1 = np.array([data[i][0][-1][0][3] for i in range(len(data))])
occupied2 = np.array([data[i][0][-1][1][3] for i in range(len(data))])
occupied3 = np.array([data[i][0][-1][2][3] for i in range(len(data))])
#print("Occupied shape:", occupied1.shape, occupied2.shape, occupied3.shape)
assert np.all(occupied1 == occupied2)
assert np.all(occupied1 == occupied3)

'''
print("Number of episodes, where:")
print("- all landmarks are covered", np.sum(occupied1 == 3))
print("- at least 2 landmarks covered", np.sum(occupied1 >= 2))
print("- at least 1 landmark covered", np.sum(occupied1 >= 1))
print("- zero landmarks covered", np.sum(occupied1 == 0))
print()
'''

new_states1 = np.array([agent1[i][3] for i in range(len(agent1))]).reshape((-1, 25, len(agent1[0][3])))
new_states2 = np.array([agent2[i][3] for i in range(len(agent2))]).reshape((-1, 25, len(agent2[0][3])))
new_states3 = np.array([agent3[i][3] for i in range(len(agent3))]).reshape((-1, 25, len(agent3[0][3])))
#print("New state shapes:", states1.shape, states2.shape, states3.shape)

# calculate distance of each agent from each landmark
landmarks1 = np.sqrt(np.sum(new_states1[:,-1,4:10].reshape((new_states1.shape[0], 3, 2))**2, axis=-1))
landmarks2 = np.sqrt(np.sum(new_states2[:,-1,4:10].reshape((new_states2.shape[0], 3, 2))**2, axis=-1))
landmarks3 = np.sqrt(np.sum(new_states3[:,-1,4:10].reshape((new_states3.shape[0], 3, 2))**2, axis=-1))
#print("Landmarks shapes:", landmarks1.shape, landmarks2.shape, landmarks3.shape)

occupied = np.logical_or(np.logical_or(landmarks1 < 0.1, landmarks2 < 0.1), landmarks3 < 0.1)
all_occupied = np.all(occupied, axis=-1)
assert np.all(all_occupied == (occupied1 == 3))

print("Agent1 covers some landmark:", np.sum(np.sum(landmarks1 < 0.1, axis=-1) > 0))
print("Agent2 covers some landmark:", np.sum(np.sum(landmarks2 < 0.1, axis=-1) > 0))
print("Agent3 covers some landmark:", np.sum(np.sum(landmarks3 < 0.1, axis=-1) > 0))
print("All landmarks occupied:", np.sum(all_occupied))
print()

y1 = landmarks1 < 0.1
y2 = landmarks2 < 0.1
y3 = landmarks3 < 0.1
#print("Target shapes:", y1.shape, y2.shape, y3.shape)

mask1 = np.sum(landmarks1 < 0.1, axis=-1) == 1
mask2 = np.sum(landmarks2 < 0.1, axis=-1) == 1
mask3 = np.sum(landmarks3 < 0.1, axis=-1) == 1
mask4 = (occupied1 == 3)
idx = np.logical_and(mask1, np.logical_and(mask2, np.logical_and(mask3, mask4)))
print("Number of episodes where each agent occupies exactly one landmark:", np.sum(idx))
print("Agent1 covered counts by landmark:", np.sum(y1[idx], axis=0))
print("Agent2 covered counts by landmark:", np.sum(y2[idx], axis=0))
print("Agent3 covered counts by landmark:", np.sum(y3[idx], axis=0))
print()

# divide sum of collisions by 2, because every collision is counted twice
# sum collisions per episode and take mean over episodes
print("Mean number of collisions per episode:", np.mean(np.sum((collisions1 + collisions2 + collisions3) / 2, axis=-1)))
print("Average agent distance from the closest landmark at the end:", 
      np.mean(np.min(np.stack([landmarks1, landmarks2, landmarks3]), axis=-1)))
print()

if args.policy_file and args.output_file:
    arglist = parse_args(['--benchmark', '--deterministic', '--num-units', str(args.num_units)])

    #tf.reset_default_graph()
    #tf.InteractiveSession().as_default()
    with tf.Session().as_default():
        # Create environment
        env = make_env('simple_spread', arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        #print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        os.path.splitext(args.input_file)

        print('Loading previous state...')
        U.load_state(args.policy_file)

        actions = trainers[0].act(states1[0])
        assert np.allclose(actions1[0], actions)
        actions = trainers[1].act(states2[0])
        assert np.allclose(actions2[0], actions)
        actions = trainers[2].act(states3[0])
        assert np.allclose(actions3[0], actions)

        h1_values = trainers[0].p_debug['h1_values']
        h2_values = trainers[0].p_debug['h2_values']

        X1_h1 = np.array([h1_values(states1[i]) for i in range(states1.shape[0])])
        X1_h2 = np.array([h2_values(states1[i]) for i in range(states1.shape[0])])

        #print("Agent1 hidden state shapes:", X1_h1.shape, X1_h2.shape)

        h1_values = trainers[1].p_debug['h1_values']
        h2_values = trainers[1].p_debug['h2_values']

        X2_h1 = np.array([h1_values(states2[i]) for i in range(states2.shape[0])])
        X2_h2 = np.array([h2_values(states2[i]) for i in range(states2.shape[0])])

        #print("Agent2 hidden state shapes:", X1_h1.shape, X1_h2.shape)

        h1_values = trainers[2].p_debug['h1_values']
        h2_values = trainers[2].p_debug['h2_values']

        X3_h1 = np.array([h1_values(states3[i]) for i in range(states3.shape[0])])
        X3_h2 = np.array([h2_values(states3[i]) for i in range(states3.shape[0])])

        #print("Agent3 hidden state shapes:", X1_h1.shape, X1_h2.shape)

    np.savez_compressed(args.output_file,
        X1_obs=states1, 
        X1_h1=X1_h1, 
        X1_h2=X1_h2,
        X1_act=actions1,
        X2_obs=states2, 
        X2_h1=X2_h1, 
        X2_h2=X2_h2,
        X2_act=actions2,
        X3_obs=states3, 
        X3_h1=X3_h1, 
        X3_h2=X3_h2,
        X3_act=actions3,
        y1=y1,
        y2=y2,
        y3=y3,
    )


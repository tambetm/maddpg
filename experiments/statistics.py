import pickle
import numpy as np
import argparse
import os.path

parser = argparse.ArgumentParser()
parser.add_argument("input_file")
args = parser.parse_args()

with open(args.input_file, "rb") as fp:
    data = pickle.load(fp)

assert len(data[0]) == 1            # dummy
assert len(data[0][0]) == 25        # episode length
assert len(data[0][0][0]) == 3      # num agents
assert len(data[0][0][0][0]) == 4   # performance data items

episodes = len(data)

# subtract one, because collision with agent itself is also counted
collisions1 = np.array([[data[i][0][t][0][1] for t in range(len(data[i][0]))] for i in range(len(data))]) - 1
collisions2 = np.array([[data[i][0][t][1][1] for t in range(len(data[i][0]))] for i in range(len(data))]) - 1
collisions3 = np.array([[data[i][0][t][2][1] for t in range(len(data[i][0]))] for i in range(len(data))]) - 1
#print("Collisions shapes:", collisions1.shape, collisions2.shape, collisions3.shape)

# divide sum of collisions by 2, because every collision is counted twice
# sum collisions per episode and take mean over episodes
mean_collisions = np.mean(np.sum((collisions1 + collisions2 + collisions3) / 2, axis=-1))

min_dists1 = np.array([[data[i][0][t][0][2] for t in range(len(data[i][0]))] for i in range(len(data))])
min_dists2 = np.array([[data[i][0][t][1][2] for t in range(len(data[i][0]))] for i in range(len(data))])
min_dists3 = np.array([[data[i][0][t][2][2] for t in range(len(data[i][0]))] for i in range(len(data))])
#print("Min_dists shapes:", min_dists1.shape, min_dists2.shape, min_dists3.shape)
assert np.all(min_dists1 == min_dists2)
assert np.all(min_dists1 == min_dists3)
mean_min_dist = np.mean(min_dists1 / 3)

occupied1 = np.array([data[i][0][-1][0][3] for i in range(len(data))])
occupied2 = np.array([data[i][0][-1][1][3] for i in range(len(data))])
occupied3 = np.array([data[i][0][-1][2][3] for i in range(len(data))])
#print("Occupied shape:", occupied1.shape, occupied2.shape, occupied3.shape)
assert np.all(occupied1 == occupied2)
assert np.all(occupied1 == occupied3)

all_landmarks_covered = np.sum(occupied1 == 3)

print(args.input_file, episodes, all_landmarks_covered, float(all_landmarks_covered) / episodes, mean_collisions, mean_min_dist, sep=',')

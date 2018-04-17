import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("output_file")
args = parser.parse_args()

with open(args.input_file, "rb") as f:
    episode_rewards = pickle.load(f)

plt.plot(episode_rewards)
plt.ylabel('Min dist')
plt.xlabel('Episode')
plt.savefig(args.output_file)

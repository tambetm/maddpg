import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("output_file")
args = parser.parse_args()

acc = np.load(args.input_file)

plt.figure(figsize=(24, 18))
for i in range(3):
    for j in range(3):
        plt.subplot(3, 3, i*3 + j + 1)
        plt.plot(acc[i, j, :, :, 1])
        plt.ylim(0, 1)
        plt.ylabel('Accuracy')
        plt.xlabel('Timesteps')
        plt.title('Agent %d predicts agent %d%s final landmark' % (i + 1, j + 1, ' (own)' if i == j else ''))
        plt.legend(['observation', 'hidden 1', 'hidden 2', 'action'], loc='upper left')
plt.savefig(args.output_file)


import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse


def train_and_test_timestep(X, y, t):
    # only episodes where the other agent covers just one landmark 
    idx = np.where(np.sum(y, axis=-1) == 1)[0]
    X = X[idx, t]
    y = y[idx]

    # convert one-hot vector into indices
    y = np.argmax(y, axis=-1)

    # split into training and test episodes (NB! this is done at episode level!)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

    # normalize input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # train linear classifier
    clf = MLPClassifier(hidden_layer_sizes=(), solver='adam')
    clf.fit(X_train, y_train)

    # calculate training and test accuracy
    return clf.score(X_train, y_train), clf.score(X_test, y_test)


parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("output_file")
args = parser.parse_args()

data = np.load(args.input_file)

X1_obs = data['X1_obs']
X1_h1 = data['X1_h1']
X1_h2 = data['X1_h2']
X1_act = data['X1_act']
X2_obs = data['X2_obs']
X2_h1 = data['X2_h1']
X2_h2 = data['X2_h2']
X2_act = data['X2_act']
X3_obs = data['X3_obs']
X3_h1 = data['X3_h1']
X3_h2 = data['X3_h2']
X3_act = data['X3_act']
y1 = data['y1']
y2 = data['y2']
y3 = data['y3']

# calculate accuracies
acc = np.empty((3, 3, 25, 4, 2))
for i in range(3):
    for j in range(3):
        for t in range(25):
            print(i, j, t, end=' ')
            acc[i, j, t, 0] = train_and_test_timestep(data['X%d_obs' % (i + 1)], data['y%d' % (j + 1)], t)
            acc[i, j, t, 1] = train_and_test_timestep(data['X%d_h1' % (i + 1)], data['y%d' % (j + 1)], t)
            acc[i, j, t, 2] = train_and_test_timestep(data['X%d_h2' % (i + 1)], data['y%d' % (j + 1)], t)
            acc[i, j, t, 3] = train_and_test_timestep(data['X%d_act' % (i + 1)], data['y%d' % (j + 1)], t)
            #acc[i, j, t, 4] = train_and_test_timestep(np.random.randn(*data['X%d_h1' % (i + 1)].shape), data['y%d' % (j + 1)], t)
            print(acc[i, j, t, :, 1])

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

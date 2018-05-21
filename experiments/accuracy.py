import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import argparse
import pickle


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

# calculate accuracies
acc = np.empty((3, 3, 25, 4, 2))
for i in range(3):
    for j in range(3):
        print(i, j, end=' ')
        for t in range(25):
            print(i, j, t, end=' ')
            acc[i, j, t, 0] = train_and_test_timestep(data['X%d_obs' % (i + 1)], data['y%d' % (j + 1)], t)
            acc[i, j, t, 1] = train_and_test_timestep(data['X%d_h1' % (i + 1)], data['y%d' % (j + 1)], t)
            acc[i, j, t, 2] = train_and_test_timestep(data['X%d_h2' % (i + 1)], data['y%d' % (j + 1)], t)
            acc[i, j, t, 3] = train_and_test_timestep(data['X%d_act' % (i + 1)], data['y%d' % (j + 1)], t)
            #acc[i, j, t, 4] = train_and_test_timestep(np.random.randn(*data['X%d_h1' % (i + 1)].shape), data['y%d' % (j + 1)], t)
            print(acc[i, j, t, :, 1])

np.save(args.output_file, acc)


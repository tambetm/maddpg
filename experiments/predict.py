import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, shuffle=False)

    # normalize input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # train linear classifier
    clf = MLPClassifier(hidden_layer_sizes=(), solver='adam')
    clf.fit(X_train, y_train)

    # calculate training and test accuracy
    print(clf.score(X_train, y_train), clf.score(X_test, y_test))
    return clf


parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("output_file")
parser.add_argument("--test_size", type=int, default=100)
args = parser.parse_args()

data = np.load(args.input_file)

# calculate accuracies and predictions
models = []
for i in range(3):
    models.append([])
    for j in range(3):
        models[i].append([])
        print(i, j, end=' ')
        for t in range(25):
            print(i, j, t, end=' ')
            clf = train_and_test_timestep(data['X%d_h1' % (i + 1)], data['y%d' % (j + 1)], t)
            models[i][j].append(clf)

with open(args.output_file, "wb") as f:
    pickle.dump(models, f)

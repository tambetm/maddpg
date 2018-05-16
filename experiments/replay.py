import numpy as np
from train import make_env
from multiagent.rendering import Transform
from skvideo.io import FFmpegWriter
import pickle
import time
import argparse
import math

def draw_arrow(env, agent_pos, landmark_pos, prob):
    x1, y1 = agent_pos
    x2, y2 = landmark_pos
    dx, dy = x2 - x1, y2 - y1
    l = math.sqrt(dx**2 + dy**2)
    f = prob * 0.15 / l
    x2 = x1 + f * dx
    y2 = y1 + f * dy
    env.viewers[0].draw_line((x1, y1), (x2, y2))

def draw_bars(env, agent_pos, probs):
    x, y = agent_pos
    w = 0.05
    h = 0.1
    x1 = x - 0.075
    y1 = y - 0.075
    h1 = h * probs[0]
    x2 = x1 + w
    y2 = y1
    h2 = h * probs[1]
    x3 = x2 + w
    y3 = y2
    h3 = h * probs[2]
    env.viewers[0].draw_polygon([(x1, y1), (x1 + w, y1), (x1 + w, y1 + h1), (x1, y1 + h1)], color=[0.85, 0.35, 0.35])
    env.viewers[0].draw_polygon([(x2, y2), (x2 + w, y2), (x2 + w, y2 + h2), (x2, y2 + h2)], color=[0.35, 0.85, 0.35])
    env.viewers[0].draw_polygon([(x3, y3), (x3 + w, y3), (x3 + w, y3 + h3), (x3, y3 + h3)], color=[0.35, 0.35, 0.85])

parser = argparse.ArgumentParser()
parser.add_argument("npz")
parser.add_argument("pkl")
parser.add_argument("--agent", type=int)
parser.add_argument("--episodes", type=int, default=100)
#parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--record_video")
args = parser.parse_args()

with open(args.pkl, "rb") as f:
    models = pickle.load(f)

data = np.load(args.npz)

if args.record_video:
    writer = FFmpegWriter(args.record_video) 

y1 = data['y1']
y2 = data['y2']
y3 = data['y3']
mask1 = np.sum(y1, axis=-1) == 1
mask2 = np.sum(y2, axis=-1) == 1
mask3 = np.sum(y3, axis=-1) == 1
idx = np.logical_and(mask1, np.logical_and(mask2, mask3))

print("Number of episodes where each agent covers one landmark:", np.sum(idx))

X1_obs = data['X1_obs'][idx]
X2_obs = data['X2_obs'][idx]
X3_obs = data['X3_obs'][idx]

agent1_pos = X1_obs[:, :, 2:4]
agent2_pos = X2_obs[:, :, 2:4]
agent3_pos = X3_obs[:, :, 2:4]

landmark1_pos = X1_obs[:, :, 4:6] + agent1_pos
landmark2_pos = X1_obs[:, :, 6:8] + agent1_pos
landmark3_pos = X1_obs[:, :, 8:10] + agent1_pos

X1_h1 = data['X1_h1'][idx]
X2_h1 = data['X2_h1'][idx]
X3_h1 = data['X3_h1'][idx]

'''
for i in range(3):
    for j in range(3):
        for t in range(25):
            #idx = (np.sum(data['y%d' % (j + 1)], axis=-1) == 1)
            acc = models[i][j][t].score(data['X%d_h1' % (i + 1)][idx, t][-args.episodes:], np.argmax(data['y%d' % (j + 1)][idx][-args.episodes:], axis=-1))
            print(i, j, t, acc)
input()
'''

env = make_env('simple_spread', None, True)
env.reset()
env.world.landmarks[0].color = [0.85, 0.35, 0.35]
env.world.landmarks[1].color = [0.35, 0.85, 0.35]
env.world.landmarks[2].color = [0.35, 0.35, 0.85]
env.render()

for e in range(agent1_pos.shape[0] - args.episodes, agent1_pos.shape[0]):
    for t in range(agent1_pos.shape[1]):
        env.world.landmarks[0].state.p_pos = landmark1_pos[e, t]
        env.world.landmarks[1].state.p_pos = landmark2_pos[e, t]
        env.world.landmarks[2].state.p_pos = landmark3_pos[e, t]
        env.world.agents[0].state.p_pos = agent1_pos[e, t]
        env.world.agents[1].state.p_pos = agent2_pos[e, t]
        env.world.agents[2].state.p_pos = agent3_pos[e, t]
        if args.agent is not None:
            prob1 = models[args.agent][0][t].predict_proba([X1_h1[e, t]])[0]
            prob2 = models[args.agent][1][t].predict_proba([X2_h1[e, t]])[0]
            prob3 = models[args.agent][2][t].predict_proba([X3_h1[e, t]])[0]
            geom = env.viewers[0].draw_circle(0.15, filled=False, linewidth=2)
            if args.agent == 0:
                geom.add_attr(Transform(translation=agent1_pos[e, t]))
            elif args.agent == 1:
                geom.add_attr(Transform(translation=agent2_pos[e, t]))
            elif args.agent == 2:
                geom.add_attr(Transform(translation=agent3_pos[e, t]))
            else:
                assert False
        else:
            prob1 = models[0][0][t].predict_proba([X1_h1[e, t]])[0]
            prob2 = models[1][1][t].predict_proba([X2_h1[e, t]])[0]
            prob3 = models[2][2][t].predict_proba([X3_h1[e, t]])[0]
        #time.sleep(1./args.fps)
        '''
        draw_arrow(env, agent1_pos[e, t], landmark1_pos[e, t], prob1[0])
        draw_arrow(env, agent1_pos[e, t], landmark2_pos[e, t], prob1[1])
        draw_arrow(env, agent1_pos[e, t], landmark3_pos[e, t], prob1[2])
        draw_arrow(env, agent2_pos[e, t], landmark1_pos[e, t], prob2[0])
        draw_arrow(env, agent2_pos[e, t], landmark2_pos[e, t], prob2[1])
        draw_arrow(env, agent2_pos[e, t], landmark3_pos[e, t], prob2[2])
        draw_arrow(env, agent3_pos[e, t], landmark1_pos[e, t], prob3[0])
        draw_arrow(env, agent3_pos[e, t], landmark2_pos[e, t], prob3[1])
        draw_arrow(env, agent3_pos[e, t], landmark3_pos[e, t], prob3[2])
        '''
        draw_bars(env, agent1_pos[e, t], prob1)
        draw_bars(env, agent2_pos[e, t], prob2)
        draw_bars(env, agent3_pos[e, t], prob3)
        img = env.render(mode='rgb_array')[0]
        if args.record_video:
            for i in range((25 - t) // 2):
                writer.writeFrame(img)
        else:
            time.sleep(1 / (t + 1))
    if args.record_video:
        for i in range(25 // 2):
            writer.writeFrame(img)
    else:
        time.sleep(1)

if args.record_video:
    writer.close()

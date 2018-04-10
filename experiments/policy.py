import numpy as np
from pyglet.window import key
from multiagent.policy import Policy

# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class InteractivePolicy(Policy):
    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__()
        self.env = env
        self.agent_index = agent_index
        # hard-coded keyboard events
        self.move = [False for i in range(6)]
        self.comm = [False for i in range(env.world.dim_c)]
        # register keyboard events with this environment's window
        env.viewers[agent_index].window.on_key_press = self.key_press
        env.viewers[agent_index].window.on_key_release = self.key_release

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            u = 0
            if self.move[0]: u = 1
            if self.move[1]: u = 2
            if self.move[2]: u = 4
            if self.move[3]: u = 3
            if self.move[5]: return None
        else:
            u = np.zeros(5) # 5-d because of no-move action
            # infinite loop until some key is pressed
            while True not in self.move:
                self.env.viewers[self.agent_index].render()
            if self.move[0]: u[2] += 1.0
            if self.move[1]: u[1] += 1.0
            if self.move[3]: u[4] += 1.0
            if self.move[2]: u[3] += 1.0
            if self.move[5]: return None
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

    # keyboard event callbacks
    def key_press(self, k, mod):
        if k==key.LEFT:  self.move[0] = True
        if k==key.RIGHT: self.move[1] = True
        if k==key.UP:    self.move[2] = True
        if k==key.DOWN:  self.move[3] = True
        if k==key.SPACE: self.move[4] = True
        if k==key.ESCAPE:self.move[5] = True
    def key_release(self, k, mod):
        if k==key.LEFT:  self.move[0] = False
        if k==key.RIGHT: self.move[1] = False
        if k==key.UP:    self.move[2] = False
        if k==key.DOWN:  self.move[3] = False
        if k==key.SPACE: self.move[4] = False
        if k==key.ESCAPE:self.move[5] = False

class SheldonPolicy(Policy):
    def __init__(self, env, landmark_id):
        super(SheldonPolicy, self).__init__()
        self.env = env
        self.landmark_id = landmark_id

    def action(self, obs):
        delta_pos = obs[(4 + self.landmark_id * 2):(4 + self.landmark_id * 2 + 2)]
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            # not tested!
            u = 0
            horizontal = abs(delta_pos[0]) > abs(delta_pos[1])
            if horizontal and delta_pos[0] < 0: u = 1 # LEFT
            if horizontal and delta_pos[0] > 0: u = 2 # RIGHT
            if not horizontal and delta_pos[1] < 0: u = 3 # UP
            if not horizontal and delta_pos[1] > 0: u = 4 # DOWN
        else:
            u = np.zeros(5) # 5-d because of no-move action
            if delta_pos[0] > 0: u[1] += delta_pos[0]  # RIGHT
            if delta_pos[0] < 0: u[2] += -delta_pos[0] # LEFT
            if delta_pos[1] > 0: u[3] += delta_pos[1]  # UP
            if delta_pos[1] < 0: u[4] += -delta_pos[1] # DOWN
        #print(delta_pos, u)
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

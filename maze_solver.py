import numpy as np
import matplotlib.pyplot as plt
import json
import random

class MazeEnv:
    def __init__(self, maze):
        self.maze = maze
        self.n_rows = len(maze)
        self.n_cols = len(maze[0])
        self.start = self.get_position("S")
        self.goal = self.get_position("G")
        self.reset()

    def get_position(self, cell_type):
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.maze[r][c] == cell_type:
                    return (r, c)

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        row, col = self.agent_pos
        if action == "UP":
            row -= 1
        elif action == "DOWN":
            row += 1
        elif action == "LEFT":
            col -= 1
        elif action == "RIGHT":
            col += 1

        if 0 <= row < self.n_rows and 0 <= col < self.n_cols and self.maze[row][col] != "X":
            self.agent_pos = (row, col)
        reward = 10 if self.agent_pos == self.goal else -1
        done = self.agent_pos == self.goal
        return self.agent_pos, reward, done

    def render(self):
        grid = np.array(self.maze)
        grid[self.agent_pos] = "A"
        for row in grid:
            print(" ".join(row))
        print()

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.init_q_table()

    def init_q_table(self):
        for row in range(self.env.n_rows):
            for col in range(self.env.n_cols):
                self.q_table[(row, col)] = {"UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0}

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
        return max(self.q_table[state], key=self.q_table[state].get)

    def update(self, state, action, reward, next_state):
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        target = reward + self.gamma * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state

    def display_policy(self):
        for row in range(self.env.n_rows):
            for col in range(self.env.n_cols):
                if (row, col) == self.env.goal:
                    print("G", end=" ")
                elif self.env.maze[row][col] == "X":
                    print("X", end=" ")
                else:
                    best_action = max(self.q_table[(row, col)], key=self.q_table[(row, col)].get)
                    print(best_action[0], end=" ")
            print()

if __name__ == "__main__":
    with open("maze_config.json", "r") as f:
        maze_data = json.load(f)

    maze = maze_data["maze"]
    env = MazeEnv(maze)
    agent = QLearningAgent(env)

    print("Training the agent...")
    agent.train(episodes=500)

    print("Optimal Policy:")
    agent.display_policy()

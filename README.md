# Game AI: Reinforcement Learning in a Maze Solver

## Overview
This project focuses on developing an AI agent that learns to navigate through a maze using reinforcement learning techniques. The AI will find the shortest path to the goal by trial and error, improving its performance over time.

---

## Purpose
The primary goals of this project are:

- To understand and implement reinforcement learning algorithms like Q-Learning.
- To simulate a grid-based maze environment for training the agent.
- To visualize the agent’s learning process and final solution.

---

## Features

- **Customizable Maze:** Users can define their own maze layout.
- **Interactive Visualization:** See the agent’s learning process and pathfinding steps in real-time.
- **Learning Algorithm:** Q-Learning implementation for training the AI agent.
- **Reward System:** Positive rewards for reaching the goal and negative rewards for hitting walls or invalid moves.

---

## Technologies Used

- **Python 3.7+**
- **NumPy:** For numerical operations.
- **Matplotlib:** For visualization.
- **Optional:** Pygame for a more interactive grid-based visualization.

---

## How It Works

### Environment:

A grid-based maze where each cell is either:

- **Start Point**
- **Goal**
- **Path**
- **Obstacle**

The agent starts at a predefined position and explores the maze.

### Q-Learning:

The agent learns through a trial-and-error process, updating its Q-table based on rewards.

Formula:
\[ Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_a Q(s', a) - Q(s, a)] \]

- \( \alpha \): Learning rate
- \( \gamma \): Discount factor
- \( r \): Reward for the action

### Reward System:

- **+10** for reaching the goal.
- **-1** for invalid moves or hitting obstacles.
- **0** for valid moves.

### Visualization:

The agent’s exploration and the shortest path found are visualized step-by-step.

---

## Files Included

- **`maze_solver.py`**: Main script for the maze environment and Q-Learning algorithm.
- **`maze_config.json`**: A configuration file for defining custom maze layouts.
- **`README.md`**: Documentation for the project.
- **Optional:** A `visualizer.py` for interactive gameplay visualization using Pygame.

---

## Example Maze Layout

```json
{
  "maze": [
    ["S", "0", "0", "X", "G"],
    ["X", "X", "0", "X", "0"],
    ["0", "0", "0", "0", "0"],
    ["X", "0", "X", "X", "X"],
    ["0", "0", "0", "X", "0"]
  ]
}
```

- **S**: Start
- **G**: Goal
- **0**: Open Path
- **X**: Obstacle

---

## Output

- **Console:** Displays the agent's learning process, Q-table updates, and the optimal policy.
- **Visualization:** Matplotlib or grid-based visualization of the shortest path.

---

## Improvements & Extensions

- Add dynamic obstacles to test the agent’s adaptability.
- Implement other reinforcement learning algorithms like SARSA.
- Use a larger or randomly generated maze.
- Incorporate Pygame for better visualization and interaction.


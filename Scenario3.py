import numpy as np
import random
from FourRooms import FourRooms

fourRoomsObj = FourRooms('rgb', stochastic=True)

Q = np.zeros((13, 13, 4))

# Defining hyperparameters
alpha = 0.1 
gamma = 0.7
epsilon = 0.1 

# Training loop
num_episodes = 1000  # adjust as needed
for episode in range(num_episodes):
    # Reset environment for new episode
    fourRoomsObj.newEpoch()
    state = fourRoomsObj.getPosition()
    packages_collected = []

    # Exploration-exploitation trade-off
    while not fourRoomsObj.isTerminal():
        # Print current state
        print("Episode:", episode, "Current State:", state)

        # Choose action using epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)  
        else:
            action = np.argmax(Q[state[1], state[0]])  

        # Print chosen action
        print("Chosen Action:", action)

        # Take action and observe reward and next state
        grid_cell, new_state, packages_remaining, is_terminal = fourRoomsObj.takeAction(action)

        # Print reward and next state
        print("Reward:", grid_cell, "New State:", new_state)

        # Check if package collected
        if grid_cell in [FourRooms.RED, FourRooms.GREEN, FourRooms.BLUE]:
            packages_collected.append(grid_cell)

            # Check if packages collected in correct order
            if packages_collected != sorted(packages_collected):
                print("Wrong order of package collection. Terminating episode.")
                break

        # Update Q-value using Q-learning update rule
        old_Q = Q[state[1], state[0], action]  
        max_future_Q = np.max(Q[new_state[1], new_state[0]])  
        new_Q = old_Q + alpha * (grid_cell + gamma * max_future_Q - old_Q)
        Q[state[1], state[0], action] = new_Q 

        state = new_state

    if packages_remaining == 0:
        print("All packages collected successfully!")

fourRoomsObj.showPath(-1)

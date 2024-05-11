import numpy as np
import random
from FourRooms import FourRooms

#Initializing FourRooms Object and setting stochastic to true
fourRoomsObj = FourRooms('simple', stochastic=True)

# Define Q-table
# Initialize Q-values arbitrarily
Q = np.zeros((13, 13, 4))

#Initializing learning rate, discount factor and exploration rate
# learning rate
alpha = 0.1  
# discount factor
gamma = 0.9  
epsilon = 0.1  # exploration rate

num_episodes = 1000
for episode in range(num_episodes):
    print("Episode:", episode)
    # Reset environment for new episode to start
    fourRoomsObj.newEpoch()
    state = fourRoomsObj.getPosition()
    print("Starting state:", state)

    # Exploration-exploitation trade-off
    while not fourRoomsObj.isTerminal():
        # Choose action using epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            #Selecting a Random Action
            action = random.randint(0, 3)  
        else:
            action = np.argmax(Q[state[1], state[0]])  # adjust indices
        print("Chosen action:", action)

        # Take action and observe reward and next state
        grid_cell, new_state, _, _ = fourRoomsObj.takeAction(action)
        print("New state:", new_state, "Reward:", grid_cell)

        # Update Q-value using Q-learning update rule
        # adjust indices
        old_Q = Q[state[1], state[0], action]  
        # adjust indices
        max_future_Q = np.max(Q[new_state[1], new_state[0]])  
        new_Q = old_Q + alpha * (grid_cell + gamma * max_future_Q - old_Q)
        Q[state[1], state[0], action] = new_Q  # adjust indices

        state = new_state

# Show final path
fourRoomsObj.showPath(-1)

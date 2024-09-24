import cvxpy as cp
import numpy as np

# Define parameters
N = 10  # Number of arms
T = 10  # Time horizon
S = 10  # Number of non-dummy states (0 to 9)
S_prime = 2 * S  # Expanded state space with dummy states
A = 2  # Action space {0, 1}
K = 1  # Activation budget (number of arms to activate at each step)

## Be careful, reward needs to be masked in our setting
potential_reward =np.random.uniform(1, 10, N)
print(f"The list of reward is {potential_reward}")
time_step = 0.01

def sample_mu_values(num_samples):
    # Sample mus
    mus = np.random.uniform(0, 1, num_samples) * 10 
    return list(mus)

def sample_lambda_values(num_samples):
    # Sample mus
    lambdas = np.random.uniform(0, 1, num_samples) * 10 
    return list(lambdas)

mu_lst = sample_mu_values(10)
lambda_lst = sample_lambda_values(10)

## This generates the normal MDP process
def generate_transition_matrix_with_dummy_states(num_arms, num_states, num_dummy_states, mu_lst, lambda_lst, time_step):
    # Initialize transition matrices
    P = np.zeros((num_arms, num_states + num_dummy_states, 2, num_states + num_dummy_states))
    K = num_states - 1

    for n in range(num_arms):
        for s in range(num_states):
            if s == 0:
                # Boundary condition for state 0 (active)
                P[n, s, 1, s + num_states] = 1

                P[n, s, 0, s + 1] = lambda_lst[n] * (K - s) * time_step
                P[n, s, 0, s] = 1 - (lambda_lst[n] * (K - s) * time_step)
            if s == num_states - 1:
                # Boundary condition for state 9 (passive)
                P[n, s, 0, s] = 1

                P[n, s, 1, s - 1 + num_states] = mu_lst[n] * s * time_step
                P[n, s, 1, s + num_states] = 1 - mu_lst[n] * s * time_step

            else:
                P[n, s, 0, s + 1] = lambda_lst[n] * (K - s) * time_step
                P[n, s, 0, s] = 1 - lambda_lst[n] * (K - s) * time_step
                P[n, s, 1, s - 1 + num_states] = mu_lst[n] * s * time_step
                P[n, s, 1, s + num_states] = 1 - mu_lst[n] * s * time_step
        # Transitions within dummy states (s in S_a)
        for s_a in range(num_states, num_states + num_dummy_states):
            normal_state_index = s_a - num_states
            # Copy transitions from the corresponding normal state for action 0 within dummy states
            P[n, s_a, 0, num_states:num_states + num_dummy_states] = P[n, normal_state_index, 0, 0:num_states]
            P[n, s_a, 1, num_states:num_states + num_dummy_states] = P[n, normal_state_index, 0, 0:num_states]

    return P


# Generate the transition matrix
P = generate_transition_matrix_with_dummy_states(N, S, S, mu_lst, lambda_lst, time_step)

# Flatten the optimization variables
mu = cp.Variable(N * S_prime * A * T, nonneg=True)


# Helper function to get the index in the flattened mu
def mu_index(n, s, a, t):
    return n * S_prime * A * T + s * A * T + a * T + t

# Define objective function
# Only receive reward when action is active
# consider the reward rate as well
objective = cp.Maximize(cp.sum([mu[mu_index(n, s, a, t)] * potential_reward[n] * a * s * time_step
                                for n in range(N)
                                for s in range(S)
                                for a in range(A)
                                for t in range(T)]) +
                        cp.sum([mu[mu_index(n, s_dummy, a, t)] * potential_reward[n] * a * (s_dummy - S) * time_step
                                for n in range(N)
                                for s_dummy in range(S, S_prime)
                                for a in range(A)
                                for t in range(T)]))


# Define constraints
constraints = []

# Activation budget constraint
for t in range(T):
    constraints.append(cp.sum([mu[mu_index(n, s, 1, t)]
                               for n in range(N)
                               for s in range(S_prime)]) <= K)
    

# Flow constraints for occupancy measure
for n in range(N):
    for t in range(1, T):
        for s in range(S_prime):
            constraints.append(cp.sum([mu[mu_index(n, s, a, t)] for a in range(A)]) ==
                               cp.sum([mu[mu_index(n, s_prime, a_prime, t - 1)] * P[n, s_prime, a_prime, s]
                                       for s_prime in range(S_prime)
                                       for a_prime in range(A)]))
            

# Initial distribution constraint (assuming uniform distribution here)
for n in range(N):
    for s in range(S):
        constraints.append(cp.sum([mu[mu_index(n, s, a, 0)] for a in range(A)]) == 1 / S)
        constraints.append(cp.sum([mu[mu_index(n, s + S, a, 0)] for a in range(A)]) == 0)

# Solve the problem
problem = cp.Problem(objective, constraints)
result = problem.solve(solver=cp.SCS)  # Try using SCS or another solver


# Check solver status
print(f"Solver status: {problem.status}")
if problem.status in ["infeasible", "unbounded"]:
    print("The problem is infeasible or unbounded.")
else:
    # Display the optimal value
    print(f"Optimal value: {result}")

    # Recover and display the optimal policy
    optimal_mu = mu.value
    if optimal_mu is not None:
        optimal_mu_4D = optimal_mu.reshape((N, S_prime, A, T))

optimal_reward = result

## Extract all the available arms
def extract_activated_arms(activated_arms):
    # Use list comprehension to extract the first element from each tuple
    arms = [t[0] for t in activated_arms]
    return arms


##  =======================================  Fluid Algorithm ===============================================

# Monte Carlo simulation using the new setting
num_simulations = 20000
lp_total_rewards = []
lp_pulls_in_dummy_states = 0 
for sim in range(num_simulations):
    lp_total_reward = 0
    arm_states = np.random.choice(range(S), N, p=[1 / S] * S)

    for t in range(T):
        # Precompute indices
        current_indices = []
        for n in range(N):
            current_state = arm_states[n]
            mu_1 = optimal_mu_4D[n, current_state, 1, t]
            mu_0 = optimal_mu_4D[n, current_state, 0, t]
            index_value = mu_1 / (mu_1 + mu_0) if (mu_1 + mu_0) > 0 else 0
            current_indices.append((n, current_state, index_value))

        current_indices.sort(key=lambda x: x[2], reverse=True)
        activated_arms = current_indices[:K]

        activated_arm_lst = extract_activated_arms(activated_arms)

        for n in activated_arm_lst:
            if (arm_states[n] > S):
                lp_total_reward += potential_reward[n] * (arm_states[n] - S) * time_step
            else:
                lp_total_reward += potential_reward[n] * arm_states[n] * time_step

        for arm in activated_arms:
            n, current_state, _ = arm
            if current_state >= S:  # Dummy states are indexed after normal states
                lp_pulls_in_dummy_states += 1
            next_state = np.random.choice(range(S_prime), p=P[n, current_state, 1, :])
            arm_states[n] = next_state

        for n in range(N):
            if n not in [arm[0] for arm in activated_arms]:
                current_state = arm_states[n]
                next_state = np.random.choice(range(S_prime), p=P[n, current_state, 0, :])
                arm_states[n] = next_state
        
    lp_total_rewards.append(lp_total_reward)

lp_average_reward = np.mean(lp_total_rewards)
print(f"Average total reward over {num_simulations} simulations: {lp_average_reward}")
# Print counts of pulls in dummy states
print(f"Number of pulls in dummy states (LP-based policy): {lp_pulls_in_dummy_states}")


# =============================== Random Policy Benchmark ==============================

# Results storage for random policy
random_total_rewards = []
random_pulls_in_dummy_states = 0
# Monte Carlo simulations for random policy
for sim in range(num_simulations):
    # Initialize the total reward for this simulation
    random_total_reward = 0

    # Initialize the state of each arm randomly in one of the normal states
    arm_states = np.random.choice(range(S), N, p=[1/S] * S)  # Starting states for each arm
    #arm_states = np.ones(N,dtype=int)  # Starting states for each arm
    # Simulate the process for the entire time horizon T
    for t in range(T):
        # Random policy: randomly select K arms to activate
        random_arms = np.random.choice(range(N), K, replace=False)

        for n in random_arms:
            if (arm_states[n] > S):
                random_total_reward += potential_reward[n] * (arm_states[n] - S) * time_step
            else:
                random_total_reward += potential_reward[n] * arm_states[n] * time_step
        #Update states for random policy
        for n in random_arms:
            current_state = arm_states[n]
            if current_state >= S:  # Dummy states are indexed after normal states
                random_pulls_in_dummy_states += 1
            # Move to the next state based on the transition probability for action 1
            next_state = np.random.choice(range(S_prime), p=P[n, current_state, 1, :])
            arm_states[n] = next_state  # Update the arm's state
        
        # For non-activated arms in random policy, transition based on action 0
        for n in range(N):
            if n not in random_arms:
                current_state = arm_states[n]
                # Move to the next state based on the transition probability for action 0
                next_state = np.random.choice(range(S_prime), p=P[n, current_state, 0, :])
                arm_states[n] = next_state  # Update the arm's state
    # Store the total rewards for this simulation
    random_total_rewards.append(random_total_reward)


# Calculate average total rewards for random policy
average_random_total_reward = np.mean(random_total_rewards)
print(f"Average total achieved value over {num_simulations} simulations (Random policy): {average_random_total_reward}")



# ==============================  Simulate Whittle index =======================================

num_simulations = 20000

def calculate_single_whittle_index(cur_state, n_state, cur_mu, cur_lambda, cur_reward):
    whittle_index = 0
    K = n_state - 1
    whittle_index = (cur_reward / (cur_mu * K)) * (cur_mu * (cur_state ** 2) - cur_lambda * ((K - cur_state) ** 2))

    return whittle_index

def calculate_whittle_index(n_arms, n_states, mu_lst, lambda_lst, reward_lst):
    whittle_indices = np.zeros((n_arms, n_states))
    K = n_states - 1
    for arm in range(n_arms):
        for state in range(n_states):
            cur_mu = mu_lst[arm]
            cur_lambda = lambda_lst[arm]
            cur_reward = reward_lst[arm]
            
            whittle_indices[arm][state] = \
                calculate_single_whittle_index(state, n_states, cur_mu, cur_lambda, cur_reward)
            
            # print(f"arm in {arm} and state in {state} yield index {whittle_indices[arm][state]}")
    
    return whittle_indices


# Calculate Whittle index for normal states
whittle_indices = calculate_whittle_index(N, S, mu_lst, lambda_lst, potential_reward)

# Set Whittle index for dummy states to zero
whittle_indices_full = np.zeros((N, S_prime))
whittle_indices_full[:, :S] = whittle_indices  # Set normal state Whittle indices
whittle_indices_full[:, S:] = 0  # Set dummy state Whittle indices to zero

# Output the Whittle indices
print("Whittle Indices (including dummy states):")
#print(whittle_indices_full)

whittle_total_rewards = []
whittle_pulls_in_dummy_states = 0

for sim in range(num_simulations):
    whittle_total_reward = 0
    # Initialize the state of each arm in one of the normal states
    arm_states = np.random.choice(range(S), N, p=[1/S] * S)

    for t in range(T):
        # Get the current Whittle index values for each arm at the current states
        current_indices = []
        for n in range(N):
            current_state = arm_states[n]
            whittle_index_value = whittle_indices_full[n, current_state]
            current_indices.append((n, current_state, whittle_index_value))

        # Sort arms by their Whittle index values in descending order
        current_indices.sort(key=lambda x: x[2], reverse=True)

        # Select the top K arms to activate (i.e., pull)
        activated_arms = current_indices[:K]

        activated_arm_lst = extract_activated_arms(activated_arms)

        for n in activated_arm_lst:
            if (arm_states[n] > S):
                whittle_total_reward += potential_reward[n] * (arm_states[n] - S) * time_step
            else:
                whittle_total_reward += potential_reward[n] * arm_states[n] * time_step

        # Update the states of activated arms based on action 1
        for arm in activated_arms:
            n, current_state, _ = arm
            if current_state >= S:  # Dummy states are indexed after normal states
                whittle_pulls_in_dummy_states += 1
            next_state = np.random.choice(range(S_prime), p=P[n, current_state, 1, :])
            arm_states[n] = next_state

        # Update the states of non-activated arms based on action 0
        for n in range(N):
            if n not in [arm[0] for arm in activated_arms]:
                current_state = arm_states[n]
                next_state = np.random.choice(range(S_prime), p=P[n, current_state, 0, :])
                arm_states[n] = next_state

    # Store the total rewards for this simulation
    whittle_total_rewards.append(whittle_total_reward)
# Calculate the average total rewards for Whittle index policy
average_whittle_total_reward = np.mean(whittle_total_rewards)
print(f"Average total achieved value over {num_simulations} simulations (Whittle index policy): {average_whittle_total_reward}")
print(f"Whittle: Number of pulls in dummy states: {whittle_pulls_in_dummy_states}")


import matplotlib.pyplot as plt

def plot_dictionary_data(data, N, T, B, type):
    """
    Generate a bar plot from a dictionary with unique colors and a legend.
    
    :param data: dict, a dictionary with key-value pairs where keys are labels and values are numeric.
    """
    # Define a list of colors; ensure there are enough colors for each bar
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
    
    # Check if the dictionary has more elements than colors available
    if len(data) > len(colors):
        print("Warning: Not enough colors specified. Some bars will reuse colors.")
        # Extend the color list by repeating the initial list enough times
        colors *= (len(data) // len(colors) + 1)

    # Create a bar plot with different colors for each bar
    plt.figure(figsize=(10, 6))
    bars = plt.bar(data.keys(), data.values(), color=colors[:len(data)])  # Slice colors to match number of keys

    # Adding titles and labels
    plt.title(f'N = {N}, T = {T}, B = {B}, type = {type}')
    plt.xlabel('Method')
    plt.ylabel('Average System Reward')

    # Optional: Add value labels on top of each bar
    for key, value in zip(data.keys(), data.values()):
        plt.text(key, value, f'{value}', ha='center', va='bottom')

    # Show the plot
    
    #plt.show()
    plt.savefig(f'{type},N={N},T={T},B={B}.png')


experiment_synthetic_result = {
    "Optimal": optimal_reward,
    "fluid": lp_average_reward,
    "whittle": average_whittle_total_reward,
    "random": average_random_total_reward
}

plot_dictionary_data(experiment_synthetic_result, N, T, K, "Enhrenfest")







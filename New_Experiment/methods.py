import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

'''
    Calculate the theoretical upper bound and the result for the fluid algorithm under the 
    given input parameters

    Input:
        - N: number of groups
        - T: time horizon
        - S: number of states
        - S_prime: number of states + dummy states
        - A: number of actions
        - K: budget in each round
        - group_member_num: number of members within a group
        - P: transition_probability_matrix
        - r: an array of reward for each state
        - num_simulations: number of simulations
    
    Output:
        - optimal_reward: the theoretical upper bound
        - average_reward: the result of the fluid algorithm
'''
def fluid_policy(N, T, S, S_prime, A, K, group_member_num, P, r, num_simulations):

    # Flatten the optimization variables
    mu = cp.Variable(N * S_prime * A * T, nonneg=True)


    # Helper function to get the index in the flattened mu
    def mu_index(n, s, a, t):
        return n * S_prime * A * T + s * A * T + a * T + t


    # Define objective function
    objective = cp.Maximize(cp.sum([mu[mu_index(n, s, a, t)] * r[s]
                                for n in range(N)
                                for s in range(S_prime)
                                for a in range(A)
                                for t in range(T)]))

    # Define constraints
    constraints = []

    # Activation budget constraint
    for t in range(T):
        constraints.append(cp.sum([mu[mu_index(n, s, 1, t)]
                               for n in range(N)
                               for s in range(S_prime)]) <= K/group_member_num)

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
    
    '''
    Important: Since every group is scaled by the same number, we can solve the LP on the smaller system. 
    However, when we aggregate the reward, we need to scale the reward back.
    '''
    optimal_reward = result * group_member_num

    # Monte Carlo simulation using the new setting
    total_rewards = []
    lp_pulls_in_dummy_states = 0

    for sim in range(num_simulations):
        total_reward = 0
        arm_states = {}
        for type in range(N):
            arm_states[type] = np.random.choice(range(S), group_member_num, p=[1 / S] * S)

        for t in range(T):
            # Precompute indices
            current_indices = []
            for type in range(N):
                for mem in range(group_member_num):
                    current_state = arm_states[type][mem]
                    mu_1 = optimal_mu_4D[type, current_state, 1, t]
                    mu_0 = optimal_mu_4D[type, current_state, 0, t]
                    index_value = (S - current_state) * mu_1 / (mu_1 + mu_0) if (mu_1 + mu_0) > 0 else -10000
                    if (index_value < 0):
                        index_value = 0
                    #index_value = mu_1 / (mu_1 + mu_0) if (mu_1 + mu_0) > 0 else -10000
                    current_indices.append((type, mem, current_state, index_value))

            current_indices.sort(key=lambda x: x[3], reverse=True)
            activated_arms = current_indices[:K]

            # Collect rewards and transition
            for type in range(N):
                for mem in range(group_member_num):
                    total_reward += r[arm_states[type][mem]]

            for arm in activated_arms:
                type, mem, current_state, _ = arm
                if current_state >= S:  # Dummy states are indexed after normal states
                    #print(f"type is {type}, mem is {mem}, current_state is {current_state}")
                    lp_pulls_in_dummy_states += 1
                next_state = np.random.choice(range(S_prime), p=P[type, current_state, 1, :])
                arm_states[type][mem] = next_state
        
            for type in range(N):
                for mem in range(group_member_num):
                    if (type, mem) not in [(arm[0], arm[1]) for arm in activated_arms]:
                        current_state = arm_states[type][mem]
                        next_state = np.random.choice(range(S_prime), p=P[type, current_state, 0, :])
                        arm_states[type][mem] = next_state
        total_rewards.append(total_reward)
    average_reward = np.mean(total_rewards)
    fluid_std = np.std(total_rewards, ddof=1)

    print(f"Average total reward over {num_simulations} simulations: {average_reward}, stdev is {fluid_std}")
    # Print counts of pulls in dummy states
    print(f"Number of pulls in dummy states (LP-based policy): {lp_pulls_in_dummy_states}")

    return optimal_reward, average_reward, fluid_std


def draw_non_zero_indices(arr, num_samples):
    #print(f"length of arr is {len(arr)}, number of sample is {num_samples}")
    # Find the indices where the array values are not 0
    non_zero_indices = np.where(arr != 0)[0]
    
    # Randomly select 'num_samples' indices from the non-zero indices
    selected_indices = np.random.choice(non_zero_indices, size=num_samples, replace=False)
    
    return selected_indices

'''
    Calculate the reward for the random policy

    Input:
        - N: number of groups
        - T: time horizon
        - S: number of states
        - S_prime: number of states + dummy states
        - A: number of actions
        - K: budget in each round
        - group_member_num: number of members within a group
        - P: transition_probability_matrix
        - r: an array of reward for each state
        - num_simulations: number of simulations
    
    Output:
        - average_random_total_reward: the average reward for each simulation
'''
def random_policy(N, T, S, S_prime, A, K, group_member_num, P, r, num_simulations):
    random_total_rewards = []
    random_pulls_in_dummy_states = 0

    ## Total number of arms
    num_arms = N * group_member_num

    # Monte Carlo simulations for random policy
    for sim in range(num_simulations):
        # Initialize the total reward for this simulation
        random_total_reward = 0 
       
        activation_mask = np.ones(num_arms)

        arm_states = {}
        for type in range(N):
            arm_states[type] = np.random.choice(range(S), group_member_num, p=[1 / S] * S)

        # Simulate the process for the entire time horizon T
        for t in range(T):
            # Random policy: randomly select K arms to activate
            # random_arms = np.random.choice(range(num_arms), K, replace=False)
            random_arms = draw_non_zero_indices(activation_mask, K)
            activated_arms = []

            for ele in random_arms:
                type, mem = divmod(ele, group_member_num)
                current_state = arm_states[type][mem]
                arm = (type, mem, current_state)
                activated_arms.append(arm)
                activation_mask[ele] = 0
                
            # Collect rewards for random policy for all arms before they transition
            for type in range(N):
                for mem in range(group_member_num):
                    current_state = arm_states[type][mem]
                    random_total_reward += r[current_state]  # Reward is simply the state value

            # Update states for random policy
            for arm in activated_arms:
                type, mem, current_state = arm
                current_state = arm_states[type][mem]
                if current_state >= S:  # Dummy states are indexed after normal states
                    random_pulls_in_dummy_states += 1
                # Move to the next state based on the transition probability for action 1
                next_state = np.random.choice(range(S_prime), p=P[type, current_state, 1, :])
                arm_states[type][mem] = next_state  # Update the arm's state

            # For non-activated arms in random policy, transition based on action 0
            for type in range(N):
                for mem in range(group_member_num):
                    if (type, mem) not in [(arm[0], arm[1]) for arm in activated_arms]:
                        current_state = arm_states[type][mem]
                        next_state = np.random.choice(range(S_prime), p=P[type, current_state, 0, :])
                        arm_states[type][mem] = next_state

        # Store the total rewards for this simulation
        random_total_rewards.append(random_total_reward)

    # Calculate average total rewards for random policy
    average_random_total_reward = np.mean(random_total_rewards)
    random_std = np.std(random_total_rewards, ddof=1)
    print(f"Average total achieved value over {num_simulations} simulations (Random policy): {average_random_total_reward}, stdev is {random_std}")
    return average_random_total_reward, random_std

'''
    Calculate the reward for the no pull policy

    Input:
        - N: number of groups
        - T: time horizon
        - S: number of states
        - S_prime: number of states + dummy states
        - A: number of actions
        - K: budget in each round
        - group_member_num: number of members within a group
        - P: transition_probability_matrix
        - r: an array of reward for each state
        - num_simulations: number of simulations
    
    Output:
        - average_no_pull_total_reward: the average reward for each simulation
'''
def no_pull_policy(N, T, S, S_prime, A, K, group_member_num, P, r, num_simulations):
    # Results storage for no pull policy
    no_pull_total_rewards = []

    #no_pulls_in_dummy_states = 0

    # Monte Carlo simulations for random policy
    for sim in range(num_simulations):
        # Initialize the total reward for this simulation
        no_pull_total_reward = 0

        arm_states = {}
        for type in range(N):
            arm_states[type] = np.random.choice(range(S), group_member_num, p=[1 / S] * S)
        
        # Simulate the process for the entire time horizon T
        for t in range(T):
            # Collect rewards for no pull policy for all arms before they transition
            for type in range(N):
                for mem in range(group_member_num):
                    current_state = arm_states[type][mem]
                    no_pull_total_reward += r[current_state]  # Reward is simply the state value

            # Since you don't pull any arm, transition based on action 0
            for type in range(N):
                for mem in range(group_member_num):
                    current_state = arm_states[type][mem]
                    next_state = np.random.choice(range(S_prime), p=P[type, current_state, 0, :])
                    arm_states[type][mem] = next_state
         
        # Store the total rewards for this simulation
        no_pull_total_rewards.append(no_pull_total_reward)

    # Calculate average total rewards for no pull policy
    average_no_pull_total_reward = np.mean(no_pull_total_rewards)
    no_pull_std = np.std(no_pull_total_rewards, ddof=1)
    print(f"Average total achieved value over {num_simulations} simulations (No-Pull policy): {average_no_pull_total_reward}, stdev is {no_pull_std}")
    return average_no_pull_total_reward, no_pull_std


'''
    Calculate the Q difference method

    Input:
        - rewards: the reward list
        - transition_probabilities: transition probabilities for a single group
        - states: list of states
        - T: time horizon

    Output:
        - Q_difference_indices: a matrix of size [T, S]
            Q_difference_indices[t][s]: 
            given a group, its whittle index in state s at time t

'''
def Q_difference_index(rewards, transition_probabilities, states, T):
    #Initialize Q-values: Now indexed by both state and time
    Q_s_1 = {t: {s: 0 for s in states} for t in range(T+1)}  # Q(s,1) for action 1 at each time step
    Q_s_0 = {t: {s: 0 for s in states} for t in range(T+1)}  # Q(s,0) for action 0 at each time step

    Q_difference_indices = {t: {s: 0 for s in states} for t in range(T+1)}  # Q(s,1) for action 1 at each time step
    for s in states:
        Q_difference_indices[T][s] = 0
        Q_s_1[T][s] = rewards[s]  # Final reward for action 1 (active)
        Q_s_0[T][s] = rewards[s]  # Final reward for action 0 (passive) with zero subsidity

    # Move backward from time T-1 to 0
    for t in range(T-1, -1, -1):  # Work backward from time T-1 to 0
        for s in states:
            # Calculate Q(s, 1) for the active action
            future_reward_1 = sum(
                transition_probabilities[s, 1, next_state] * max(Q_s_1[t+1][next_state], Q_s_0[t+1][next_state])
                for next_state in states
            )
            Q_s_1[t][s] = rewards[s] + future_reward_1

            # Calculate Q(s, 0) for the passive action, including lambda_value subsidy
            future_reward_0 = sum(
                transition_probabilities[s, 0, next_state] * max(Q_s_1[t+1][next_state], Q_s_0[t+1][next_state])
                for next_state in states
            )
            Q_s_0[t][s] = rewards[s] + future_reward_0

            Q_difference_indices[t][s] = Q_s_1[t][s] - Q_s_0[t][s]

    return Q_difference_indices

"""
    Transforms a dictionary with T keys and S values under each key into a T by S matrix.
    
    Parameters:
    - dictionary: A dictionary where each key (time step) maps to another dictionary of S values (states)
    
    Returns:
    - matrix: A numpy array of shape (T, S) representing the dictionary as a matrix
"""
def dict_to_matrix(dictionary):
    # Extract time steps (T) and states (S) from the dictionary
    T = len(dictionary)
    S = len(next(iter(dictionary.values())))  # Number of states (S) from the first time step
    
    # Initialize an empty matrix of size T by S
    matrix = np.zeros((T, S))
    
    # Fill the matrix with values from the dictionary
    for t, states in dictionary.items():
        for s, value in states.items():
            matrix[t, s] = value
    
    return matrix

''' Calculate the Q difference for all types 
    
    Input:
        - N: number of groups
        - rewards: list of rewards under each state
        - transition_probabilities:
        - S: number of states
        - T: number of horizon

    Output:
        - Q_difference_all_arm: a matrix of size (N, T, S)
        Q_difference_all_arm[n, t, s]:
        for group n, the finite whittle index in state s at time t

'''
def calculate_all_arm_Q_difference(N,rewards,transition_probabilities,S,T):
    Q_difference_all_arm = np.zeros((N,T + 1,S))
    states = list(range(S))
    
    for arm in range(N):
        Q_difference_all_arm[arm]= dict_to_matrix(Q_difference_index(rewards, transition_probabilities[arm], states, T))
    return Q_difference_all_arm

''' 
    Calculate the reward for Q difference policy

    Input:
        - N: number of groups
        - T: time horizon
        - S: number of states
        - S_prime: number of states + dummy states
        - A: number of actions
        - K: budget in each round
        - group_member_num: number of members within a group
        - P: transition_probability_matrix
        - r: an array of reward for each state
        - num_simulations: number of simulations
    
    Output:
        - average_general_Q_difference_total_reward: average reward for Q difference policy

'''
def Q_difference_policy(N, T, S, S_prime, A, K, group_member_num, P, r, num_simulations):
    # Calculate Whittle index for normal states
    general_Q_difference_indices = calculate_all_arm_Q_difference(N, r, P, S_prime, T)

    # Output the Whittle indices
    print("Q difference (including dummy states):")

    general_Q_difference_total_rewards = []
    #general_whittle_pulls_in_dummy_states = 0

    for sim in range(num_simulations):
        Q_difference_reward = 0
        # Initialize the state of each arm in one of the normal states
        arm_states = {}
        for type in range(N):
            arm_states[type] = np.random.choice(range(S), group_member_num, p=[1 / S] * S)

        for t in range(T):
            # Get the current Whittle index values for each arm at the current states
            current_indices = []
            for type in range(N):
                for mem in range(group_member_num):
                    current_state = arm_states[type][mem]
                    Q_difference_value = general_Q_difference_indices[type, t, current_state]
                    current_indices.append((type, mem, current_state, Q_difference_value))

            # Sort arms by their Whittle index values in descending order
            current_indices.sort(key=lambda x: x[3], reverse=True)

            # Select the top K arms to activate (i.e., pull)
            activated_arms = current_indices[:K]

            # Collect rewards for all arms before transition
            for type in range(N):
                for mem in range(group_member_num):
                    Q_difference_reward += r[arm_states[type][mem]]  # Reward is based on the current state

            # Change the state for activated arms
            for arm in activated_arms:
                type, mem, current_state, _ = arm
                #if current_state >= S:  # Dummy states are indexed after normal states
                next_state = np.random.choice(range(S_prime), p=P[type, current_state, 1, :])
                arm_states[type][mem] = next_state
        
            for type in range(N):
                for mem in range(group_member_num):
                    if (type, mem) not in [(arm[0], arm[1]) for arm in activated_arms]:
                        current_state = arm_states[type][mem]
                        next_state = np.random.choice(range(S_prime), p=P[type, current_state, 0, :])
                        arm_states[type][mem] = next_state

        # Store the total rewards for this simulation
        general_Q_difference_total_rewards.append(Q_difference_reward)

    # Calculate the average total rewards for Whittle index policy
    average_general_Q_difference_total_rewards = np.mean(general_Q_difference_total_rewards)
    Q_difference_std = np.std(general_Q_difference_total_rewards, ddof=1)
    print(f"Average total achieved value over {num_simulations} simulations (General Q difference index policy): {average_general_Q_difference_total_rewards}, stdev is {Q_difference_std}")
    return average_general_Q_difference_total_rewards,  Q_difference_std 



def value_iteration(rewards, transition_probabilities, discount_factor, states, lambda_value, max_iterations=10000,
                    delta=1e-6):
    """
    Parameters:
- rewards: A dictionary where the key is the state and the value is the immediate reward for that state
- transition_probabilities: A dictionary where the key is the state and the value is another dictionary representing the probabilities of transitioning to other states from this state
- discount_factor: The discount factor γ, 0 < γ < 1
- states: A list of states
- lambda_value: For action 0, the subsidy added each time
- max_iterations: The maximum number of iterations
- delta: The threshold for determining convergence

Returns:
- Q_s_1: The Q-value for each state when taking action 1
- Q_s_0: The Q-value for each state when taking action 0, including the lambda subsidy

    """

    Q_s_1 = {s: 0 for s in states}  
    Q_s_0 = {s: 0 for s in states}  

    for iteration in range(max_iterations):
        Q_s_1_new = Q_s_1.copy()
        Q_s_0_new = Q_s_0.copy()

        # Iteratively update Q-values
        for s in states:
            # calculate Q(s, 1)
            future_reward_1 = sum(
                transition_probabilities[s,1,next_state] * max(Q_s_1[next_state], Q_s_0[next_state])
                for next_state in states
            )
            Q_s_1_new[s] = rewards[s] + discount_factor * future_reward_1

            # calculate Q(s, 0)，including lambda
            future_reward_0 = sum(
                transition_probabilities[s,0,next_state] * max(Q_s_1[next_state], Q_s_0[next_state])
                for next_state in states
            )
            Q_s_0_new[s] = rewards[s] + lambda_value + discount_factor * future_reward_0

        # check whether converge
        if max(abs(Q_s_1_new[s] - Q_s_1[s]) for s in states) < delta and max(
                abs(Q_s_0_new[s] - Q_s_0[s]) for s in states) < delta:
            break

        Q_s_1 = Q_s_1_new
        Q_s_0 = Q_s_0_new
    
    #print(iteration,max(abs(Q_s_1_new[s] - Q_s_1[s]) for s in states),max(abs(Q_s_0_new[s] - Q_s_0[s]) for s in states))
    return Q_s_1, Q_s_0

def calculate_whittle_index_with_dp(rewards, transition_probabilities, discount_factor, S, max_iterations=50000,
                                    delta=1e-6):
    whittle_indices = np.zeros(S)
    states=list(range(S))

    for state in states:
        lambda_value = 1  
        lambda_step = 0.1  
        iteration = 0
        while iteration < max_iterations:
           #print(iteration)
            # use dp to calculate Q(s, 1) 和 Q(s, 0)
            Q_s_1, Q_s_0 = value_iteration(rewards, transition_probabilities, discount_factor, states, lambda_value)

            #print(
            #    f"State {state}, Iteration {iteration}: Q(s, 1) = {Q_s_1[state]}, Q(s, 0) = {Q_s_0[state]}, Lambda = {lambda_value}")

            # check if Q(s, 1) = Q(s, 0)
            if abs(Q_s_1[state] - Q_s_0[state]) < delta:
                break

            # adjust lambda based on Q value
            if Q_s_1[state] > Q_s_0[state]:
                lambda_value += lambda_step
            else:
                lambda_value -= lambda_step

            # adjust step size
            lambda_step *= 0.9
            iteration += 1
        whittle_indices[state] = lambda_value

    return whittle_indices

# use binary search to find lambda
def calculate_whittle_index_with_dp_binary_search(rewards, transition_probabilities, discount_factor, S, max_iterations=10000,
                                    delta=1e-6):

    whittle_indices = np.zeros(S)
    states = list(range(S))

    for state in states:
        lambda_low = -100  # lower bound
        lambda_high = 100  # upper bound
        iteration = 0

        while iteration < max_iterations:
            
            lambda_value = (lambda_low + lambda_high) / 2

            # use dp to calculate Q(s, 1) and Q(s, 0)
            Q_s_1, Q_s_0 = value_iteration(rewards, transition_probabilities, discount_factor, states, lambda_value)
            #print(iteration,lambda_low,lambda_high,Q_s_1[state],Q_s_0[state])
            # check if Q(s, 1) = Q(s, 0) 
            if abs(Q_s_1[state] - Q_s_0[state]) < delta:
                break

            # adjust lambda
            if Q_s_1[state] > Q_s_0[state]:
                lambda_low = lambda_value  # lower bound
            else:
                lambda_high = lambda_value  # upper bound

            iteration += 1

        #print(f"State {state}, Iteration {iteration}, Lambda = {lambda_value}, Difference = {abs(Q_s_1[state] - Q_s_0[state])}")

        whittle_indices[state] = lambda_value

    return whittle_indices


def calculate_all_arm_whittle(N,rewards,transition_probabilities,S,discount_factor=0.9):
    whittle_all_arm=np.zeros((N,S))
    
    for arm in range(N):
        whittle_all_arm[arm]=calculate_whittle_index_with_dp_binary_search(rewards,transition_probabilities[arm],discount_factor,S)
    return whittle_all_arm

''' 
    Calculate the reward for infinite whittle policy

    Input:
        - N: number of groups
        - T: time horizon
        - S: number of states
        - S_prime: number of states + dummy states
        - A: number of actions
        - K: budget in each round
        - group_member_num: number of members within a group
        - P: transition_probability_matrix
        - r: an array of reward for each state
        - num_simulations: number of simulations
    
    Output:
        - average_general_whittle_total_reward : average reward for infinite whittle policy

'''
def infinite_whittle_policy(N, T, S, S_prime, A, K, group_member_num, P, r, num_simulations):
    # Calculate Whittle index for normal states
    general_whittle_indices = calculate_all_arm_whittle(N, r, P, S_prime)

    # Output the Whittle indices
    print("Whittle Indices (including dummy states):")

    general_whittle_total_rewards = []
    #general_whittle_pulls_in_dummy_states = 0

    for sim in range(num_simulations):
        whittle_total_reward = 0
        # Initialize the state of each arm in one of the normal states
        arm_states = {}
        for type in range(N):
            arm_states[type] = np.random.choice(range(S), group_member_num, p=[1 / S] * S)

        for t in range(T):
            # Get the current Whittle index values for each arm at the current states
            current_indices = []
            for type in range(N):
                for mem in range(group_member_num):
                    current_state = arm_states[type][mem]
                    whittle_index_value = general_whittle_indices[type, current_state]
                    current_indices.append((type, mem, current_state, whittle_index_value))

            # Sort arms by their Whittle index values in descending order
            current_indices.sort(key=lambda x: x[3], reverse=True)

            # Select the top K arms to activate (i.e., pull)
            activated_arms = current_indices[:K]

            # Collect rewards for all arms before transition
            for type in range(N):
                for mem in range(group_member_num):
                    whittle_total_reward += r[arm_states[type][mem]]  # Reward is based on the current state

            # Change the state for activated arms
            for arm in activated_arms:
                type, mem, current_state, _ = arm
                #if current_state >= S:  # Dummy states are indexed after normal states
                next_state = np.random.choice(range(S_prime), p=P[type, current_state, 1, :])
                arm_states[type][mem] = next_state
        
            for type in range(N):
                for mem in range(group_member_num):
                    if (type, mem) not in [(arm[0], arm[1]) for arm in activated_arms]:
                        current_state = arm_states[type][mem]
                        next_state = np.random.choice(range(S_prime), p=P[type, current_state, 0, :])
                        arm_states[type][mem] = next_state
        
        # Store the total rewards for this simulation
        general_whittle_total_rewards.append(whittle_total_reward)
        
    # Calculate the average total rewards for Whittle index policy
    average_general_whittle_total_reward = np.mean(general_whittle_total_rewards)
    general_whittle_std = np.std(general_whittle_total_rewards, ddof=1)
    print(f"Average total achieved value over {num_simulations} simulations (General Infinite Whittle index policy): {average_general_whittle_total_reward}, stdev is {general_whittle_std}")
    return average_general_whittle_total_reward, general_whittle_std


''' 
    Calculate the reward for original whittle policy (with one pull constraint)

    Input:
        - N: number of groups
        - T: time horizon
        - S: number of states
        - A: number of actions
        - K: budget in each round
        - group_member_num: number of members within a group
        - P: transition_probability_matrix
        - r: an array of reward for each state
        - num_simulations: number of simulations
    
    Output:
        - average_original_whittle_total_reward : average reward for original whittle policy
        - original_whittle_std: standard deviation for MC simulations

'''
def original_whittle_policy(N, T, S, A, K, group_member_num, P, r, num_simulations):
    # Calculate Whittle index for normal states
    original_whittle_indices = calculate_all_arm_whittle(N, r, P, S)

    # Output the Whittle indices
    print("Original Whittle Indices (including dummy states):")

    original_whittle_total_rewards = []

    for sim in range(num_simulations):
        original_whittle_total_reward = 0

        activation_mask = {}
        for type in range(N):
            activation_mask[type] = np.ones(group_member_num)

        # Initialize the state of each arm in one of the normal states
        arm_states = {}
        for type in range(N):
            arm_states[type] = np.random.choice(range(S), group_member_num, p=[1 / S] * S)

        for t in range(T):
            # Get the current Whittle index values for each arm at the current states
            current_indices = []
            for type in range(N):
                for mem in range(group_member_num):
                    current_state = arm_states[type][mem]
                    if activation_mask[type][mem] == 0:
                        whittle_index_value = -np.inf ## make sure that it will not be selected
                    else:
                        whittle_index_value = original_whittle_indices[type, current_state]
                    current_indices.append((type, mem, current_state, whittle_index_value))

            # Sort arms by their Whittle index values in descending order
            current_indices.sort(key=lambda x: x[3], reverse=True)

            # Select the top K arms to activate (i.e., pull)
            activated_arms = current_indices[:K]

            # Collect rewards for all arms before transition
            for type in range(N):
                for mem in range(group_member_num):
                    original_whittle_total_reward += r[arm_states[type][mem]]  # Reward is based on the current state

            # Change the state for activated arms
            for arm in activated_arms:
                type, mem, current_state, _ = arm
                next_state = np.random.choice(range(S), p=P[type, current_state, 1, :])
                arm_states[type][mem] = next_state
                activation_mask[type][mem] = 0
        
            for type in range(N):
                for mem in range(group_member_num):
                    if (type, mem) not in [(arm[0], arm[1]) for arm in activated_arms]:
                        current_state = arm_states[type][mem]
                        next_state = np.random.choice(range(S), p=P[type, current_state, 0, :])
                        arm_states[type][mem] = next_state
        
        # Store the total rewards for this simulation
        original_whittle_total_rewards.append(original_whittle_total_reward)
    
    # Calculate the average total rewards for Whittle index policy
    average_original_whittle_total_reward = np.mean(original_whittle_total_rewards)
    original_whittle_std = np.std(original_whittle_total_rewards, ddof=1)
    print(f"Average total achieved value over {num_simulations} simulations (General Infinite Whittle index policy): {average_original_whittle_total_reward}, stdev is {original_whittle_std}")
    return average_original_whittle_total_reward, original_whittle_std



'''
    Calculate the finite version of the whittle index
    Input:
        - lambda_value: 
        - states: a list of all the states 
        - rewards: reward for each state
        - transition_probabilities: matrix, P[s, a, next_s]
        - t: the starting timestamp
        - T: the time horizon

    Output:
        - Given the current lambda value, what is the finite Q value 
            for each time step after t (including t)

'''
def finite_value_calculation(lambda_value, states, rewards, transition_probabilities, t, T):
    ## Initialize the dictionaries, Q_s_1[t][s]: corresponding Q value
    Q_s_1 = {}
    Q_s_0 = {}

    for time in range(t, T):
        Q_s_1[time] = {state: 0 for state in states}
        Q_s_0[time] = {state: 0 for state in states}

    for time in range(T-1, t-1, -1):
        for s in states:
            if (time == T-1):
                ## According to our formulation, the last time step you collect reward
                ## is T-1
                future_reward_1 = 0
                future_reward_0 = 0
            else:
                ## If not at the last time step, collect reward according to 
                ## transition probability
                future_reward_1 = sum(
                    transition_probabilities[s, 1, next_state] * max(Q_s_1[time+1][next_state], Q_s_0[time+1][next_state])
                    for next_state in states
                )

                future_reward_0 = sum(
                    transition_probabilities[s, 0, next_state] * max(Q_s_1[time+1][next_state], Q_s_0[time+1][next_state])
                    for next_state in states
                )
           
            Q_s_1[time][s] = rewards[s] + future_reward_1
            Q_s_0[time][s] = rewards[s] + future_reward_0 + lambda_value
    
    return Q_s_1, Q_s_0

# use binary search to find lambda, Note in the finite version, we need to find an index for each time step
def calculate_finite_whittle_with_dp_binary_search(rewards, transition_probabilities, S, T, max_iterations=100,
                                    delta=1e-6):

    finite_whittle_indices = {}
    for time in range(T):
        finite_whittle_indices[time] = np.zeros(S)
    states = list(range(S))


    for t in range(T-1, -1, -1):
        for state in states:
            lambda_low = -100  # lower bound
            lambda_high = 100  # upper bound
            iteration = 0

            while iteration < max_iterations:
                
                lambda_value = (lambda_low + lambda_high) / 2

                # use dp to calculate Q(s, 1) and Q(s, 0)
                Q_s_1, Q_s_0 = finite_value_calculation(lambda_value, states, rewards, transition_probabilities, t, T)
                
                # check if Q_s_1(t, s) = Q_s_0(t, s) 
                if abs(Q_s_1[t][state] - Q_s_0[t][state]) < delta:
                    break

                # adjust lambda
                if Q_s_1[t][state] > Q_s_0[t][state]:
                    lambda_low = lambda_value  # lower bound
                else:
                    lambda_high = lambda_value  # upper bound

                iteration += 1

            finite_whittle_indices[t][state] = lambda_value

    return finite_whittle_indices


def calculate_all_arm_finite_whittle(N,rewards,transition_probabilities,S,T):
    finite_whittle_all_arm = np.zeros((N,T,S))
    
    for arm in range(N):
        finite_whittle_indices =  calculate_finite_whittle_with_dp_binary_search(rewards, transition_probabilities[arm], S, T)
        ## convert the dictionary into a T by S matrix
        finite_whittle_matrix = np.vstack(list(finite_whittle_indices.values()))
        finite_whittle_all_arm[arm] = finite_whittle_matrix
    return finite_whittle_all_arm


''' 
    Calculate the reward for finite whittle policy

    Input:
        - N: number of groups
        - T: time horizon
        - S: number of states
        - S_prime: number of states + dummy states
        - A: number of actions
        - K: budget in each round
        - group_member_num: number of members within a group
        - P: transition_probability_matrix
        - r: an array of reward for each state
        - num_simulations: number of simulations
    
    Output:
        - average_finite_whittle_total_reward: average reward for finite whittle policy

'''
def finite_whittle_policy(N, T, S, S_prime, A, K, group_member_num, P, r, num_simulations):
    # Calculate Whittle index for normal states
    general_finite_whittle_indices = calculate_all_arm_finite_whittle(N,r,P,S_prime,T)

    # Output the Whittle indices
    print("Finite whittle (including dummy states):")

    general_finite_whittle_total_rewards = []
    #general_whittle_pulls_in_dummy_states = 0

    for sim in range(num_simulations):
        finite_whittle_reward = 0
        # Initialize the state of each arm in one of the normal states
        arm_states = {}
        for type in range(N):
            arm_states[type] = np.random.choice(range(S), group_member_num, p=[1 / S] * S)

        for t in range(T):
            # Get the current Whittle index values for each arm at the current states
            current_indices = []
            for type in range(N):
                for mem in range(group_member_num):
                    current_state = arm_states[type][mem]
                    finite_whittle_value = general_finite_whittle_indices[type, t, current_state]
                    current_indices.append((type, mem, current_state, finite_whittle_value))

            # Sort arms by their Whittle index values in descending order
            current_indices.sort(key=lambda x: x[3], reverse=True)

            # Select the top K arms to activate (i.e., pull)
            activated_arms = current_indices[:K]

            # Collect rewards for all arms before transition
            for type in range(N):
                for mem in range(group_member_num):
                    finite_whittle_reward += r[arm_states[type][mem]]  # Reward is based on the current state

            # Change the state for activated arms
            for arm in activated_arms:
                type, mem, current_state, _ = arm
                #if current_state >= S:  # Dummy states are indexed after normal states
                next_state = np.random.choice(range(S_prime), p=P[type, current_state, 1, :])
                arm_states[type][mem] = next_state
        
            for type in range(N):
                for mem in range(group_member_num):
                    if (type, mem) not in [(arm[0], arm[1]) for arm in activated_arms]:
                        current_state = arm_states[type][mem]
                        next_state = np.random.choice(range(S_prime), p=P[type, current_state, 0, :])
                        arm_states[type][mem] = next_state

        # Store the total rewards for this simulation
        general_finite_whittle_total_rewards.append(finite_whittle_reward)

    # Calculate the average total rewards for Whittle index policy
    average_general_finite_whittle_total_rewards = np.mean(general_finite_whittle_total_rewards)
    finite_whittle_std = np.std(general_finite_whittle_total_rewards, ddof=1)
    print(f"Average total achieved value over {num_simulations} simulations (General Finite Whittle index policy): {average_general_finite_whittle_total_rewards}, stdev is {finite_whittle_std}")
    return average_general_finite_whittle_total_rewards, finite_whittle_std

def plot_methods_with_confidence(data, N, T, K, group_mem, num_simulations, type, lower_bound_method, show_error = False):
    """
    Plot a bar chart where the x-axis represents different methods, the y-axis represents the values.
    Error bars represent the 95% confidence interval calculated from the standard deviation, 
    except for the 'Optimal' method which doesn't have an error bar.

    :param data: dict, keys are method names and values are tuples (value, standard_deviation or None).
    :param num_simulations: int, the number of simulations that were used to compute the mean and standard deviation.
    """
    methods = []
    values = []
    errors = []

    for method, (value, std_dev) in data.items():
        if (method == 'Optimal'):
            upper_bound = value
        if (method == lower_bound_method):
            lower_bound = value
           
    # Populate lists, skipping error for methods with None as standard deviation
    for method, (value, std_dev) in data.items():
        methods.append(method)
        adjust_value = (value - lower_bound) * (1/ (upper_bound-lower_bound))
        values.append(adjust_value)
        if std_dev is not None:
            # Calculate the error if standard deviation is provided
            error = 1.96 * (std_dev / np.sqrt(num_simulations))
            errors.append(error)
        else:
            # Append 0 where there is no standard deviation (will not display an error bar)
            errors.append(0)

    # Set up the colors for the bars
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'yellow', 'pink', 'brown', 'gray', 'lime', 'magenta']
    if len(methods) > len(colors):
        colors *= (len(methods) // len(colors) + 1)  # Ensure there are enough colors

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    if (show_error):
        bars = plt.bar(methods, values, color=colors[:len(methods)], yerr=errors, capsize=5, error_kw={'elinewidth':1.5, 'ecolor':'black'})
    else:
        bars = plt.bar(methods, values, color=colors[:len(methods)], capsize=5)
        
    
    # Adding titles and labels
    plt.title(f'N = {N}, T = {T}, B = {K}, group_size = {group_mem}, type = {type}')
    plt.xlabel('Method')
    plt.ylabel('Value')
    
    # Optional: Add value labels on top of each bar
    for bar, err in zip(bars, errors):
        yval = bar.get_height()
        # if err > 0:  # Only display the value if there's an error bar
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'N={N},T={T},B={K},group_size={group_mem},type={type}.png')
import numpy as np

def finite_whittle_index(rewards, transition_probabilities, states, T):
    #Initialize Q-values: Now indexed by both state and time
    Q_s_1 = {t: {s: 0 for s in states} for t in range(T+1)}  # Q(s,1) for action 1 at each time step
    Q_s_0 = {t: {s: 0 for s in states} for t in range(T+1)}  # Q(s,0) for action 0 at each time step

    finite_whittle_indices = {t: {s: 0 for s in states} for t in range(T+1)}  # Q(s,1) for action 1 at each time step
    for s in states:
        finite_whittle_indices[T][s] = 0
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

            finite_whittle_indices[t][s] = Q_s_1[t][s] - Q_s_0[t][s]

    return finite_whittle_indices

def dict_to_matrix(dictionary):
    """
    Transforms a dictionary with T keys and S values under each key into a T by S matrix.
    
    Parameters:
    - dictionary: A dictionary where each key (time step) maps to another dictionary of S values (states)
    
    Returns:
    - matrix: A numpy array of shape (T, S) representing the dictionary as a matrix
    """
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

def calculate_all_arm_finite_whittle(N,rewards,transition_probabilities,S,T):
    whittle_all_arm = np.zeros((N,T + 1,S))
    states = list(range(S))
    
    for arm in range(N):
        whittle_all_arm[arm]= dict_to_matrix(finite_whittle_index(rewards, transition_probabilities[arm], states, T))
    return whittle_all_arm
        
# states = [0, 1, 2]
# rewards = [2,1,0]

# transition_probabilities = np.array([[
#         [[ 0, 0.7 , 0.3],
#         [ 0.2, 0.6, 0.2]],

#        [[ 0,  0.5,  0.5],
#         [ 0.1, 0.6, 0.3]],

#        [[0, 0.2, 0.8],
#         [0.2, 0.5, 0.3]]],

#         [
#         [[ 0.2,0.6,0.2],
#         [ 0.2, 0.6, 0.2]],

#        [[ 0.1,  0.6,  0.3],
#         [ 0.1, 0.6, 0.3]],

#        [[0.2, 0.5, 0.3],
#         [0.2, 0.5, 0.3]]]]
#         )


# #whittle_indices = calculate_whittle_index_with_dp(rewards, transition_probabilities, discount_factor, states)
# whittle_indices = calculate_all_arm_finite_whittle(2,rewards,transition_probabilities, 3, 5)
# whittle_indices


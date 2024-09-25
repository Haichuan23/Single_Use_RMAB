import numpy as np


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
                lambda_low = lambda_value  # 调整下界
            else:
                lambda_high = lambda_value  # 调整上界

            iteration += 1

        #print(f"State {state}, Iteration {iteration}, Lambda = {lambda_value}, Difference = {abs(Q_s_1[state] - Q_s_0[state])}")

        whittle_indices[state] = lambda_value

    return whittle_indices


def calculate_all_arm_whittle(N,rewards,transition_probabilities,S,discount_factor=0.9):
    whittle_all_arm=np.zeros((N,S))
    
    for arm in range(N):
        whittle_all_arm[arm]=calculate_whittle_index_with_dp_binary_search(rewards,transition_probabilities[arm],discount_factor,S)
    return whittle_all_arm
        


# example
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

# discount_factor = 0.9

# #whittle_indices = calculate_whittle_index_with_dp(rewards, transition_probabilities, discount_factor, states)
# whittle_indices = calculate_all_arm_whittle(2,rewards, transition_probabilities, 3)

#print("Whittle :", whittle_indices)
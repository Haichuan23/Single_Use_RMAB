import cvxpy as cp
import numpy as np


### We consider an Ehrenfest project 
### State space: x = 0, 1, 2, ... , K
### The reward rate is cx in the active phase, 0 in the passive phase
### In the active phase, x -> x-1 with intensity mu \cdot x
### In the passive phase, x -> x+1 with intensity lambda \cdot (K-x)

'''
The formula for whittle index is:
v(x) = (c / (mu * K)) * (mu * (x ** 2) - lambda * ((K - x) ** 2))
'''
def calculate_whittle_index(current_state_lst, n_state, mu_lst, lambda_lst, potential_reward_lst, one_pull_mask_lst):
    num_arms = len(current_state_lst)
    whittle_lst = [0] * num_arms

    assert len(mu_lst) == len(lambda_lst) == len(potential_reward_lst) == len(one_pull_mask_lst) == num_arms, \
    "List lengths or number of arms do not match"

    for i in range(num_arms):
        cur_x = current_state_lst[i]  ## The state of the ith arm
        cur_mu = mu_lst[i]
        cur_lambda = lambda_lst[i]
        potential_reward = potential_reward_lst[i]
        K = n_state - 1
      
        if (one_pull_mask_lst[i] == 0):
            cur_whittle = 0
        else:
            cur_whittle = (potential_reward / (cur_mu * K)) * (cur_mu * (cur_x ** 2) - cur_lambda * ((K - cur_x) ** 2))
        whittle_lst[i] = cur_whittle

    return whittle_lst

'''
    Activated_arms are a list of tuple elements: (type, mem, current_state, index_value)
    Now I want to extract only the first two elements: (type, mem), so that I can identify 
    each arm.
'''
def extract_activated_arms(activated_arms):
    # Use list comprehension to extract the first element from each tuple
    arms = [(t[0], t[1]) for t in activated_arms]
    return arms



## =================================================== fluid algorithm ===========================================================

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
        - num_simulations: number of simulations
        - potential_reward_lst: the reward rate if it's pulled
        - time_step: used to discretize the setting

    
    Output:
        - optimal_reward: the theoretical upper bound
        - lp_average_reward: the result of the fluid algorith
'''

def fluid_policy(N, T, S, S_prime, A, K, group_member_num, P, \
                 num_simulations, potential_reward_lst, time_step):
    
    # Flatten the optimization variables
    mu = cp.Variable(N * S_prime * A * T, nonneg=True)

    # Helper function to get the index in the flattened mu
    def mu_index(n, s, a, t):
        return n * S_prime * A * T + s * A * T + a * T + t
    
    # Define objective function
    # Only receive reward when action is active
    # consider the reward rate as well
    objective = cp.Maximize(cp.sum([mu[mu_index(n, s, a, t)] * potential_reward_lst[n] * a * s * time_step
                                    for n in range(N)
                                    for s in range(S)
                                    for a in range(A)
                                    for t in range(T)]) +
                            cp.sum([mu[mu_index(n, s_dummy, a, t)] * potential_reward_lst[n] * a * (s_dummy - S) * time_step
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
    lp_total_rewards = []
    lp_pulls_in_dummy_states = 0

    for sim in range(num_simulations):
        lp_total_reward = 0
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
                    index_value = (potential_reward_lst[type] * time_step) * mu_1 / (mu_1 + mu_0) if (mu_1 + mu_0) > 0 else -10000
                    current_indices.append((type, mem, current_state, index_value))

            current_indices.sort(key=lambda x: x[3], reverse=True)
            activated_arms = current_indices[:K]

            activated_arm_lst = extract_activated_arms(activated_arms)

            for arm in activated_arm_lst:
                type, mem = arm
                current_state = arm_states[type][mem]
                if (current_state > S):
                    ## In the extended state space
                    lp_total_reward += potential_reward_lst[type] * (current_state - S) * time_step
                else:
                    ## In the original state space
                    lp_total_reward += potential_reward_lst[type] * current_state * time_step
            
            for arm in activated_arms:
                type, mem, current_state, _ = arm
                # if (current_state >= S):
                #     lp_pulls_in_dummy_states += 1
                next_state = np.random.choice(range(S_prime), p=P[type, current_state, 1, :])
                arm_states[type][mem] = next_state
        
            for type in range(N):
                for mem in range(group_member_num):
                    if (type, mem) not in [(arm[0], arm[1]) for arm in activated_arms]:
                        current_state = arm_states[type][mem]
                        next_state = np.random.choice(range(S_prime), p=P[type, current_state, 0, :])
                        arm_states[type][mem] = next_state
            
        lp_total_rewards.append(lp_total_reward)
    lp_average_reward = np.mean(lp_total_rewards)
    fluid_std = np.std(lp_total_rewards, ddof=1)

    print(f"Average total reward over {num_simulations} simulations: {lp_average_reward}, stdev is {fluid_std}")
    # Print counts of pulls in dummy states
    print(f"Number of pulls in dummy states (LP-based policy): {lp_pulls_in_dummy_states}")

    return optimal_reward, lp_average_reward, fluid_std


## ======================================================= Random Policy Benchmark ======================================================


def draw_non_zero_indices(arr, num_samples):
    non_zero_indices = np.where(arr != 0)[0]
    
    if (len(non_zero_indices) < num_samples):
        ## Not enough arms to be pulled due to single use constraint
        ## Pull all of them then
        return non_zero_indices
    
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
        - num_simulations: number of simulations
        - potential_reward_lst: the reward rate if it's pulled
        - time_step: used to discretize the setting

    
    Output:
        - optimal_reward: the theoretical upper bound
        - lp_average_reward: the result of the fluid algorith
'''

def random_policy(N, T, S, S_prime, A, K, group_member_num, P, \
                 num_simulations, potential_reward_lst, time_step):
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
                    if (current_state > S):
                        random_total_reward += potential_reward_lst[type] * (current_state - S) * time_step
                    else:
                        random_total_reward += potential_reward_lst[type] * current_state * time_step

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
        



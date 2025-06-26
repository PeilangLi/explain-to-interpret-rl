import numpy as np
from collections import defaultdict
from multiprocessing import Manager, Process
import copy
import itertools
import torch as th
import math
from utils import F_not_i, tqdm_label

class Characteristics:
    """
    Calculates the characteristic values for different shapley values:
        - Local SVERL
        - Global SVERL
        - Shapley applied to value function
        - Shapley applied to policy
    """

    def __init__(self, env, states_to_explain, corr=False):

        self.env = env
        self.states_to_explain = states_to_explain

        # For Shapley calculations
        self.F_card = env.observation_space.shape[0]
        self.F = np.arange(self.F_card)

    def local_sverl_C_values(self, num_rolls, pi_Cs, multi_process=False, num_p=1):
        """
        Calculates local SVERL characteristics.
        
        num_rolls : Number of Monte Carlo roll outs for expected return.
        pi_Cs : All policies caused by removing every subset of features.
        """

        self.num_rolls = num_rolls
        self.pi_Cs = pi_Cs

        # Function for calculating partial policy for local SVERL
        self.get_policy = self.get_policy_local

        return self.get_all_C_values(self.get_local_global, multi_process, num_p)
    
    def fast_local_sverl_C_values(self, pi_Cs, num_rolls=1, valid_dict=None, multi_process=False, num_p=1):
        """
        Calculates local SVERL characteristics.
        Only valid for deterministic environments where states cannot be revisited.
        Much faster and more accurate.
        
        pi_Cs : All policies caused by removing every subset of features.
        """

        self.pi_Cs = pi_Cs

        # Do all the roll outs once now. Different values if available actions are state dependent or not.
        if valid_dict is None: self.action_values = {tuple(state) : np.mean([[self.play_episode(state.copy(), self.pi_Cs[tuple(self.F)], action) 
                                  for action in range(self.env.num_actions)] 
                                  for _ in tqdm_label(range(int(num_rolls)), 'Calculating Characteristics {}/{}'.format(i + 1, len(self.states_to_explain)))], axis=0)
                                  for i, state in enumerate(self.states_to_explain)}
            
        else: self.action_values = {tuple(state) : np.mean([[self.play_episode(state.copy(), self.pi_Cs[tuple(self.F)], action) if action in valid_dict[state.tobytes()] else 0 
                                  for action in range(self.env.num_actions)]
                                  for _ in tqdm_label(range(int(num_rolls)), 'Calculating Characteristics {}/{}'.format(i + 1, len(self.states_to_explain)))], axis=0)
                                  for i, state in enumerate(self.states_to_explain)}

        return self.get_all_C_values(self.get_fast_local, multi_process, num_p)

    def global_sverl_C_values(self, num_rolls, pi_Cs, multi_process=False, num_p=1):
        """
        Calculates global SVERL characteristics.
        Need pi_C for every state in env.
        
        num_rolls : Number of Monte Carlo roll outs for expected return.
        pi_Cs : All policies caused by removing every subset of features.
        """
        
        self.num_rolls = num_rolls
        self.pi_Cs = pi_Cs

        # Function for calculating partial policy for global SVERL
        self.get_policy = lambda state, C : copy.deepcopy(self.pi_Cs[tuple(C)]) 

        return self.get_all_C_values(self.get_local_global, multi_process, num_p)

    def shapley_on_policy(self, pi_Cs, multi_process=False, num_p=1):
        """
        Calculates Shapley applied to policy characteristics.
        
        pi_Cs : All policies caused by removing every subset of features.
        """

        self.pi_Cs = pi_Cs
        
        return self.get_all_C_values(self.get_shapley_on_policy, multi_process, num_p)

    def shapley_on_value(self, v_Cs, multi_process=False, num_p=1):
        """
        Calculates Shapley applied to value function characteristics.
        
        v_Cs : All expected return predictions caused by removing every subset of features.
        """

        self.v_Cs = v_Cs

        return self.get_all_C_values(self.get_shapley_on_value, multi_process, num_p)
        

# --------------------------------------------------------------------- Calculating a generic characteristic

    def get_all_C_values(self, get_C_values, multi_process=False, num_p=1):
        """
        Calculates all characteristic values for a given characteristic function.
        Multi or single processing. 
        """
        
        if multi_process: 

            characteristic_values = Manager().dict()
            all_C = F_not_i(self.F)

            for r in tqdm_label(range(int(np.ceil(len(all_C) / num_p))), 'Calculating Characteristics'):

                processes = [Process(target=self.worker, args=(C, characteristic_values, get_C_values)) for C in all_C[r * num_p : (r + 1) * num_p]]

                for p in processes:    
                    p.start()

                for p in processes:
                    p.join()

            return dict(characteristic_values)
            
        else: return {tuple(C): get_C_values(C) for C in tqdm_label(F_not_i(self.F), 'Calculating Characteristics')}
    
    def worker(self, C, characteristic_values, get_C_values): characteristic_values[tuple(C)] = get_C_values(C)

# --------------------------------------------------------------------- Local + Global SVERL 

    def get_local_global(self, C):
        """
        The SVERL characteristic values for one coalition for all states.
        play_policy dictates whether global or local is being calculated.
        """

        characteristic_values = defaultdict(float)

        for state in self.states_to_explain:

            play_policy = self.get_policy(state, C)

            for _ in range(int(self.num_rolls)): # Monte Carlo roll outs
                characteristic_values[tuple(state)] += self.play_episode(state.copy(), play_policy)

            characteristic_values[tuple(state)] /= self.num_rolls

        return dict(characteristic_values)
    
    def get_fast_local(self, C):
        """
        The local SVERL characteristic values for one coalition for all states.
        Only valid for deterministic environments where states cannot be revisited.
        Much faster and more accurate.

        V^{\pi_C}(s) = \sum_{a \in \A}{Q(s, a) * pi(a|s_C)}
        """

        return {tuple(state) : (value * self.pi_Cs[tuple(C)][tuple(state)]).sum() for state, value in self.action_values.items()}
    
    def get_policy_local(self, state, C):
        """
        Calculates the policy which local SVERL values uses for characteristic calculations.
        """

        play_policy = copy.deepcopy(self.pi_Cs[tuple(self.F)]) # Fully observed policy.
        play_policy[tuple(state)] = self.pi_Cs[tuple(C)][tuple(state)] # Partial policy for state being explained.

        return play_policy
    
    def play_episode(self, state, play_policy, action=None):
        """
        Plays an episode to evaluate a policy. (Monte Carlo roll out)
        """

        # Env set to state being explained, either using saved instance or built into env class.
        if self.reset_by_copy: self.env = copy.deepcopy(self.instances[tuple(state)])
        else: state, _ = self.env.reset(state)

        ret = 0

        if action is None: action = np.random.choice(self.env.num_actions, p=play_policy[tuple(state)])

        while True:

            # Usual RL, choose action, execute, update
            state, reward, terminated, truncated, _ = self.env.step(action)
            ret += reward

            if terminated or truncated: break
            else: action = np.random.choice(self.env.num_actions, p=play_policy[tuple(state)])

        return ret
    
# --------------------------------------------------------------------- Shapley applied to value function

    def get_shapley_on_value(self, C):
        """
        The Shapley applied to value function characteristic values for one coalition for all states.
        """

        return {tuple(state) : self.v_Cs[tuple(C)][tuple(state)] for state in self.states_to_explain}
    
# --------------------------------------------------------------------- Shapley applied to policy
    
    def get_shapley_on_policy(self, C):
        """
        The Shapley applied to policy characteristic values for one coalition for all states.
        """

        return {tuple(state.tolist()) : self.pi_Cs[tuple(C)][tuple(state.tolist())] for state in self.states_to_explain}



# --------------------------------------------------------------------- Shapley Interaction Index

    def get_interaction_index(self, interaction_order=2):

        # All possible 'order=interaction_order' combination, here is 2:
        total_feature = list(range(self.env.observation_space.shape[0]))
        combinations = list(itertools.combinations(total_feature, interaction_order))
        discrete_derivative = {}
        interaction_index = {}
        
        # continuous action normal (gaussian) distribution
        discrete_derivative_mean = {}
        discrete_derivative_var = {}
        

        for T in tqdm_label(combinations, 'Calculating Interaction Index'):
            N_not_T = [i for i in total_feature if i not in T]
            Cs = [c for r in range(len(N_not_T) + 1) for c in itertools.combinations(N_not_T, r)]

            # only for interation order = 2
            for state in tuple(self.states_to_explain.tolist()):

                for C in Cs:
                    parameter = (math.factorial(len(C)) * math.factorial(len(total_feature)-len(C)-len(T))) / math.factorial(len(total_feature)-len(T)+1) # parameter for multiple with discrete_derivative
                    
                    if isinstance(self.pi_Cs[tuple(sorted(C))][tuple(state)], th.distributions.Normal): # continuous action
                        discrete_derivative_mean[tuple(C)] =  (self.pi_Cs[tuple(sorted(C+T))][tuple(state)].mean + self.pi_Cs[tuple(sorted(C))][tuple(state)].mean
                            - self.pi_Cs[tuple(sorted(C+(T[0],)))][tuple(state)].mean - self.pi_Cs[tuple(sorted(C+(T[1],)))][tuple(state)].mean) # 2-order discrete_derivative
                        discrete_derivative_var[tuple(C)] =  (self.pi_Cs[tuple(sorted(C+T))][tuple(state)].variance + self.pi_Cs[tuple(sorted(C))][tuple(state)].variance
                            + self.pi_Cs[tuple(sorted(C+(T[0],)))][tuple(state)].variance + self.pi_Cs[tuple(sorted(C+(T[1],)))][tuple(state)].variance) # 2-order discrete_derivative
                        discrete_derivative_mean[tuple(C)] = parameter * discrete_derivative_mean[tuple(C)]
                        discrete_derivative_var[tuple(C)] = (parameter**2) * discrete_derivative_var[tuple(C)]
                        
                    else: # discrete action
                        discrete_derivative[tuple(C)] =  (self.pi_Cs[tuple(sorted(C+T))][tuple(state)] + self.pi_Cs[tuple(sorted(C))][tuple(state)] 
                                                    - self.pi_Cs[tuple(sorted(C+(T[0],)))][tuple(state)] - self.pi_Cs[tuple(sorted(C+(T[1],)))][tuple(state)]) # 2-order discrete_derivative
                        discrete_derivative[tuple(C)] = parameter * discrete_derivative[tuple(C)]

                if isinstance(self.pi_Cs[tuple(sorted(C))][tuple(state)], th.distributions.Normal): # continuous action
                    discrete_derivative_mean_sum = th.sum(th.stack(list(discrete_derivative_mean.values())), axis=0) # sum up for all C
                    discrete_derivative_var_sum = th.sum(th.stack(list(discrete_derivative_var.values())), axis=0) # sum up for all C
                    interaction_index[tuple(state), tuple(T)] = th.distributions.Normal(loc=discrete_derivative_mean_sum, scale=math.sqrt(discrete_derivative_var_sum))
                else:
                    discrete_derivative_sum = th.sum(th.stack(list(discrete_derivative.values())), axis=0) # sum up for all C
                    interaction_index[tuple(state), tuple(T)] = discrete_derivative_sum # output interaction index

                # empty dictionary for next iteration
                discrete_derivative = {}
                discrete_derivative_mean = {}
                discrete_derivative_var = {}


        return interaction_index
    
    
# --------------------------------------------------------------------- Shapley Value of combined features (not for continuous)

    def get_combined_features_shapley(self, num_features=2):
        total_feature = list(range(self.env.observation_space.shape[0]))
        combinations = list(itertools.combinations(total_feature, num_features))
        value_diff = {}
        combined_shapley_value = {}
        for T in tqdm_label(combinations, 'Calculating Combined Shapley Values'):
            N_not_T = [i for i in total_feature if i not in T]
            Cs = [c for r in range(len(N_not_T) + 1) for c in itertools.combinations(N_not_T, r)]

            # only for interation order = 2
            for state in tuple(self.states_to_explain.tolist()):

                for C in Cs:
                    parameter = (math.factorial(len(C)) * math.factorial(len(total_feature)-len(C)-len(T))) / math.factorial(len(total_feature)-len(T)+1) # parameter for multiple with value_diff
                    
                    value_diff[tuple(C)] =  (self.pi_Cs[tuple(sorted(C+T))][tuple(state)] - self.pi_Cs[tuple(sorted(C+(T[1],)))][tuple(state)]) # 2-order value_diff
                    value_diff[tuple(C)] = parameter * value_diff[tuple(C)]
                    
                # sum up all coliation C for combined shapley value
                value_diff_sum = th.sum(th.stack(list(value_diff.values())), axis=0) # sum up for all C
                combined_shapley_value[tuple(state), tuple(T)] = value_diff_sum # output interaction index
                
                # empty the dict for next iteration
                value_diff = {}
        
        return combined_shapley_value
import numpy as np
import torch as th
import math
from collections import defaultdict
from utils import F_not_i, tqdm_label

class Shapley:
    """
    Calculates Shapley values given characteristic values.
    """

    def __init__(self, states_to_explain):
        
        # For Shapley calculations
        self.F_card = len(states_to_explain[0])
        self.F = np.arange(self.F_card)
        self.states = states_to_explain

    def run(self, characteristic_values):
        """
        Calculates all the shapley values for every state and feature.
        """
        # For discrete action
        shapley_values = defaultdict(lambda: [[] for _ in range(self.F_card)])
        # For continuous action
        shapley_mean = defaultdict(lambda: [[] for _ in range(self.F_card)])
        shapley_stddev = defaultdict(lambda: [[] for _ in range(self.F_card)])

        for state in tqdm_label(self.states, "Calculating Shapley values"):

            # All characteristic values for a given state.
            C_values = {C: value_table[tuple(state.tolist())] for C, value_table in characteristic_values.items()}
            
            for feature in self.F:

                for C in F_not_i(self.F, feature): # All coalitions without feature

                    # Cardinal of C
                    C_card = len(C)

                    # Add our feature to the current coalition
                    C_with_i = np.append(C, feature).astype(int)
                    C_with_i.sort()

                    # Rolling sum, following formula
                    if isinstance(C_values[tuple(C)], th.distributions.Normal):
                        
                        # difference between two normal distribution:
                        mean_diff = C_values[tuple(C_with_i)].mean - C_values[tuple(C)].mean
                        var_diff = C_values[tuple(C_with_i)].variance + C_values[tuple(C)].variance
                        std_diff = math.sqrt(var_diff)
                        diff_C_value = th.distributions.Normal(loc=mean_diff, scale=std_diff)
                        
                        # Shapley value mean and stddev value:
                        shapley_mean[tuple(np.round(tuple(state.tolist()), 2))][feature].append(np.math.factorial(C_card) * np.math.factorial(self.F_card - C_card - 1) * diff_C_value.mean)
                        shapley_stddev[tuple(np.round(tuple(state.tolist()), 2))][feature].append(np.math.factorial(C_card) * np.math.factorial(self.F_card - C_card - 1) * diff_C_value.stddev)

                    else:
                        shapley_values[tuple(np.round(tuple(state.tolist()), 2))][feature].append(np.math.factorial(C_card) * np.math.factorial(self.F_card - C_card - 1) * (C_values[tuple(C_with_i)] - C_values[tuple(C)]))

                # Final weighting and return
                if isinstance(C_values[tuple(C)], th.distributions.Normal):
                    shapley_mean[tuple(np.round(tuple(state.tolist()), 2))][feature] = th.sum(th.stack(shapley_mean[tuple(np.round(tuple(state.tolist()), 2))][feature]), axis=0) / np.math.factorial(self.F_card)
                    shapley_stddev[tuple(np.round(tuple(state.tolist()), 2))][feature] = th.sum(th.stack(shapley_stddev[tuple(np.round(tuple(state.tolist()), 2))][feature]), axis=0) / np.math.factorial(self.F_card)
                    shapley_values[tuple(np.round(tuple(state.tolist()), 2))][feature] = th.distributions.Normal(shapley_mean[tuple(np.round(tuple(state.tolist()), 2))][feature], shapley_stddev[tuple(np.round(tuple(state.tolist()), 2))][feature])
                else:
                    shapley_values[tuple(np.round(tuple(state.tolist()), 2))][feature] = th.sum(th.stack(shapley_values[tuple(np.round(tuple(state.tolist()), 2))][feature]), axis=0) / np.math.factorial(self.F_card)

        return dict(shapley_values)
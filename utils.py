from tqdm import tqdm
import copy
from collections import defaultdict
import numpy as np
import torch as th
from torch.distributions.normal import Normal


from stable_baselines3.common.env_util import make_vec_env

# Specific device
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

        
def get_state_dist(model, vec_env, sample_size):
    """
    Approximates the limiting state distribution.
    """
    
    obs = vec_env.reset()

    state_dist = defaultdict(float)
    
    reset_num = 0
    for _ in tqdm_label(range(int(sample_size)), 'Approximating State Distribution'):
        
        action, _state = model.predict(obs, deterministic=True)
        
        obs = np.round(obs, 2)
        
        state_dist[tuple(map(tuple, obs))] += 1

        obs, rewards, dones, info = vec_env.step(action)
        reset_num = reset_num + 1
        
        if dones or reset_num == 20:
            obs = vec_env.reset()
            
    for state in state_dist:

        state_dist[state] /= int(sample_size)

    return state_dist


def F_not_i(F, feature=-1):
    """
    Finds all subsets not containing a feature.
    Given no feature, it returns all subsets of F.
    """ 

    all_C = []
    F_card = len(F)

    for i in range(1 << F_card):
        pos = [F[j] for j in F if (i & (1 << j))]
        if feature not in pos:
            all_C.append(pos)

    return all_C

def get_pi_C(env, model,  C, state_dist, states_to_explain):
    """
    Calculates pi_C for given states.
    """
    try: # discrete action
        policy = model.policy.get_distribution(states_to_explain).distribution.probs.detach() # remove the gradient
    except: # continous action
        policy = model.policy.get_distribution(states_to_explain).distribution
    
    if len(C) == env.observation_space.shape[0]: 
        pi_C = {}
        
        if isinstance(policy, Normal): # continous action
            for i in range(len(policy.loc)):
                pi_C[tuple(states_to_explain.tolist()[i])] = Normal(model.policy.get_distribution(states_to_explain).distribution.mean[i][0], model.policy.get_distribution(states_to_explain).distribution.stddev[i][0])
            return pi_C
        
        else: # discrete action
            for i in range(len(policy)):
                pi_C[tuple(states_to_explain.tolist()[i])] = policy[i]
            return pi_C
    
    else:

        # Mask out features not in C to find states which share observations.
        all_states = th.tensor(list(state_dist.keys()), device=device)

        state_dim = env.observation_space.shape[0]
        mask_states = mask_state(states_to_explain, state_dim, C)
        mask_all_states = mask_state(all_states, state_dim, C)

        state_dist_full = th.tensor(list(state_dist.values()), device=device) + 1e-16

        pi_C = {}
        temp_pi_C = {}

        for m_state in th.unique(mask_states, dim=0):

            ind = (mask_all_states == m_state).all(axis=2)
            state_dist_cond = state_dist_full[ind.squeeze()] /state_dist_full[ind.squeeze()].sum() # Conditional limiting state occupancy distribution.
            
            # tensor to tuple for dictionary
            m_state = tuple(m_state.tolist())

            if isinstance(policy, Normal):
                all_policy = model.policy.get_distribution(all_states).distribution
                temp_pi_C[tuple(m_state)] = get_expected_normal_distribution(all_policy, state_dist_cond, ind)
            else:

                temp_pi_C[tuple(m_state)] = (model.policy.get_distribution(all_states).distribution.probs.detach()[ind.squeeze()] * state_dist_cond[:, None]).sum(axis=0)

        for state, m_state in zip(states_to_explain, mask_states):

            # Convert tensor to tuple to create dictionary
            m_state = tuple(m_state.tolist())
            state = tuple(state.tolist())
            pi_C[state] = temp_pi_C[m_state] 

        return pi_C

def DQN_get_pi_C(env, model,  C, state_dist, states_to_explain):
    """
    Calculates pi_C for given states.
    """
    """
    Converts the agent's Q table into a policy table.
    """

    policy = defaultdict(lambda: np.full(env.action_space.n, 1/env.action_space.n))

    for state in states_to_explain:
        q_values = np.array(model.q_net(state.unsqueeze(0)).detach().cpu()[0])
        
        # Q values with slightly different values but should be same policy. 
        q_values = q_values.round(2)
        
        state = tuple(np.array(state.cpu()))

        policy[state] = th.tensor((q_values == q_values.max()).astype(float))
        policy[state] /= policy[state].sum()
    
    if len(C) == env.observation_space.shape[0]: 
        # Return the same policy
        pi_C = {}
        
        for i in range(len(policy)):
            pi_C[tuple(states_to_explain.tolist()[i])] = policy[tuple(np.array(states_to_explain[i].cpu()))]
        return pi_C
            
    else:

        # Mask out features not in C to find states which share observations.
        all_states = th.tensor(list(state_dist.keys()), device=device)

        state_dim = env.observation_space.shape[0]
        mask_states = mask_state(states_to_explain, state_dim, C)
        mask_all_states = mask_state(all_states, state_dim, C)

        state_dist_full = th.tensor(list(state_dist.values()), device=device) + 1e-16

        pi_C = {}
        temp_pi_C = {}
        
        # Add for policy lookup in pi_C = \sum_{s \in S}{\pi(a|s) * p(s|s_C)}
        temp_policy = []
        for state in all_states:
            q_values = np.array(model.q_net(state.unsqueeze(0)).detach().cpu())
            # Q values with slightly different values but should be same policy. 
            q_values = q_values.round(2)
            temp_policy.append((q_values == q_values.max()).astype(float))
    
        for m_state in th.unique(mask_states, dim=0):

            ind = (mask_all_states == m_state).all(axis=2).cpu()
            state_dist_cond = state_dist_full[ind.squeeze()] /state_dist_full[ind.squeeze()].sum() # Conditional limiting state occupancy distribution.
            
            # tensor to tuple for dictionary
            m_state = tuple(m_state.tolist())
            
            temp_pi_C[tuple(m_state)] = (th.tensor(temp_policy)[ind]* state_dist_cond[:, None].cpu()).sum(axis=0)# this is faster
                # remove the gradient

        # Set partially observed policies for fully observed states using^
        for state, m_state in zip(states_to_explain, mask_states):

            # Convert tensor to tuple to create dictionary
            m_state = tuple(m_state.tolist())
            state = tuple(state.tolist())
            pi_C[state] = temp_pi_C[m_state] 

        return pi_C

def mask_state(state, obs_dim, C):
    """
    Takes a state and masks out state features according to a coalition.
    """

    not_C = [i for i in range(obs_dim) if i not in C]
    if state is dict:
        out = state.clone()
        print(out, 'here!!!\n\n\n')
    else:
        out = state.clone()
        out[..., not_C] = -1

    return out

def get_global_dist(state_dist, sample_size):
    '''
    Generate another state distribution based on the sample size and standardize it
    '''
    
    global_dist = {}
    sorted_state_dist = sorted(state_dist, key=lambda x: state_dist[x], reverse=True)
    
    for i in range(sample_size):
        global_dist[sorted_state_dist[i][0]] = state_dist[sorted_state_dist[i]]
    
    # Get the values from the dictionary
    values = list(global_dist.values())
    # Calculate the sum of the values
    total = sum(values)
    # Normalize the values
    normalized_values = [value / total for value in values]
    normalized_global_dist = {key: value for key, value in zip(global_dist.keys(), normalized_values)}
    
    return normalized_global_dist

def get_expected_normal_distribution(policy, state_dist_cond, ind):
    # Ensure state_dist_cond sums to 1
    state_dist_cond = state_dist_cond / state_dist_cond.sum()

    # Calculate expected mean
    expected_mean = (policy.loc.squeeze()[ind.squeeze()] * state_dist_cond).sum()

    # Calculate expected variance
    # E[X^2] - (E[X])^2
    expected_second_moment = ((policy.loc.squeeze()[ind.squeeze()]**2 + policy.variance.squeeze()[ind.squeeze()]) * state_dist_cond).sum()
    expected_variance = expected_second_moment - expected_mean**2

    # Create new Normal distribution
    expected_distribution = th.distributions.Normal(loc=expected_mean, scale=th.sqrt(expected_variance))

    return expected_distribution

def tqdm_label(iterator, label):
    """
    Takes an iterator and produces a labelled tqdm progress bar.
    """

    pbar = tqdm(iterator)
    pbar.set_description(label)
    return pbar

def process_shapley_values(shapley_value, model, color_map=None):
    """
    Generalized function to process Shapley values and model predictions.
    
    Parameters:
    -----------
    shapley_value : dict
        Dictionary where keys are tuples and values are lists of tensors
    model : object
        Model object with predict method
    color_map : dict, optional
        Mapping from prediction values to colors. 
        Default: {0: 'orange', 1: 'blue', 2: 'green'}
    
    Returns:
    --------
    dict: Contains 'shapley_value', 'obs_value', and 'colors' lists
    """
    
    # Default color mapping
    if color_map is None:
        color_map = {0: 'orange', 1: 'green', 2: 'green'}
    
    # Determine number of features from first key
    if not shapley_value:
        return {'shapley_value': [], 'obs_value': [], 'colors': []}
    
    first_key = next(iter(shapley_value.keys()))
    num_features = len(first_key)
    
    # Initialize lists dynamically
    shapley_data = [[] for _ in range(num_features)]
    obs_data = [[] for _ in range(num_features)]
    colors = []
    
    # Process each shapley value entry
    for shapley_key in shapley_value.keys():
        # Extract plot values for each feature
        for i in range(num_features):
            shapley_data[i].append(shapley_value[shapley_key][i][0].cpu().detach().numpy())
            obs_data[i].append(shapley_key[i])
        
        # Get model prediction and assign color
        prediction = model.predict(shapley_key, deterministic=True)[0]
        
        # Handle case where prediction is a numpy array
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
        elif hasattr(prediction, '__len__') and len(prediction) == 1:
            prediction = prediction[0]
        
        if prediction in color_map:
            colors.append(color_map[prediction])
        else:
            print(f"Warning: Unknown prediction value {prediction}")
            colors.append('gray')  # Default color for unknown predictions
    
    return {
        'shapley_value': shapley_data,
        'obs_value': obs_data, 
        'colors': colors
    }

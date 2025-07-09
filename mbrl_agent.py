#%%writefile mbrl_agent.py

import numpy as np

#========== Utilitarian Functions ===============

# Function to compute the Kullback-Leibler (KL) divergence
def kl_divergence(dist1, dist2, epsilon=1e-10):
    """
    Compute the Kullback-Leibler (KL) divergence between two distributions.
    
    Arguments:
    - dist1: First probability distribution (list or array)
    - dist2 : Second probability distribution (list or array)

    Returns:
    - KL divergence value
    """

    dist1 = np.asarray([dist1[-1], dist1[0], dist1[1]]) if not isinstance(dist1, np.ndarray) else dist1
    dist2 = np.asarray([dist2[-1], dist2[0], dist2[1]]) if not isinstance(dist2, np.ndarray) else dist2
    dist1 = np.clip(dist1, epsilon, 1)
    dist2 = np.clip(dist2, epsilon, 1)
    return np.sum(dist1 * np.log(dist1 / dist2))


# Function to compute the symmetrized Jensen-Shannon (JS) divergence
def weighted_js_divergence(dist1, dist2, weight1, weight2):
    """
    Compute the symmetrized Jensen-Shannon divergence between two distributions.

    Arguments:
    - dist1: First probability distribution
    - dist2 : Second probability distribution
    - weight1: Weight of the first distribution
    - weight2: Weight of the second distribution

    Returns:
    - Symmetrized JS divergence value
    """
    dist1_array = np.array([dist1[-1], dist1[0], dist1[1]])
    dist2_array = np.array([dist2[-1], dist2[0], dist2[1]])
    
    # Compute the mixture distribution
    mix = (weight1 * dist1_array + weight2 * dist2_array) / (weight1 + weight2)
    
    return 0.5 * kl_divergence(dist1_array, mix) + 0.5 * kl_divergence(dist2_array, mix)

# Function to estimate a probability distribution from a sample
def estimate_distribution(samples):
    """
    Compute a probability distribution from a list of samples.

    Arguments:
    - samples: List of observed values (-1, 0, or 1)

    Returns:
    - Normalized probability distribution dictionary
    """
    unique, counts = np.unique(samples, return_counts=True)
    total = len(samples)

    # Compute probabilities
    dist = {val: count / total for val, count in zip(unique, counts)}

    # Ensure all possible values (-1, 0, 1) have a minimum probability
    for val in [-1, 0, 1]:
        if val not in dist:
            dist[val] = 1e-5

    # Normalize probabilities
    total_prob = sum(dist.values())
    dist = {key: value / total_prob for key, value in dist.items()}

    return dist


# ============ Adding module and Merging module ================

# Function to process a dilemma and update contexts
def update_contexts_with_dilemma(dilemma, contexts_dict, known_actions, threshold):
    """
    Process a dilemma and update existing contexts or create a new one if necessary.

    Arguments:
    - dilemma: Dictionary containing 'Reward', 'Action', and 'State' values
    - contexts_dict: Dictionary of existing contexts
    - known_actions: List of actions considered
    - threshold: Threshold for KL divergence to determine new contexts
    """
    reward_distribution = estimate_distribution(dilemma['Reward'])  

    # Check if the action exists in the list
    if dilemma['Action'] in known_actions:
        context_list = list(contexts_dict[dilemma['Action']].keys())
        DKL_context = []

        # Compute KL divergence for each context
        for i in range(len(context_list)):
            DKL_context.append(kl_divergence(reward_distribution, contexts_dict[dilemma['Action']]['C'+str(i+1)]['Distribution']))

        min_dkl, context_min_dkl = np.min(DKL_context), np.argmin(DKL_context)

        if min_dkl > threshold:
            # Create a new context if KL divergence exceeds the threshold
            contexts_dict[dilemma['Action']]['C'+str(1+len(contexts_dict[dilemma['Action']]))] = {
                'Distribution': reward_distribution,
                'Outcomes': dilemma['Reward'],
                'States': [[dilemma['State'], dilemma['Reward']]],
            }
        else:
            # Update existing context
            contexts_dict[dilemma['Action']]['C'+str(context_min_dkl+1)]['Outcomes'] = np.concatenate(
                (contexts_dict[dilemma['Action']]['C'+str(context_min_dkl+1)]['Outcomes'], dilemma['Reward']))
            contexts_dict[dilemma['Action']]['C'+str(context_min_dkl+1)]['States'].append([dilemma['State'], dilemma['Reward']])
            contexts_dict[dilemma['Action']]['C'+str(context_min_dkl+1)]['Distribution'] = estimate_distribution(
                contexts_dict[dilemma['Action']]['C'+str(context_min_dkl+1)]['Outcomes'])

    else:
        # If action is new, add it and create a new context
        known_actions.append(dilemma['Action'])
        contexts_dict[dilemma['Action']] = {
            'C1': {
                'Distribution': reward_distribution,
                'Outcomes': dilemma['Reward'],
                'States': [[dilemma['State'], dilemma['Reward']]],
            }
        }


# Function to merge similar contexts
def merge_similar_contexts(contexts_dict, threshold):
    """
    Merge similar contexts based on JS divergence.

    Arguments:
    - contexts_dict: Dictionary of contexts
    - threshold: Merging threshold
    """
    action_list = list(contexts_dict.keys())

    for action in action_list:
        swJS_list = [[], []]

        if len(contexts_dict[action]) > 1:
            for i in range(len(contexts_dict[action])):
                for j in range(i+1, len(contexts_dict[action])):
                    swJS_list[0].append([i, j])
                    swJS_list[1].append(
                        weighted_js_divergence(contexts_dict[action]['C'+str(i+1)]['Distribution'],
                             contexts_dict[action]['C'+str(j+1)]['Distribution'],
                             len(contexts_dict[action]['C'+str(i+1)]['Outcomes']),
                             len(contexts_dict[action]['C'+str(j+1)]['Outcomes']))
                    )

            min_swJS, context_min_swJS = np.min(swJS_list[1]), np.argmin(swJS_list[1])

            if min_swJS < threshold:
                i, j = swJS_list[0][context_min_swJS]
                n = len(contexts_dict[action])
    
                print(contexts_dict[action]['C'+str(j+1)])

                # Merge contexts
                contexts_dict[action]['C'+str(i+1)]['Outcomes'] += contexts_dict[action]['C'+str(j+1)]['Outcomes']
                contexts_dict[action]['C'+str(i+1)]['States'] += contexts_dict[action]['C'+str(j+1)]['States']
                contexts_dict[action]['C'+str(i+1)]['Distribution'] = estimate_distribution(contexts_dict[action]['C'+str(i+1)]['Outcomes'])

                # Replace merged context with the last context
                contexts_dict[action]['C'+str(j+1)] = contexts_dict[action]['C'+str(n)]
                del contexts_dict[action]['C'+str(n)]

# ========= MBRL Agent ===========
# Main MBRL agent function
def run_mbrl_agent(dilemma_list, contexts_dict, known_actions, adding_threshold, merging_threshold):
    """
    Main loop for the MBRL agent. Processes a list of dilemmas, updating or merging contexts accordingly.

    Parameters:
        dilemma_list (list of dict): Each dict must have:
            - 'Reward' (list of int): Observed rewards, e.g., [1, -1, 0]
            - 'Action' (str): The action taken
            - 'State' (any): The associated state
        contexts_dict (dict): Dictionary of contexts per action.
            Example structure:
                {
                    'ActionA': {
                        'C1': {
                            'Distribution': {-1: 0.2, 0: 0.3, 1: 0.5},
                            'Outcomes': [1, 1, -1],
                            'States': [[state1, 1], [state2, -1]]
                        },
                        ...
                    }
                }
        known_actions (list of str): List of actions already encountered
        adding_threshold (float): KL divergence threshold to create a new context
        merging_threshold (float): JS divergence threshold to merge contexts

    Returns:
        dict: Updated contexts_dict
        
    Example:
        updated_contexts = run_mbrl_agent(dilemmas, {}, [], 0.5, 0.2)
    """

    for dilemma in dilemma_list:
        update_contexts_with_dilemma(dilemma, contexts_dict, known_actions, adding_threshold)
        merge_similar_contexts(contexts_dict, merging_threshold)

    return contexts_dict
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 17:54:47 2018

@author: Clayton
"""

import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
from itertools import combinations
#from scipy.signal import savgol_filter

# Given an action ID, returns the joint speeds 
def action_ID_to_speeds(action_ID):
    wheel_diameter = 41
    wheels_distance = 53
    
    fast_translation = 60 #60mm/s
    slow_translation = 20 #20mm/s
    fast_rotation = 1.5 #1.5rad/s
    slow_rotation = 0.5 #0.5rad/s
    
    fast_translation_omega = fast_translation*2/wheel_diameter
    slow_translation_omega = slow_translation*2/wheel_diameter
    fast_rotation_omega = fast_rotation*wheels_distance/wheel_diameter
    slow_rotation_omega = slow_rotation*wheels_distance/wheel_diameter
    
    if action_ID == 0: # Still
        right_joint_speed = 0
        left_joint_speed = 0
    elif action_ID == 1: # Fast forward
        right_joint_speed = fast_translation_omega
        left_joint_speed = fast_translation_omega
    elif action_ID == 2: # Fast backward
        right_joint_speed = -fast_translation_omega
        left_joint_speed = -fast_translation_omega
    elif action_ID == 3: # Fast rotate CCW
        right_joint_speed = fast_rotation_omega
        left_joint_speed = -fast_rotation_omega
    elif action_ID == 4: # Fast rotate CW
        right_joint_speed = -fast_rotation_omega
        left_joint_speed = fast_rotation_omega
    elif action_ID == 5: # Slow forward
        right_joint_speed = slow_translation_omega
        left_joint_speed = slow_translation_omega
    elif action_ID == 6: # Slow backward
        right_joint_speed = -slow_translation_omega
        left_joint_speed = -slow_translation_omega
    elif action_ID == 7: # Slow rotate CCW
        right_joint_speed = slow_rotation_omega
        left_joint_speed = -slow_rotation_omega
    elif action_ID == 8: # Slow rotate CW
        right_joint_speed = -slow_rotation_omega
        left_joint_speed = slow_rotation_omega
        
    return right_joint_speed, left_joint_speed

# Given an array of action values and a temperature, returns an action ID according
# to Boltzmann exploration policy
def boltzmann_exploration(q_output, temperature):
    # Greedy action
    if temperature == 0:
        return np.argmax(q_output)
    else:
        # Small (and equivalent) modification to boltzmann function to avoid overflow
        max_q_output = max(q_output)
        q_output  = [q_out - max_q_output for q_out in q_output]
        # Obtains the exponentiation for all action-values in q_output
        boltzmann_exponentials = [math.exp(q_a/temperature) for q_a in q_output]
        sum_boltzmann_exponentials = sum(boltzmann_exponentials)
        # Obtains the probabilities for each action_ID
        boltzmann_probabilities = [boltzmann_exponential/sum_boltzmann_exponentials for \
                                   boltzmann_exponential in boltzmann_exponentials]
        # action_ID chosen 
        action_ID = int(np.random.choice(len(q_output), 1, p = boltzmann_probabilities))
        
        return action_ID

# Organize minibatch, indices and weights according to action ID
def organize_minibatch(minibatch, idx, weights, num_actions):
    # For all contents in the minibatch
    states = [[] for _ in range(0, num_actions)]
    rewards  = [[] for _ in range(0, num_actions)]
    next_states = [[] for _ in range(0, num_actions)]
    new_idx = [[] for _ in range(0, num_actions)]
    new_weights = [[] for _ in range(0, num_actions)]

    transition_idx = 0
    # Separating by action ID
    for transition in minibatch:
        state = transition[0]
        action_ID = transition[1]
        reward = transition[2]
        next_state = transition[3]

        states[action_ID].append(state)
        rewards[action_ID].append(reward)
        next_states[action_ID].append(next_state)
        new_idx[action_ID].append(idx[transition_idx])
        new_weights[action_ID].append(weights[transition_idx])
        
        transition_idx += 1
    
    return states, rewards, next_states, new_idx, new_weights

# Clip value by limits
def clip_value(value, min_limit, max_limit):
    if value > max_limit:
        value = max(value, max_limit)
    elif value < min_limit:
        value = min(value, min_limit)
        
    return value

# Find center of mass of certain agents
def find_com(agent_positions):
    transposed_positions = np.transpose(agent_positions)
    num_agents = len(agent_positions)
    # Center of mass coordinates
    x_com = sum(transposed_positions[0])/num_agents
    y_com = sum(transposed_positions[1])/num_agents
    
    return x_com, y_com

# Find Euclidean distance between two points
def find_distance_between_points(point1, point2):
    distance = math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))
    return distance

# Find cosine of the angle between two vectors
def find_cosine(v0, v1):
    return (np.dot(v0, v1))/(np.linalg.norm(v0)*np.linalg.norm(v1))

# Find Euclidean distance between an agent and other agents
def find_distance_to_other_robots(agent_positions, main_agent_position):
    distances_to_other_robots = []
    
    # Find the distance
    for agent_position in agent_positions:
        distances_to_other_robots.append(find_distance_between_points(agent_position, main_agent_position))
    
    return distances_to_other_robots

# Find the position, distance and IDs of the nearest robots of a specific robot
def find_n_nearest_robots(agent_positions, agent_ID, num_nearest_robots):
    num_agents = len(agent_positions)
    
    assert num_agents > num_nearest_robots
    
    # Find distances to all other robots
    distances_to_other_robots = find_distance_to_other_robots(agent_positions, agent_positions[agent_ID])
    distances_to_other_robots_cp = distances_to_other_robots.copy()
    # Sort in ascending order from left to right
    distances_to_other_robots.sort()
    
    # Find num_nearest_robots IDs
    nearest_robots_IDs = [distances_to_other_robots_cp.index(x) for x in distances_to_other_robots[1:num_nearest_robots + 1]]
    # Find their positions
    nearest_robots_positions = [agent_positions[x] for x in nearest_robots_IDs]

    # Find cosine of the angles between the vectors formed starting from the agent with ID agent_ID to its
    # num_nearest_robots IDs
    vectors = [np.subtract(x, agent_positions[agent_ID]) for x in nearest_robots_positions]
    cosines = [find_cosine(vectors[i], vectors[j]) for (i, j) in combinations(range(0, num_nearest_robots), 2)]

    return distances_to_other_robots[1:num_nearest_robots + 1], nearest_robots_positions, nearest_robots_IDs, cosines

# Find discrepancy in square formation
def find_discrepancy_square_formation(agent_ID, agent_positions, desired_distance):
    R = 1000 # Upscale factor
    num_agents = len(agent_positions)
    distances_to_nearest_robots, nearest_robots_positions, _, _ = \
    find_n_nearest_robots(agent_positions, agent_ID, num_agents - 1)
    
    # Error in distance to nearest robots
    distance_error_nearest = [math.pow(x - y, 2) for (x, y) in \
                               zip(distances_to_nearest_robots[:2], [desired_distance]*2)]
    # Error in distance to diagonal robot
    distance_error_diagonal = [math.pow(x - y, 2) for (x, y) in \
                               zip(distances_to_nearest_robots[2:], [desired_distance*1.41]*(num_agents - 3))]
    distance_error_diagonal = min(distance_error_diagonal)

    # Sum of all errors in distance to nearest robots
    distance_error_nearest = sum(distance_error_nearest)
    
    discrepancy = (distance_error_nearest + distance_error_diagonal)*R
    
    return discrepancy

# Find discrepancy in square formation for the n nearest robots
def find_discrepancy_square_formation_n_nearest_robots(agent_ID, agent_positions, desired_distance, num_agents):
    _, _, nearest_robots_IDs, _ = find_n_nearest_robots(agent_positions, agent_ID, num_agents - 1)
    
    n_nearest_robots_discrepancy = 0
    for ID in nearest_robots_IDs:
        n_nearest_robots_discrepancy += find_discrepancy_square_formation(ID, agent_positions, desired_distance)

    return n_nearest_robots_discrepancy

# Find discrepancy in dispersion
def find_discrepancy_dispersion(agent_ID, agent_positions, desired_distance):
    R = 1000 # Upscale factor

    distance_nearest_robot, _, _, _  = find_n_nearest_robots(agent_positions, agent_ID, 1)
    error_distance = math.pow(distance_nearest_robot[0] - desired_distance, 2)

    return error_distance*R

# Find discrepancy in dispersion for the n nearest robots
def find_discrepancy_dispersion_n_nearest_robots(agent_ID, agent_positions, desired_distance, num_agents):
    _, _, nearest_robots_IDs, _ = find_n_nearest_robots(agent_positions, agent_ID, num_agents - 1)
    
    n_nearest_robots_discrepancy = 0
    for ID in nearest_robots_IDs:
        n_nearest_robots_discrepancy += find_discrepancy_dispersion(ID, agent_positions, desired_distance)

    return n_nearest_robots_discrepancy

# Find difference in discrepancy in square formation
def find_diff_discrepancy_square_formation(agent_ID, agent_positions, last_agent_positions, desired_distance):
    last_discrepancy = find_discrepancy_square_formation(agent_ID, last_agent_positions, desired_distance)
    current_discrepancy = find_discrepancy_square_formation(agent_ID, agent_positions, desired_distance)
    
    return last_discrepancy - current_discrepancy

# Find difference in discrepancy in dispersion
def find_diff_discrepancy_dispersion(agent_ID, agent_positions, last_agent_positions, desired_distance):
    last_discrepancy = find_discrepancy_dispersion(agent_ID, last_agent_positions, desired_distance)
    current_discrepancy = find_discrepancy_dispersion(agent_ID, agent_positions, desired_distance)
    
    return last_discrepancy - current_discrepancy

# Find difference in discrepancy in square formation for the n nearest robots
def find_diff_discrepancy_square_formation_n_nearest_robots(agent_ID, agent_positions, last_agent_positions, \
                                                            desired_distance, num_agents):
    last_discrepancy = find_discrepancy_dispersion_n_nearest_robots(agent_ID, last_agent_positions, \
                                                                    desired_distance, num_agents)
    current_discrepancy = find_discrepancy_dispersion_n_nearest_robots(agent_ID, agent_positions, \
                                                                    desired_distance, num_agents)
    
    return last_discrepancy - current_discrepancy

# Save variables
def save(filepath, save_variable_list):
    with open(filepath, 'wb') as f:
        pickle.dump(save_variable_list, f)
        
def load(filepath):
    with open(filepath, 'rb') as f:
        info_task_list, total_summary_list = pickle.load(f)
        
    return info_task_list, total_summary_list

def load_and_show_results(filepath):
    info_task_list, total_summary_list = load(filepath)
    print_training_info(info_task_list)
    plot_training_results(total_summary_list)
    
def smooth1d(x, window_len):
    # copied from http://www.scipy.org/Cookbook/SignalSmooth

    s = np.r_[2*x[0] - x[window_len:1:-1], x, 2*x[-1] - x[-1:-window_len:-1]]
    w = np.hanning(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]

def plot_training_results(total_summary_list):
    #window_size = 3
    #order = 2
    fig = plt.figure(figsize=(15, 10))
    #fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

    rewards = fig.add_subplot(221)
    avg_d = fig.add_subplot(222)
    min_d = fig.add_subplot(223)
    max_d = fig.add_subplot(224)
    fig.subplots_adjust(wspace = 0.25, hspace = 0.25)
    
    rewards.set_title('Average joint reward per episode')
    rewards.set_xlabel('episode')
    rewards.set_ylabel('reward')

    avg_d.set_title('Average discrepancy per episode')
    avg_d.set_xlabel('episode')
    avg_d.set_ylabel('discrepancy')
    
    min_d.set_title('Minimum discrepancy per episode')
    min_d.set_xlabel('episode')
    min_d.set_ylabel('discrepancy')
    
    max_d.set_title('Maximum discrepancy per episode')
    max_d.set_xlabel('episode')
    max_d.set_ylabel('discrepancy')
    
    rewards.plot([entry[0] for entry in total_summary_list])
    avg_d.plot([entry[1] for entry in total_summary_list])
    min_d.plot([entry[2] for entry in total_summary_list])
    max_d.plot([entry[3] for entry in total_summary_list])
    #rewards.axis([0, 10, 0, 10])

    plt.show()
    
def print_training_info(info_task_list):
    
    num_episodes, time_steps, num_agents, em_capacity, em_reset_frequency, \
    batch_size, training_frequency, boltzmann_temperature, copy_weights_frequency, \
    steps_limit, discount_factor, task = info_task_list

    if task == 0:
        print('Dispersion task')
        print('---------------')
    elif task == 1:
        print('Square formation task')
        print('---------------------')
    
    print('Number of episodes: %d' % num_episodes)
    print('Time steps:         %d' % time_steps)
    print('Number of agents:   %d' % num_agents)
    print('Batch size:         %d' % batch_size)
    print('Boltzmann temp:     %d' % boltzmann_temperature)
    print('Steps limit:        %d' % steps_limit)
    print('Discount factor:    %.2f' % discount_factor)
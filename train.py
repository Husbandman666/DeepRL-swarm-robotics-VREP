# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 19:18:11 2018

@author: Clayton
"""

import vrep
import numpy as np
import agent as ag
import environment as env
import drqn as nn
import prioritized_EM as em
import tensorflow as tf
import utils

# Parameters for training
num_episodes = 400 # Number of episodes
restore = 0 # Restore from saved model
save_model_frequency = 25 # Frequency of saving models (in episodes)
path_to_model_to_restore = "./models/4SF.ckpt-20" # Path to the saved model
path_to_model_to_save = "./models/4SF.ckpt"
path_to_results_to_save = "./results.pkl"
time_steps = 10 # Number of time steps to consider in the recurrent network
num_agents = 4 # Number of agents
em_capacity = 10000 # Capacity of the experience memory
em_reset_frequency = 100 # Frequency for resetting the experience memory (in episodes) NOT APPLIED
batch_size = 400 # Batch size for training
training_frequency = 1 # Frequency of training (steps)
boltzmann_temperature = 6 # Boltzmann temperature
copy_weights_frequency = 50 # Frequency for copying weights from policy to target network
steps_limit = 300 # Limit of steps per episode
discount_factor = 0.99 # Discount factor
task = 1 # 1 stands for square formation, 0 stands for dispersion
num_actions = 9 # Number of actions
num_neurons = [14, 14, 12] # Number of neurons in each of the three layers
num_lstm_units = 14 # Number of LSTM units

if __name__ == "__main__":
    vrep.simxFinish(-1) # Close all open connections
    client_ID = vrep.simxStart('127.0.0.1', 19999, True, False, 5000, 5) # Connect to V-REP
    
    assert client_ID != -1, "Could not connect to remote API server."
    print('Connected to remote API server. Agents might move chaotically until the simulation is fully initialized.')
    
    # Start simulation
    vrep.simxStartSimulation(client_ID, vrep.simx_opmode_blocking)
    
    # Initialize all agents
    agent_list = [] # List containing agents
    
    for i in range(0, num_agents):
        if i == 0:
            agent_list.append(ag.Agent(client_ID, i))
        else:
            agent_list.append(ag.Agent(client_ID, i, name = "ePuck#" + str(i - 1)))
                                          
    # Initialize the environment
    print('Creating environment.')
    e = env.Environment(client_ID, agent_list, desired_distance = 0.28, \
                        skip_steps = 1, show_display = 0, steps_limit = steps_limit)
    
    # Initialize replay memory 
    print('Initializing prioritized experience memory.')
    p_em = em.PrioritizedExperienceMemory(capacity = em_capacity)
    
    # Initialize the deep Q-networks
    print('Initializing networks.')
    num_inputs = 8
    num_outputs = num_actions
    policy_net = nn.DeepRecurrentQNetwork('policy', time_steps = time_steps, discount_factor = discount_factor,
                                          num_actions = num_actions, num_neurons = num_neurons, 
                                          num_lstm_units = num_lstm_units)
    target_net = nn.DeepRecurrentQNetwork('target', time_steps = time_steps, discount_factor = discount_factor,
                                          num_actions = num_actions, num_neurons = num_neurons,
                                          num_lstm_units = num_lstm_units)
    
    # Initialize session in TensorFlow
    print('Starting session.')
    session = tf.Session()
    saver = tf.train.Saver(max_to_keep = 100)
    
    if restore:
        print('Restoring saved model.')
        saver.restore(session, path_to_model_to_restore)
    else:
        session.run(tf.global_variables_initializer())
        
    total_reward_list = [] # List containing all rewards of all agents per step
    total_discrepancy_list = []  # List containing discrepancies for each step
    
    total_summary_list = [] # List containing relevant info about the training
    
    print('Starting simulation.')
    for episode in range(0, num_episodes):
        e.start_new_episode(episode)
        step = 0    # Step in the episode
        end = 0     # If true, the episode must end
        total_loss = 0  # Total loss in the trainings for each episode
        loss_counts = 0 # Number of times the loss function was minimized
        total_reward_list = [] # List containing all rewards of all agents per step
        total_discrepancy_list = []  # List containing discrepancies for each step
        
        # Start the episode
        while not end:
            enabled_agents = [x for x in range(0, num_agents)] # Enabled agents for this step
            # Choose an action for all agents
            for a in agent_list:
                readings = a.get_sensor_buffer_readings(step, time_steps) # Obtain current sensor readings
                
                # Only for enabled agents
                if a.get_agent_ID() in enabled_agents:
                    with tf.device('/gpu:0'):
                        # Inference dictionary for placeholders
                        inference_dict = {policy_net.X: [readings], policy_net.is_training: False}
                        # Action values
                        q_out = session.run(policy_net.squeezed_total_output, feed_dict = inference_dict)
                        # Action ID according to boltzmann exploration policy
                        action_ID = utils.boltzmann_exploration(q_out, boltzmann_temperature)
                # In case agent is not enabled
                else:
                    action_ID = 0
                
                # Obtain joint speeds according to action_ID
                right_joint_speed, left_joint_speed = utils.action_ID_to_speeds(action_ID) 
                a.actuate(right_joint_speed, left_joint_speed, action_ID) # Actuate agent
                
            # Go to next step and iterate
            step = step + 1
            # Obtain rewards, discrepancy and whether the task has reached its end
            if task == 0:
                rewards, discrepancy, end = e.iterate_training_dispersion(step, enabled_agents)
            elif task == 1:
                rewards, discrepancy, end = e.iterate_training_square_formation(step, enabled_agents)
            
            # Stores discrepancy and rewards
            total_discrepancy_list.append(discrepancy)
            total_reward_list.append(sum([rewards[key] for key in rewards]))
                
            # Store sequence in the replay memory
            for a in agent_list:
                new_readings = a.get_sensor_buffer_readings(step, time_steps) # Get new sensor readings
                agent_ID = a.get_agent_ID() # Get agent ID
                # In case the agent is enabled
                if agent_ID in enabled_agents:
                    last_action_ID = a.get_last_action_ID() # Obtain last action
                    # Store into experience memory (by default with maximum priority)
                    p_em.add(a.get_sensor_buffer_readings(step - 1, time_steps), a.get_last_action_ID(), 
                                   rewards[str(agent_ID)], new_readings)
                    # Accumulate rewards for the episode
                    a.accum_rewards(rewards[str(agent_ID)])
                    
                    
            if not step % training_frequency:
                # Sample minibatches from the experience memory
                minibatch, idx, weights = p_em.sample(batch_size)
                # Separate by action_ID
                states, rewards, next_states, idx, weights = utils.organize_minibatch(minibatch, idx, weights, num_actions)
                outputs = [[] for _ in range(0, num_actions)]
                # Train with GPU
                with tf.device('/gpu:0'):
                    for action_ID in range(0, num_actions):
                        if len(states[action_ID]) != 0:
                            # feed_dict for target network
                            target_dict = {target_net.X: next_states[action_ID],
                                           target_net.is_training: False}
                            # Obtain total output from target network
                            target_net_outputs = session.run(target_net.total_output, feed_dict = target_dict)

                            # feed_dict for policy network
                            policy_dict = {policy_net.BS: len(states[action_ID]),
                                           policy_net.R: rewards[action_ID],
                                           policy_net.TO: target_net_outputs,
                                           policy_net.X: next_states[action_ID],
                                           policy_net.is_training: False}
                            # Obtain desired outputs to be trained with
                            outputs[action_ID] = session.run(policy_net.DDQN, feed_dict = policy_dict)

                            # For computing TD error
                            policy_dict = {policy_net.X: states[action_ID],
                                           policy_net.is_training: False}
                            policy_net_outputs = session.run(policy_net.total_output, feed_dict = policy_dict)
                            
                            # Compute TD error
                            policy_net_outputs_action_ID = [q[action_ID] for q in policy_net_outputs]
                            abs_td_error = [abs(int(output) - q) for (output, q) in \
                                            zip(outputs[action_ID], policy_net_outputs_action_ID)]
                            
                            # Update priorities in the experience memory
                            p_em.update_priorities(abs_td_error, idx[action_ID])
                            
                            # feed_dict for training
                            train_dict = {policy_net.X: states[action_ID], 
                                          policy_net.Y: outputs[action_ID],
                                          policy_net.is_training: True,
                                          policy_net.loss_weights: np.expand_dims(weights[action_ID], axis = 1)}
                            
                            # Train and obtain the value of the loss
                            loss, _ = session.run([policy_net.loss[action_ID], 
                                                   policy_net.optimize[action_ID]], feed_dict = train_dict)
                            
                            total_loss = loss + total_loss # Accumulate loss
                            loss_counts = loss_counts + 1  # Count how many losses have been accumulated
                
                # Copy weights from policy network to target network
                if step % copy_weights_frequency == 0 and step != 0:
                    with tf.device('/gpu:0'):
                        target_net.copy_weights(session, policy_net)
                
                # In case it is the end of the episode
                if end:
                    avg_total_reward = np.mean(total_reward_list)
                    avg_total_discrepancy = np.mean(total_discrepancy_list)
                    min_total_discrepancy = np.min(total_discrepancy_list)
                    max_total_discrepancy = np.max(total_discrepancy_list)
                    
                    summary_discrepancy_episode = [avg_total_reward, avg_total_discrepancy, min_total_discrepancy, \
                                                   max_total_discrepancy]
                    summary_discrepancy_episode = [round(content, 3) for content in summary_discrepancy_episode]
                    total_summary_list.append(summary_discrepancy_episode)
                    
                    print("Avg R, Avg D, Min D, Max D: %s. Average Loss: %f. Episode %i starting. " 
                          % (', '.join(map(str, summary_discrepancy_episode)), total_loss/(loss_counts*batch_size), \
                             episode + 2))
                    
                    # Save model
                    if (episode + 1) % save_model_frequency == 0:
                        saver.save(session, path_to_model_to_save, global_step = episode + 1)
                        print("Checkpoint saved")
                    
                    # Reset experience memory
                    #if (episode + 1) % em_reset_frequency == 0:
                       #p_em.empty()
    
                    break
                
    info_task_list = [num_episodes, time_steps, num_agents, em_capacity, em_reset_frequency, batch_size, 
                      training_frequency, boltzmann_temperature, copy_weights_frequency, steps_limit, discount_factor, task]

    # Save all information about training
    utils.save(path_to_results_to_save, [info_task_list, total_summary_list])
    print("Training results saved.")
    # Close session in Tensorflow
    session.close()
    # Close all open connections with VREP
    vrep.simxFinish(-1)
    print("Closed connection to VREP")
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 00:04:28 2018

@author: Clayton
"""

import vrep
import agent as ag
import environment as env
import drqn as nn
import tensorflow as tf
import utils

# Parameters for inference
num_episodes = 10 # Number of episodes
path_to_model_to_restore = "./models/4SF.ckpt-20" # Path to the saved model
time_steps = 10 # Number of time steps to consider in the recurrent network
num_agents = 4 # Number of agents
task = 1 # 1 stands for square formation, 0 stands for dispersion
num_actions = 9 # Number of actions
num_neurons = [24, 22, 18] # Number of neurons in each of the three layers
num_lstm_units = 24 # Number of LSTM units

if __name__ == "__main__":
    vrep.simxFinish(-1) # Close all open connections
    client_ID = vrep.simxStart('127.0.0.1', 19999, True, False, 5000, 5) # Connect to V-REP
    
    assert client_ID != -1, "Could not connect to remote API server."
    print ('Connected to remote API server')
    
    # Start simulation
    vrep.simxStartSimulation(client_ID, vrep.simx_opmode_blocking)
    
    # Initialize all agents
    agent_list = []
    
    for i in range(0, num_agents):
        if i == 0:
            agent_list.append(ag.Agent(client_ID, i))
        else:
            agent_list.append(ag.Agent(client_ID, i, name = "ePuck#" + str(i - 1)))
                                          
    # Initialize the environment
    print('Creating environment.')
    e = env.Environment(client_ID, agent_list, desired_distance = 0.28, skip_steps = 1, 
                        show_display = 1, steps_limit = 300)
    
    # Initialize the deep Q-networks
    num_inputs = 8
    num_outputs = num_actions
    target_net = nn.DeepRecurrentQNetwork('target', time_steps = time_steps, num_actions = num_actions,
                                          num_neurons = num_neurons, num_lstm_units = num_lstm_units)
    
    # Initialize session in TensorFlow
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # Restore saved model
    saver = tf.train.Saver()
    saver.restore(session, path_to_model_to_restore)

    print('Starting simulation.')
    for episode in range(0, num_episodes):
        e.start_new_episode(episode)
        step = 0    # Step in the episode
        end = 0     # If true, the episode must end
        
        # Start the episode
        while not end:
            enabled_agents = [x for x in range(0, num_agents)]
            # Choose an action for all agents
            for a in agent_list:
                readings = a.get_sensor_buffer_readings(step, time_steps) # Obtain current sensor readings
                # Only for enabled agents
                if a.get_agent_ID() in enabled_agents:
                    # feed_dict for inference
                    inference_dict = {target_net.X: [readings],
                                      target_net.is_training: False}
                    # Obtain action ID
                    action_ID = session.run(target_net.max_action_id_squeezed, \
                                            feed_dict = inference_dict)
                # In case agent is not enabled
                else:
                    action_ID = 0
                
                # Obtain joint speeds according to action_ID
                right_joint_speed, left_joint_speed = utils.action_ID_to_speeds(action_ID) 
                a.actuate(right_joint_speed, left_joint_speed, action_ID) # Actuate agent
            
            # Go to next step and iterate
            step = step + 1
            if task == 0:
                discrepancy, end = e.iterate_inference_dispersion(step)
            elif task == 1:
                discrepancy, end = e.iterate_inference_square_formation(step)
            
            
            # Episode is finished
            if end:
                print(discrepancy)
                break
            
    # Close session in Tensorflow
    session.close()
    # Close all open connections with VREP
    vrep.simxFinish(-1)
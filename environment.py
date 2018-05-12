# -*- coding: utf-8 -*-
"""
Created on Fri May 11 00:05:06 2018

@author: Clayton
"""


import vrep
import time
import utils

class Environment:
    def __init__(self, client_ID, agents, desired_distance, success_threshold = 1, fail_threshold = 500,
                 skip_steps = 1, show_display = 1, steps_limit = 300, steps_before_evaluation = 1):
        
        self.__client_ID = client_ID # VREP Client ID
        self.__agents = agents  # List containing all agents
        self.__num_agents = len(agents) # Number of agents
        self.__desired_distance = desired_distance # Desired distance, either for dispersion or pattern formation
        self.__success_threshold = success_threshold # Discrepancy below which the episode succesfully finishes
        self.__fail_threshold = fail_threshold # Discrepancy above which the episode fails
        self.__skip_steps = skip_steps # Number of steps to skip during iteration in VREP
        self.__show_display = show_display # Show the display in VREP. Set to false to increase performance
        self.__steps_limit = steps_limit # Maximum number of steps, above which the task reaches its end
        self.__steps_before_evaluation = steps_before_evaluation # How often (in steps) rewards are given

        # Start in synchronous mode
        error = vrep.simxSynchronous(client_ID, 1)
        assert error == 0, "Could not start simulation in synchronous mode."
        
        self.__step = 0
        
    ## Getters and setters
    def get_client_ID(self):
        return self.__client_ID
    
    def get_num_agents(self):
        return self.__num_agents
    
    def get_agents(self):
        return self.__agents
    
    def get_success_threshold(self):
        return self.__success_threshold
    
    def set_success_threshold(self, success_threshold):
        self.__success_threshold = success_threshold
        
    def get_fail_threshold(self):
        return self.__fail_threshold
    
    def set_fail_threshold(self, fail_threshold):
        self.__fail_threshold = fail_threshold

    def get_skip_steps(self):
        return self.__skip_steps
    
    def set_skip_steps(self, skip_steps):
         self.__skip_steps = skip_steps
         
    def get_show_display(self):
        return self.__show_display
    
    def set_show_display(self, show_display):
        self.__show_display = show_display
        
    def get_steps_limit(self):
        return self.__steps_limit
    
    def set_steps_limit(self, steps_limit):
        self.__steps_limit = steps_limit
        
    def get_steps_before_evaluation(self):
        return self.__steps_before_evaluation
    
    def set_steps_before_evaluation(self, steps_before_evaluation):
        self.__steps_before_evaluation = steps_before_evaluation
    
    def start_new_episode(self, episode):
        # Stop the simulation if it is running
        vrep.simxStopSimulation(self.__client_ID, vrep.simx_opmode_blocking)
        time.sleep(1) # One second delay
        # Start simulation
        vrep.simxStartSimulation(self.__client_ID, vrep.simx_opmode_blocking)
        # Set display to enabled or not
        vrep.simxSetBooleanParameter(self.__client_ID, vrep.sim_boolparam_display_enabled, self.__show_display, vrep.simx_opmode_oneshot)
        # Enable threaded rendering
        vrep.simxSetBooleanParameter(self.__client_ID, vrep.sim_boolparam_threaded_rendering_enabled, 1, vrep.simx_opmode_oneshot)
        
        # Catch error
        error = vrep.simxSynchronous(self.__client_ID, 1)
        assert error == 0, "Could not start simulation in synchronous mode."
        
        # Reset all agents
        for agent in self.__agents:
            agent.restart()
     
    # Get the agent positions for a specific step
    def __get_agent_positions(self, step):
        agent_positions = []
        
        # Get and store the agent positions
        for agent in self.__agents:
            agent_positions.append(agent.get_position(step))
            
        return agent_positions
    
    def __iterate(self, step):
        # Continue the simulation for the number of steps defined in skip_steps
        for i in range(0, self.__skip_steps):
            vrep.simxSynchronousTrigger(self.__client_ID)
        
        # Current agents' positions
        agent_positions = self.__get_agent_positions(step)
        
        # Last agents' positions
        if step - self.__steps_before_evaluation >= 0:
            last_agent_positions = self.__get_agent_positions(step - self.__steps_before_evaluation)
        else:
            last_agent_positions = self.__get_agent_positions(0)
            
        return agent_positions, last_agent_positions
    
    def iterate_training_square_formation(self, step, agent_IDs_turn):
        # Iterate, get current and last agent positions
        agent_positions, last_agent_positions = self.__iterate(step)
        
        total_discrepancy = 0
        rewards = {}
        
        # For all agents
        for agent_ID in range(0, self.__num_agents):
            total_discrepancy += utils.find_discrepancy_square_formation(agent_ID, agent_positions, self.__desired_distance)
            
            # Discrepancy due to nearest robots
            diff_discrepacy_n_nearest_robots = utils.find_diff_discrepancy_square_formation_n_nearest_robots(agent_ID, \
                                                                                                             agent_positions, \
                                                                                                             last_agent_positions, \
                                                                                                             self.__desired_distance, \
                                                                                                             2)
            # Own's discrepancy 
            diff_discrepancy_own =  utils.find_diff_discrepancy_square_formation(agent_ID, agent_positions, last_agent_positions, \
                                                                                 self.__desired_distance)
            # Reward the agent according to change in discrepancies
            if agent_ID in agent_IDs_turn and step % self.__steps_before_evaluation == 0:
                rewards[str(agent_ID)] = diff_discrepancy_own + diff_discrepacy_n_nearest_robots
            elif agent_ID in agent_IDs_turn and step % self.__steps_before_evaluation != 0:
                rewards[str(agent_ID)] = 0
            
            # Always decrease the reward when the agent moves, to incentivize it to move as little as possible
            if self.__agents[agent_ID].get_last_action_ID() != 0:
                    rewards[str(agent_ID)] -= 0.1
                    
            
        # Decide whether or not the training should end at the next step
        if step == self.__steps_limit:
            end = 1
        elif ((total_discrepancy > self.__fail_threshold) or (total_discrepancy < self.__success_threshold)) and total_discrepancy > 0:
            end = 1
        else:
            end = 0

        return rewards, total_discrepancy, end
        
    def iterate_training_dispersion(self, step, agent_IDs_turn):
        # Iterate, get current and last agent positions
        agent_positions, last_agent_positions = self.__iterate(step)
        
        total_discrepancy = 0
        rewards = {}
        
        # For all agents
        for agent_ID in range(0, self.__num_agents):
            total_discrepancy += utils.find_discrepancy_dispersion(agent_ID, agent_positions, self.__desired_distance)
           
            # Own's discrepancy 
            diff_discrepancy_own =  utils.find_diff_discrepancy_dispersion(agent_ID, agent_positions, last_agent_positions, \
                                                                           self.__desired_distance)
            # Reward the agent according to change in discrepancies
            if agent_ID in agent_IDs_turn and step % self.__steps_before_evaluation == 0:
                rewards[str(agent_ID)] = diff_discrepancy_own
            elif agent_ID in agent_IDs_turn and step % self.__steps_before_evaluation != 0:
                rewards[str(agent_ID)] = 0
            
            # Always decrease the reward when the agent moves, to incentivize it to move as little as possible
            if self.__agents[agent_ID].get_last_action_ID() != 0:
                    rewards[str(agent_ID)] -= 0.05
                    
        # Decide whether or not the training should end at the next step
        if step == self.__steps_limit:
            end = 1
        elif ((total_discrepancy > self.__fail_threshold) or (total_discrepancy < self.__success_threshold)) and total_discrepancy > 0:
            end = 1
        else:
            end = 0

        return rewards, total_discrepancy, end
        
    def iterate_inference_square_formation(self, step):
        end = 0
        # Current agents' positions
        agent_positions = self.__get_agent_positions(step)
        total_discrepancy = 0
        # Calculate the discrepancy of every agent and sum them all
        for agent_ID in range(0, self.__num_agents):
            total_discrepancy += utils.find_discrepancy_square_formation(agent_ID, agent_positions, 
                                                                         self.__desired_distance)
            
        # Continue the simulation for the number of steps defined in skip_steps
        for i in range(0, self.__skip_steps):
            vrep.simxSynchronousTrigger(self.__client_ID)
            
        # Decide whether or not the training should end at the next step
        if step == self.__steps_limit:
            end = 1
    
        return total_discrepancy, end

    def iterate_inference_dispersion(self, step):
        end = 0
        # Current agents' positions
        agent_positions = self.__get_agent_positions(step)
        total_discrepancy = 0
        # Calculate the discrepancy of every agent and sum them all
        for agent_ID in range(0, self.__num_agents):
            total_discrepancy += utils.find_discrepancy_dispersion(agent_ID, agent_positions, 
                                                                   self.__desired_distance)
            
        # Continue the simulation for the number of steps defined in skip_steps
        for i in range(0, self.__skip_steps):
            vrep.simxSynchronousTrigger(self.__client_ID)
            
        # Decide whether or not the training should end at the next step
        print(step)
        if step == self.__steps_limit:
            end = 1
    
        return total_discrepancy, end
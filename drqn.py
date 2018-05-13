# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 18:15:55 2018

@author: Clayton
"""

import tensorflow as tf

class DeepRecurrentQNetwork:
    def __init__(self, name, num_states = 8, num_actions = 5, num_neurons = [20, 15, 10], num_lstm_units = 25, 
                 num_cells_lstm = 1, time_steps = 4, lr = 0.001, discount_factor = 0.99, seed = 1):
        with tf.device('/device:GPU:0'):
        
            # Saves input arguments
            self.name = name
            self.num_inputs = num_states   # Number of states (inputs)
            self.num_outputs = num_actions # Number of actions (outputs)
            self.num_neurons = num_neurons # Number of neurons (organized by layers) for the feedforward network
            self.num_layers = len(num_neurons) # Number of layers in the MLP
            self.num_lstm_units = num_lstm_units # Number of LSTM units
            self.num_cells_lstm = num_cells_lstm # Number of LSTM cells
            self.time_steps = time_steps # Number of time steps considered in the LSTM cell
            self.lr = lr # Learning rate for the optimization algorithm
            self.discount_factor = discount_factor # Discount factor for the expected return
            self.seed = seed  # Seed for random weight initialization
            
            # Placeholders
            self.X = tf.placeholder("float", [None, self.time_steps, self.num_inputs]) # Inputs
            self.Y = tf.placeholder("float", [None, 1]) # Outputs
            self.R = tf.placeholder("float") # Rewards
            self.TO = tf.placeholder("float", [None, self.num_outputs]) # Total output
            self.BS = tf.placeholder(tf.int64) # Batch size
            self.is_training = tf.placeholder(tf.bool) # Used for dropout
            self.loss_weights = tf.placeholder("float", [None, 1]) # Due to prioritized experience replay
            
            # Networks with the same name share the same weights and biases
            with tf.variable_scope(self.name) as scope:
                # Attempt to initialize weights and biases
                try:
                    self.__init_weights_and_biases()
                    self.__init_training_op()  # Initialize ops for training
                # If weights and biases already exist, reuse them
                except ValueError:
                    scope.reuse_variables()
                    self.__init_weights_and_biases()
                    self.__init_training_op()  # Initialize ops for training
            
            
            self.__init_inference_op() # Initialize ops for inference
            self.__init_RL_op()      # Initialize ops for reinforcement learning 
            self.__init_copy_op()    # Initialize ops for weight copying between networks

    # Initialize weights and biases
    def __init_weights_and_biases(self):
        # Defines initializer according to Xavier and Bengio's method
        initializer = tf.contrib.layers.xavier_initializer(seed = self.seed)
        
        # Networks with multi layer LSTM cell has not yet been implemented
        if self.num_cells_lstm > 1:
             raise NotImplementedError("Implementation missing for more than one LSTM cell.")
        
        # Create LSTM layers
        if self.num_cells_lstm > 1:
            stacked_lstm_layers = []
            for i in range(0, self.num_cells_lstm):
                stacked_lstm_layers.append(tf.contrib.rnn.LSTMCell(self.num_lstm_units, initializer = initializer, forget_bias = 1))
            
            # Multi layer LSTM cell
            self.lstm_cell = tf.contrib.rnn.MultiRNNCell(stacked_lstm_layers)
        else:
            # Single layer LSTM cell
            self.lstm_cell = tf.contrib.rnn.LSTMCell(self.num_lstm_units, initializer = initializer, forget_bias = 1)
        
        # Outputs of the LSTM cell
        self.lstm_outputs, _  = tf.nn.dynamic_rnn(self.lstm_cell, self.X, dtype=tf.float32)
        self.lstm_outputs = tf.transpose(self.lstm_outputs, [1, 0, 2])
        # Last output of the LSTM cell
        self.last_lstm_output = tf.gather(self.lstm_outputs, int(self.lstm_outputs.get_shape()[0]) - 1)
        
        # Weights and biases for the feedforward network
        self.weights = {}
        self.biases = {}
        for i in range(0, self.num_layers):
            var_name_weight = 'w%s' % str(i)
            var_name_bias = 'b%s' % str(i)
            if i == 0:
                self.weights[var_name_weight] = tf.get_variable(var_name_weight, [self.num_lstm_units, self.num_neurons[i]], 
                                                                initializer = initializer)
            else:
                self.weights[var_name_weight] = tf.get_variable(var_name_weight, [self.num_neurons[i - 1], 
                                                         self.num_neurons[i]], initializer = initializer)
    
            self.biases[var_name_bias] = tf.get_variable(var_name_bias, [self.num_neurons[i]], initializer = initializer)
                
        
        self.weights['wout'] = tf.get_variable("wout", [self.num_neurons[self.num_layers - 1], self.num_outputs], 
                                               initializer = initializer)
         
        self.biases['bout'] = tf.get_variable("bout", [self.num_outputs], initializer = initializer)

    def __init_training_op(self):
        # Total output of the network, i.e., action values
        self.total_output = self.forward(self.last_lstm_output, self.is_training)
        # For inference, squeeze the total output
        self.squeezed_total_output = tf.squeeze(self.total_output)
        # Outputs of the network (each action corresponds to one output)
        self.out = []
        # List of loss functions for each output
        self.loss = []
        # List of optimization operations for each output
        self.optimize = []
            
        for i in range(0, self.num_outputs):
            self.out.append(tf.transpose(tf.gather(tf.transpose(self.total_output), [i])))

            # Loss function for each output
            self.weighted_squared_difference = tf.multiply(self.loss_weights, tf.square(self.Y - self.out[i]))
            self.loss.append(tf.reduce_mean(self.weighted_squared_difference))
            # Optimization operation using Adam to minimize the loss function
            self.optimize.append(tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.loss[i]))
    
    # Initialize inference operations
    def __init_inference_op(self):
        # Action_ID with maximum value
        self.max_action_id = tf.argmax(self.total_output, axis = 1)
        self.max_action_id_squeezed = tf.squeeze(self.max_action_id)
        # Value of the action ID with maximum value
        self.max_action_value = tf.reduce_max(self.total_output, axis = 1)
        self.max_action_value_squeezed = tf.squeeze(self.max_action_value)
        
    # Initiliaze operation for DDQN algorithm
    def __init_RL_op(self):
        self.DDQN = self.find_Y_DDQN(self.BS, self.R, self.TO)
    
    # Initilize weight/bias copying ops
    def __init_copy_op(self):
        # Placeholders and copy ops
        self.weights_ph = {}
        self.weights_cp = {}
        self.biases_ph = {}
        self.biases_cp = {}
        for i in range(0, self.num_layers):
            var_name_weight = 'w%s' % str(i)
            var_name_bias = 'b%s' % str(i)
            if i == 0:
                self.weights_ph[var_name_weight] = tf.placeholder("float", [self.num_lstm_units, self.num_neurons[i]])
            else:
                self.weights_ph[var_name_weight] = tf.placeholder("float", [self.num_neurons[i - 1], self.num_neurons[i]])
            
            self.biases_ph[var_name_bias] = tf.placeholder("float", [self.num_neurons[i]])
            self.weights_cp[var_name_weight] = tf.assign(self.weights[var_name_weight], self.weights_ph[var_name_weight])
            self.biases_cp[var_name_bias] = tf.assign(self.biases[var_name_bias], self.biases_ph[var_name_bias])
                
        self.weights_ph['wout'] = tf.placeholder("float", [self.num_neurons[self.num_layers - 1], self.num_outputs])
        self.biases_ph['bout'] = tf.placeholder("float", [self.num_outputs])
        self.weights_cp['wout'] = tf.assign(self.weights['wout'], self.weights_ph['wout'])
        self.biases_cp['bout'] = tf.assign(self.biases['bout'], self.biases_ph['bout'])
        
        self.lstm_kernel_ph = tf.placeholder("float")
        self.lstm_bias_ph = tf.placeholder("float")
        self.lstm_kernel_cp = tf.assign(self.lstm_cell.weights[0], self.lstm_kernel_ph)
        self.lstm_bias_cp = tf.assign(self.lstm_cell.weights[1], self.lstm_bias_ph)
    
    # Feedforward network
    def forward(self, x, is_training): 
        for i in range(0, self.num_layers):
            var_name_weight = 'w%s' % str(i)
            var_name_bias = 'b%s' % str(i)
            x = tf.add(tf.matmul(x, self.weights[var_name_weight]), self.biases[var_name_bias])
            x = tf.nn.relu(x)
            
        x = tf.add(tf.matmul(x, self.weights['wout']), self.biases['bout'])
        return x
    
    # Operation for DDQN algorithm
    def find_Y_DDQN(self, batch_size, rewards, target_net_total_output):
        # Batch indices
        batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
        # Indices to gather
        gather_idx = tf.concat([batch_idx, tf.expand_dims(self.max_action_id, 1)], axis = 1)
        
        x = tf.gather_nd(target_net_total_output, gather_idx)
        y = tf.add(rewards, tf.multiply(self.discount_factor, x))
        y = tf.expand_dims(y, axis = 1)
        
        return y

    def copy_weights(self, session, copy_from):
        # Obtain weights from the network to be copied from
        feed_dict = {}
        
        for i in range(0, self.num_layers):
            var_name_weight = 'w%s' % str(i)
            var_name_bias = 'b%s' % str(i)
            feed_dict[self.weights_ph[var_name_weight]] = session.run(copy_from.weights[var_name_weight])
            feed_dict[self.biases_ph[var_name_bias]] = session.run(copy_from.biases[var_name_bias])
        
        feed_dict[self.weights_ph['wout']] = session.run(copy_from.weights['wout'])
        feed_dict[self.biases_ph['bout']] = session.run(copy_from.biases['bout'])
        feed_dict[self.lstm_kernel_ph] = session.run(copy_from.lstm_cell.weights[0])
        feed_dict[self.lstm_bias_ph] = session.run(copy_from.lstm_cell.weights[1])
        
        copy_ops = [self.weights_cp[key] for key in self.weights_cp] + \
                   [self.biases_cp[key] for key in self.biases_cp] + \
                   [self.lstm_kernel_cp, self.lstm_bias_cp]
        # Perform the operation of copying weights
        session.run(copy_ops, feed_dict = feed_dict)
#!/usr/bin/env python

################################################################################
################################################################################
## Authorship
################################################################################
################################################################################
##### Project
############### susgym
##### Name
############### rl_agent.py
##### Author
############### Trenton Langer
##### Creation Date
############### 20230108
##### Description
############### Reinfforcement learning based agents for use in an OpenAI gym based environment
##### Project
############### YYYYMMDD - (CONTRIBUTOR) EXAMPLE



################################################################################
################################################################################
## Imports
################################################################################
################################################################################

# Helpers
from . import agent_helpers

# Reinforcement Learning
# from keras.layers import Input, Dense, Activation
# from keras.models import Sequential, clone_model
# from keras.optimizers import Adam
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers

tf.config.set_visible_devices([], 'GPU')
tf.get_logger().setLevel('WARNING')

# Other
from collections import deque
import numpy as np
import random

# Debug
from pprint import pprint


################################################################################
################################################################################
## Subroutines
################################################################################
################################################################################

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
"""
description:
-> Updates a Tensor towards a new target, either entirely or in small steps
parameters:
-> weights: Tensor to update
-> targe_weights: Desired Tensor
-> tau: Factor for scale of update. Tau=1 overwrites weights with new targets
return:
-> No return
"""
@tf.function # Decorated as Tensorflow function for execution optimization
def update_target(weights, target_weights, tau):
    for (a, b) in zip(weights, target_weights):
        a.assign(b * tau + a * (1 - tau))

"""
description: randomly selects indices from decoded state information
parameters:
-> decoded_state: dict of state information
-> action: numpy array to modify
return:
-> No return. Modifies action in place
"""
def randValidGuess(dstate, act):
    num_opps = dstate["num_players"]-1
    for opp_idx in range(0, num_opps):
        valid_opp_chars = np.where(dstate["knowledge"][opp_idx] == 1)[0]
        act[0-(num_opps-opp_idx)] = valid_opp_chars[np.random.randint(valid_opp_chars.size)]



################################################################################
################################################################################
## Classes
################################################################################
################################################################################

"""
description:
-> RL based agent, epsilon-greedy choice between DDPG network or stochastic (valid) actions
"""
class ddpgSusAgent(agent_helpers.susAgent):
    def __init__(self, eps=0.1, lr=0.01, gamma=0.99, tau=1, batch_size=64):
        # Parent Setup
        super().__init__()
        # Basic Setup
        self.curr_state, self.curr_action = None, None
        self.invalid_last_act = False
        # Model Parameters
        self.eps = eps # Explore/Exploit rate (epsilon)
        self.lr = lr # Learning rate (for network training)
        self.gamma = gamma # Discount rate (gamma) for future reward scaling
        self.tau = tau # Target update rate (tau) for policy to target network updates
        # Models
        self.batch_size = batch_size
        self.replay_memory = agent_helpers.ReplayBuffer(batch_size=self.batch_size) # list of [state, action, reward, new state, terminal]
        self.actor_optimizer = tf.keras.optimizers.Adam(self.lr)
        self.actor, self.target_actor = None, None # created on first action select (need act/obs spaces)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.lr)
        self.critic, self.target_critic = None, None # created on first action select (need act/obs spaces)
        # Functionality Controls
        self.train = False # True => e-greedy, update network; False => 'optimal' actions only

    def update(self, next_state, reward, done, info):
        # Parent
        super().update(next_state, reward, done, info)
        # Basic Updates
        if np.all(self.curr_state == next_state): self.invalid_last_act = True
        # Network Updates
        self.replay_memory.store((self.curr_state, self.curr_action, reward, next_state))
        # Generate Batch
        if self.train and len(self.replay_memory) > self.batch_size:
            batch_states, batch_actions, batch_rewards, batch_next_states = self.replay_memory.biased_sample(bias=self.batch_size) # Half recent, half random - TODO: helps with invalid acts?
            batch_states = tf.convert_to_tensor(batch_states)
            batch_actions = tf.convert_to_tensor(batch_actions)
            batch_rewards = tf.convert_to_tensor(batch_rewards)
            batch_next_states = tf.convert_to_tensor(batch_next_states)
            self._learn(batch_states, batch_actions, batch_rewards, batch_next_states)

    def reset(self):
        # Basic
        super().reset()
        # Model Setup
        if self.actor is not None:
            update_target(self.target_actor.variables, self.actor.variables, self.tau)
            update_target(self.target_critic.variables, self.critic.variables, self.tau)

    def save_weights(self):
        if self.train and False:
            self.actor.save_weights('./checkpoints/ddpg_actor3')
            self.critic.save_weights('./checkpoints/ddpg_critic3')
        else:
            pass

    def pick_action(self, state, act_space, obs_space):
        # Setup
        action = np.zeros(act_space.shape, dtype=np.int8)
        # Network Setup
        if self.actor is None: # All models created at same time, any being None indicates not initialized
            self._create_models(act_space, obs_space)
        # Action Creation
        if np.any(state[1:4] == 0): # Update to check non-decoded state (RL agent deosnt decode unless needed)
            # Deocde State
            dstate = agent_helpers.decode_state(state, self.num_characters)
            # Character Identity Guesses
            self._act_charGuess(dstate, action, act_space, obs_space)
        else:
            if self.invalid_last_act or (self.train and np.random.rand() <= self.eps): # Explore (random move selection) - Disable for trained results testing
                # Deocde State
                dstate = agent_helpers.decode_state(state, self.num_characters)
                # Character Die Moves
                self._act_dieMove(dstate, action, act_space, obs_space)
                # Action Card Selection
                self._act_actCards(dstate, action, act_space, obs_space)
                # Internal Var Updates
                self.invalid_last_act = False # Failed network use flags, clear after random action selected as follow up
            else: # Exploit (select action based on network q-values)
                # Generate Action from Actor Network
                rl_act = tf.squeeze(self.actor(state)).numpy()
                action = np.rint(rl_act).astype(np.int8) # Round to nearest int and cast
        # Return
        self.curr_state, self.curr_action = state, action # Track for replay buffer saves in update method
        return action

    def _act_charGuess(self, decoded_state, action, act_space, obs_space):
        randValidGuess(decoded_state, action)

    @tf.function # Decorated as Tensorflow function for execution optimization
    def _learn(self, batch_states, batch_actions, batch_rewards, batch_next_states):
        # Update Critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(batch_next_states, training=True)
            y = batch_rewards + self.gamma * self.target_critic( # TODO: reward not normalized - dominating and blocking leanring?
                [batch_next_states, target_actions], training=True
            )
            critic_value = self.critic([batch_states, batch_actions], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )
        # Update Actor
        with tf.GradientTape() as tape:
            actions = self.actor(batch_states, training=True)
            critic_value = self.critic([batch_states, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

    def _create_models(self, act_space, obs_space):
        # Setup
        hidden_actor_layers = 7
        hidden_critic_layers = 7

        # Normalization Setup
        act_norm_data = np.zeros((2, act_space.shape[0]), dtype=np.float32)
        act_norm_data[1] = act_space.nvec - 1 # Gym MultiDiscrete n vector is not inclusive (n = max + 1)

        obs_norm_data = np.zeros((2, obs_space.shape[0]), dtype=np.float32)
        obs_norm_data[1] = obs_space.nvec - 1 # Gym MultiDiscrete n vector is not inclusive (n = max + 1)

        obs_norm = layers.Normalization()
        obs_norm.adapt(obs_norm_data) # Seed mean/stddev for Normalization with full range of data

        crit_norm = layers.Normalization()
        crit_norm.adapt(np.concatenate((obs_norm_data, act_norm_data), axis=1))

        # Create Actor Models
        hidden_actor_step = 0 #int((obs_space.shape[0] - act_space.shape[0])/hidden_actor_layers)

        inputs = layers.Input(shape=(obs_space.shape[0],))
        norm_inputs = obs_norm(inputs)
        prev_layer = norm_inputs
        for i in range(0, hidden_actor_layers):
            outputs = layers.Dense(obs_space.shape[0] - i*hidden_actor_step, activation="relu")(prev_layer)
            prev_layer = outputs
        outputs = layers.Dense(act_space.shape[0], activation="sigmoid")(outputs)
        outputs *= act_norm_data[1]
        self.actor = tf.keras.Model(inputs, outputs)
        # print("Actor Model:")
        # print(self.actor.summary())
        # tf.keras.utils.plot_model(self.actor, to_file='../act.png', show_shapes=True) # Requires pydot and graphviz

        # Create Critic Models
        input_size_critic = obs_space.shape[0] + act_space.shape[0]
        hidden_critic_step = 0 #int(input_size_critic/hidden_critic_layers)

        input_state = layers.Input(shape=(obs_space.shape[0]))
        input_act = layers.Input(shape=(act_space.shape[0]))
        concat = layers.Concatenate()([input_state, input_act])
        norm_concat = crit_norm(concat)
        prev_layer = norm_concat
        for i in range(0, hidden_actor_layers):
            outputs = layers.Dense(input_size_critic - i*hidden_critic_step, activation="relu")(prev_layer)
            prev_layer = outputs
        outputs = layers.Dense(1, activation="sigmoid")(outputs)
        self.critic = tf.keras.Model([input_state, input_act], outputs)
        # print("Critic Model:")
        # print(self.critic.summary())

        # Load Weights
        # self.actor.load_weights('./checkpoints/ddpg_actor3')
        # self.critic.load_weights('./checkpoints/ddpg_critic3')

        # Copy Policy to Target
        self.target_actor = tf.keras.models.clone_model(self.actor)
        self.target_critic = tf.keras.models.clone_model(self.critic)


################################################################################
################################################################################
## Main
################################################################################
################################################################################
if __name__ == "__main__":
    print("No executable code, meant for use as module only")

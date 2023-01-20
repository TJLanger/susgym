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



################################################################################
################################################################################
## Classes
################################################################################
################################################################################

"""
description:
-> RL based agent, epsilon-greedy choice between DDPG network or stochastic (valid) actions
"""
class rlSusAgent():
    def __init__(self, eps=0.1, lr=0.001, gamma=0.99, tau=1, batch_size=64):
        # Basic Setup
        self.reward = 0
        self.num_players = None
        self.num_characters = 10
        self.curr_state, self.curr_action = None, None
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

    def pick_action(self, state, act_space, obs_space):
        # Setup
        action = np.zeros(act_space.shape, dtype=np.int8)
        # Network Setup
        if self.actor is None: # All models created at same time, any being None indicates not initialized
            self._create_models(act_space, obs_space)
        # State Decode
        num_players, ii, bank_gems, pg, char_locs, die_rolls, player_char, act_cards, knowledge = agent_helpers.decode_state(state, self.num_characters)
        if self.num_players is None: self.num_players = num_players
        # Action Creation
        if np.any(bank_gems == 0):
            # Character Identity Guesses
            agent_helpers.randActComp_charGuess(action, self.num_players, self.num_characters)
            # num_opps = self.num_players-1
            # for opp_idx in range(0, num_opps):
            #     valid_opp_chars = np.where(knowledge[opp_idx] == 1)[0]
            #     action[0-(num_opps-opp_idx)] = valid_opp_chars[np.random.randint(valid_opp_chars.size)]
        else:
            # Select Exploration vs Exploitation
            if np.random.rand() <= self.eps: # Explore (random move selection)
                # Character Die Moves
                agent_helpers.randActComp_dieMove(action, die_rolls, char_locs, self.num_characters)
                # Action Card Selection
                act_card_idx = np.random.randint(0, 2)
                act_order = np.random.randint(0, 2)
                agent_helpers.randActComp_actCards(action, state, act_card_idx, act_order, self.num_characters)
            else: # Exploit (select action based on network q-values)
                # Generate Action from Actor Network
                rl_act = tf.squeeze(self.actor(state)).numpy()
                action = np.rint(rl_act).astype(np.int8) # Round to nearest int and cast
        # Return
        self.curr_state, self.curr_action = state, action # Track for replay buffer saves in update method
        return action

    def update(self, next_state, reward, done, info):
        # Basic Updates
        self.reward += reward
        # Network Updates
        self.replay_memory.store((self.curr_state, self.curr_action, reward, next_state))
        # Generate Batch
        if len(self.replay_memory) > self.batch_size:
            batch_states, batch_actions, batch_rewards, batch_next_states = self.replay_memory.sample()
            batch_states = tf.convert_to_tensor(batch_states)
            batch_actions = tf.convert_to_tensor(batch_actions)
            batch_rewards = tf.convert_to_tensor(batch_rewards)
            batch_next_states = tf.convert_to_tensor(batch_next_states)
            self._learn(batch_states, batch_actions, batch_rewards, batch_next_states)

    @tf.function # Decorated as Tensorflow function for execution optimization
    def _learn(self, batch_states, batch_actions, batch_rewards, batch_next_states):
        # Update Critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(batch_next_states, training=True)
            y = batch_rewards + self.gamma * self.target_critic(
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

    def getReward(self):
        return self.reward

    def reset(self):
        # Basic Setup
        self.reward = 0
        # Model Setup
        if self.actor is not None:
            update_target(self.target_actor.variables, self.actor.variables, self.tau)
            update_target(self.target_critic.variables, self.critic.variables, self.tau)

    def save_weights(self):
        pass
        # self.actor.save_weights('../checkpoints/ddpg_actor3')
        # self.critic.save_weights('../checkpoints/ddpg_critic3')

    def _create_models(self, act_space, obs_space):
        # Setup
        hidden_actor_layers = 5
        hidden_critic_layers = 5

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
        hidden_actor_step = int((obs_space.shape[0] - act_space.shape[0])/hidden_actor_layers)

        inputs = layers.Input(shape=(obs_space.shape[0],))
        norm_inputs = obs_norm(inputs)
        prev_layer = norm_inputs
        for i in range(0, hidden_actor_layers):
            outputs = layers.Dense(obs_space.shape[0] - i*hidden_actor_step, activation="relu")(prev_layer)
            prev_layer = outputs
        outputs = layers.Dense(act_space.shape[0], activation="sigmoid")(outputs)
        outputs *= act_norm_data[1]
        self.actor = tf.keras.Model(inputs, outputs)

        # Create Critic Models
        input_size_critic = obs_space.shape[0] + act_space.shape[0]
        hidden_critic_step = int(input_size_critic/hidden_critic_layers)

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

        # Load Weights
        # self.actor.load_weights('../checkpoints/ddpg_actor3')
        # self.critic.load_weights('../checkpoints/ddpg_critic3')

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

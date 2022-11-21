#!/usr/bin/env python

################################################################################
################################################################################
## Authorship
################################################################################
################################################################################
##### Project
############### susgym
##### Name
############### example.py
##### Author
############### Trenton Langer
##### Creation Date
############### 20221117
##### Description
############### Basic gym control loop and testing
##### Project
############### YYYYMMDD - (CONTRIBUTOR) EXAMPLE



################################################################################
################################################################################
## Imports
################################################################################
################################################################################
# OpenAI gym
import gym
from suspicion_gym import suspicion_gym
from agents.rand_agent import randAgent

# Other
import argparse
import numpy as np
import time

# Debug
from pprint import pprint



################################################################################
################################################################################
## Global
################################################################################
################################################################################



################################################################################
################################################################################
## Subroutines
################################################################################
################################################################################

################################################################################
# name:             argParser
# description:      argument parser
# parameters:
#    No Params
# return:
#    args           dictionary with parsed arguments
################################################################################
def argParser():
    # setup
    parser = argparse.ArgumentParser(
        description = "Demo interaction for suspicion_gym and agents",
        prefix_chars = '+-',
    )
    parser.add_argument('-n', '--num_players', nargs='?', default=4, type=int)
    parser.add_argument('-e', '--episodes', nargs='?', default=1, type=int)
    parser.add_argument('-g', '--gui', action='store_false', default=False)
    parser.add_argument('+g', '++gui', action='store_true')
    parser.add_argument('-d', '--debug', action='store_false', default=True)
    parser.add_argument('+d', '++debug', action='store_true')
    # return
    return parser.parse_args()

################################################################################
# name:             playSuspicion
# description:      example control loop for Suspicion gym environment
# parameters:
#    No Params
# return:
#    args           dictionary with parsed arguments
################################################################################
def playSuspicion(num_players=4, gui=False, episodes=1, debug=False):
    # setup
    gui_size = 1000 if gui else None
    max_step = None
    # create env
    susEnv = gym.make("Suspicion-v1", num_players=num_players,
        gui_size=gui_size, gui_delay=1)
    # create agents
    agents = [randAgent() for i in range(num_players)]
    rewards = [[] for i in range(len(agents))]
    # loop epochs
    starttime = time.time()
    for episode in range(episodes):
        # init
        timestep = 0
        epoch_over = False
        state = susEnv.reset()
        for agent in agents:
            agent.reset()
        if debug:
            print("Initial State\n%s\n" % str(state))
        # interact
        while max_step is None or timestep < max_step: # no max step set, or below limit
            # validate
            if epoch_over:
                break
            # setup
            agent_idx, state = susEnv.observe() # Current ENV state, after self/opponent actions. Needs to update player specific state bits to current player
            agent = agents[agent_idx]
            if debug:
                print("\n\n\nTime: (%s),\tAgent: (%s)" % (str(timestep),str(agent_idx)))
                print("\tState: %s" % str(state))
            # pick action
            action = agent.pick_action(state, susEnv.action_space, susEnv.observation_space)
            if debug:
                print("Picking Action: %s, Len (%s)" % (str(action),str(len(action))))
            # apply action
            obs, reward, done, info = susEnv.step(action) # Obs = state after players turn
            agent.update(obs, reward, done, info)
            if debug:
                print("\tNew State: %s" % str(obs))
                print("\tReward: %s" % str(reward))
                print("\tDone: %s" % str(done))
                print("\tInfo: %s" % str(info))
            # visualize
            susEnv.render()
            # increment
            timestep += 1
        # cleanup
        for agent_idx in range(len(agents)):
            rewards[agent_idx].append(agents[agent_idx].getReward())
    # cleanup
    endtime = time.time()
    runtime = endtime - starttime
    if debug:
        print(time.strftime("Runtime: %H hours, %M minutes, %S seconds", time.gmtime(int(runtime))))
    # Return
    return rewards


################################################################################
################################################################################
## Classes
################################################################################
################################################################################



################################################################################
################################################################################
## Main
################################################################################
################################################################################
if __name__ == "__main__":
    # Parse arguments
    args = argParser()
    if args.debug: pprint(args)
    # Execute game
    out = playSuspicion(
        num_players = args.num_players,
        gui = args.gui,
        episodes = args.episodes,
        debug = args.debug
    )
    # Output
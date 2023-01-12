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
from agents.rand_agent import randSusAgent

# Other
import argparse
import importlib
import numpy as np
import os
import signal
import time

# Debug
from pprint import pprint



################################################################################
################################################################################
## Global
################################################################################
################################################################################

# Exceptions
class QuitEpisode(Exception):
    "Raised from sig handler to end episode loop but still print results"
    pass



################################################################################
################################################################################
## Subroutines
################################################################################
################################################################################

################################################################################
# name:             sigHandle
# description:      signal handler
# parameters:
#    signal_number  numeric code for specific signal number (default argument passed to handler)
#    stack_frame    None or a frame object (default argument passed to handler)
# return:
#    No return
################################################################################
def signal_handler(signal_number, stack_frame):
    raise QuitEpisode
signal.signal(signal.SIGINT, signal_handler)

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
    parser.add_argument('-a', '--agents', nargs='+', type=str) # No default, filled based on num_players if not given
    parser.add_argument('-n', '--num_players', nargs='?', default=3, type=int)
    parser.add_argument('-e', '--episodes', nargs='?', default=1, type=int)
    parser.add_argument('-g', '--gui', action='store_false', default=False)
    parser.add_argument('+g', '++gui', action='store_true')
    parser.add_argument('-d', '--debug', action='store_false', default=True)
    parser.add_argument('+d', '++debug', action='store_true')
    # Parse
    parsed = parser.parse_args()
    # Generate Agent List
    if parsed.agents is None: parsed.agents = []
    file_cnt = 0
    for idx in range(len(parsed.agents)):
        agent = parsed.agents[idx]
        if os.path.isfile(agent):
            parsed.agents.remove(agent)
            fh = open(agent, 'r')
            for line in fh.readlines():
                try:
                    module_name, class_name = line.strip().rsplit(".", 1)
                    inputAgent = getattr(importlib.import_module(module_name), class_name)
                    parsed.agents.insert(idx+file_cnt, inputAgent())
                    file_cnt += 1
                except Exception:
                    pass
        else:
            module_name, class_name = agent.rsplit(".", 1)
            inputAgent = getattr(importlib.import_module(module_name), class_name)
            parsed.agents[idx+file_cnt] = inputAgent()
    while len(parsed.agents) < parsed.num_players:
        parsed.agents.append(randSusAgent())
    # Return
    return parsed

################################################################################
# name:             playSuspicion
# description:      example control loop for Suspicion gym environment
# parameters:
#    No Params
# return:
#    args           dictionary with parsed arguments
################################################################################
def playSuspicion(agents, gui=False, episodes=1, debug=False):
    # setup
    gui_size = 1000 if gui else None
    max_step = None
    # create env
    susEnv = gym.make("Suspicion-v1", num_players=len(agents),
        gui_size=gui_size, gui_delay=1, debug=debug)
    # create agents
    rewards = np.zeros((episodes,len(agents)), dtype=np.float64) # float to allow future partial rewards
    # loop epochs
    try:
        for episode in range(episodes):
            # init
            timestep = 0
            epoch_over = False
            state = susEnv.reset()
            for agent in agents:
                agent.reset()
            dones = [False for i in range(len(agents))]
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
                if dones[agent_idx]:
                    break
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
                dones[agent_idx] = done
                # Validate
                if np.all(np.array(dones) == True):
                    break
            # cleanup
            for agent_idx in range(len(agents)):
                rewards[episode][agent_idx] = agents[agent_idx].getReward()
    except QuitEpisode:
        # Trim rewards array to played games
        sums = np.sum(rewards, axis=1)
        first_unplayed = np.where(sums == 0)[0][0] # All completed games have points - at least gem takes to end game
        rewards = rewards[0:first_unplayed]
    # Cleanup
    time.sleep(5)
    susEnv.cleanup()
    # Return
    return rewards

################################################################################
# name:             getResults
# description:      analyzes and prints tournament results
# parameters:
#    agents         list of agents playing the games
#    rewards        2d array of rewards (num agents x num episodes)
#    runtime        length of gameplay
#    debug          boolean flag for debug mode
# return:
#    results        string with gameplay information and analysis
################################################################################
def getResults(agents, rewards, runtime=None, debug=False):
    # setup
    results = "\n" if debug else ""
    num_episodes, num_agents = rewards.shape
    # analyze games
    #wins = np.argmax(rewards, axis=1) # doesnt account for ties, only first winner (bias towards agent 0, then 1, ...)
    wins = [list(np.argwhere(e == np.max(e)).flatten()) for e in rewards]
    # header
    results += "Tournament results (%s games played):\n" % str(num_episodes)
    # agent details
    for agent_idx in range(len(agents)):
        # Calcs
        num_wins = np.count_nonzero([len(x) == 1 and x[0] == agent_idx for x in wins])
        num_wins_and_ties = np.count_nonzero([agent_idx in x for x in wins])
        num_ties = num_wins_and_ties - num_wins
        win_rate = round(100*num_wins_and_ties/num_episodes, 2)
        # Prints
        results += "\tAgent %s: %s\n" % (str(agent_idx), str(agents[agent_idx].__class__))
        results += "\t\tNumber Wins: %s\n" % num_wins
        results += "\t\tNumber Ties: %s\n" % num_ties
        results += "\t\tWin Rate (Wins and ties): %s%%\n" % win_rate
        results += "\t\tAverage Score: %s\n" % str(np.average(rewards[:,agent_idx]))
        results += "\t\tMedian Score: %s\n" % str(np.median(rewards[:,agent_idx]))
        results += "\t\tMin Score: %s\n" % str(np.min(rewards[:,agent_idx]))
        results += "\t\tMax Score: %s\n" % str(np.max(rewards[:,agent_idx]))
    # runtime
    if runtime is None:
        pass
    elif runtime > 3600:
        results += time.strftime("Runtime: %H hours, %M minutes, %S seconds", time.gmtime(int(runtime))) + "\n"
    elif runtime > 60:
        results += time.strftime("Runtime: %M minutes, %S seconds", time.gmtime(int(runtime))) + "\n"
    else:
        results += time.strftime("Runtime: %S seconds", time.gmtime(int(runtime))) + "\n"
    # return
    return results



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
    starttime = time.time()
    rewards = playSuspicion(
        agents = args.agents,
        gui = args.gui,
        episodes = args.episodes,
        debug = args.debug
    )
    runtime = time.time() - starttime
    # Output
    print(getResults(args.agents, rewards, runtime, args.debug))

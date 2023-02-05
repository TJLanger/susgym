# susgym
Environment and agents for the board game Suspicion, based on OpenAI Gym

## Environment
The Suspicion environment maintains the game state, and applies actions provided by agents to that internal state. Following an applied action it returns the updated state and the associated reward. This reward can optionally be true to the game (positive only reward, given only in the terminal state), or expected reward impacts (positive and negative evaluations of decisions based on expected final score).

The environment is primarily interacted with using these methods:
- reset(): Initializes a new game, and updates internal variables accordingly
- observe(): Returns the agent index whose turn it is, and the current state
- step(action): Validates and applies an action provided by an agent, returning the new state and associated reward
- render(): Updates the selected visual (GUI or Terminal) with state information

The environment also has standard Gym attributes, for agents to interpret:
- action_space: Defines the format for valid actions that can be provided to the step() method
- observation_space: Defines the format for observations (states) that will be provided to an agent

## Agents
An agents primary function is to generate actions for the environment to apply, but agents are expected to support additional methods to facilitate reinforcement learning. It is best practice for agents to inherit from the susAgent class provided in the agent_helpers.py module, to ensure all required or typically used methods are defined. These methods are:
- pick_action(state, act_space, obs_space): Generates an action, formatted to match the action space, based on the current state
- update(obs, reward, done, info): Utilize the new state and associated reward information to improve the action selection policy
- reset(): Perform any operations needed prior to a game being played
- close(): Perform any operations needed prior to an agent reference going out of scope

## Playing Suspicion
The file example.py contains code to instantiate agents based on a text file or command line input, instantiate the Suspicion environment, play a specified number of games (episodes) - where the appropriate agent is queried for an action to be applied, and track and report on final game scores and cumulative rewards. The critical pseudocode for this is:

```
agents = [agent() for _ in range(4)] # Instantiate agents
susEnv = gym.make("Suspicion-v1", num_players=len(agents)) # Instantiate Environment
while True:
    agent_turn, state = susEnv.observe()
    action = agents[agent_turn].pick_action(state, susEnv.action_space, susEnv.observation_space)  # Pick Action
    obs, reward, done, info = susEnv.step(action) # Apply Action
    agents[agent_turn].update(obs, reward, done, info) # Update Agent Policy
    if all_agents_done:
        break
```

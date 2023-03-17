# Agents
Each file contains a different agent or set of agents for use in the Suspicion gym environment

## Agent Helpers
A set of shared functionality for faster agent development, and a generic base agent.

### Functions
- decode_state: parses state array into labeled dictionary components
- get_room_gems: list gems available in each room of the board
- get_valid_actions: generate 2D array of all valid actions
- randActComp_dieMove: Modify action array in-place for die roll movements
- randActComp_actCards: Modify action array in-place for action card selection and application
- randActComp_charGuess: Modify action array in-place for random identity guesses

### General Classes
- ReplayBuffer: Buffer for storing state/action/reward/state' tuples
    -  `__init__`(capacity, batch): Instantiate, setting max size and size of batch to be retrieved
    - store(obs_tuple): Insert tuple containing (state, action, reward, next_state) into buffer
    - sample(): Stochastically sample and return a batch of data
    - biased_sample(bias): Stochastically sample a batch of data, with a percentage focused on recent entries

### Base Agent
- susAgent: Agent template, using fully stochastic methods
    - update(next_state, reward, done, info): Method to provide environment output to agent
    - reset(): Method to update internal variables to initial conditions
    - close(): Method stub, inheriting agents can use to perform any needed cleanup operations
    - pick_action(state, act_space, obs_space): Method to generate action array, built on sub methods
        - _act_dieMove(...): Method for modifying action array for die moves, defaulting to randActComp_dieMove
        - _act_actCards(...): Method for modifying action array for action cards, defaulting to randActComp_dieMove
        - _act_charGuess(...): Method for modifying action array for identity guessing, defaulting to randActComp_dieMove

This base agent is designed to be inherited from, to expedite agent development. Inheriting agents can override a single action sub method to test particular feature improvements, but keeping other functionality consist.

## Domain Intelligent Agents
- validGuessSusAgent: Agent stochastically guessing character identities based on state information
    - Only overrides _act_charGuess method
- constraintGuessSusAgent: Agent using constraint optimization for identity guessing
    - Only overrides _act_charGuess method
- entropyAskSusAgent: Agent applying entropy/information calculations to choose action card applications
    - Overrides _act_actCards to choose action card targets, but also uses same _act_charGuess method as the validGuessSusAgent

## Reinforcement Learning Agents
- rlGuessSusAgent: Agent using a Deep Q-Learning approach for identity guessing
    - Overrides multiple agent methods to support training and use of RL network
- nfqSusAgent: Agent choosing from valid actions based on an NFQ/DQN algorithm
- ddpgSusAgent: Afent generating action arrays using a DDPG style algorithm

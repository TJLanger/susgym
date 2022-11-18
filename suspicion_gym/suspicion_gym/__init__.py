from gym.envs.registration import register

register(
    id='Suspicion-v1',
    entry_point='suspicion_game.envs:SuspicionEnv',
    max_episode_steps=10000,
)

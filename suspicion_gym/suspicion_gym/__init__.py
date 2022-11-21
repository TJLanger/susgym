from gym.envs.registration import register

register(
    id='Suspicion-v1',
    entry_point='suspicion_gym.suspicion_gym.envs:SuspicionEnv',
    max_episode_steps=10000,
)

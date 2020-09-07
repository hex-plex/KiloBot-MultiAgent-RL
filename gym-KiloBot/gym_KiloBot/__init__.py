from gym.envs.registration import register

register(
    id='KiloBot-v0',
    entry_point='gym_KiloBot.envs:KiloBotEnv',
)

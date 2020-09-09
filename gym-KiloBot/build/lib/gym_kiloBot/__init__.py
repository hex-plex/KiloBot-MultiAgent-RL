from gym.envs.registration import register

register(
    id='kiloBot-v0',
    entry_point='gym_kiloBot.envs:KiloBotEnv',
)

from gym.envs.registration import register

register(
    id='VirtualOffice-v0',
    entry_point='gym_virtual_office.envs:VirtualOfficeEnv',
    # max_episode_steps=5000,
    # reward_threshold=-100,
)
from gym.envs.registration import register

register(
    id="MyFetchPickAndPlace-v1",
    entry_point="my_fetch.pick_and_place:FetchPickAndPlaceEnv",
)

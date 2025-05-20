import os

from gymnasium.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

ENV_IDS = []

for task in ["Reach", "Slide", "Push", "PickAndPlace", "Stack", "Flip"]:
    for reward_type in ["sparse", "dense"]:
        for control_type in ["ee", "joints", "torques"]:
            for obs_type in ["ee", "joints"]:
                reward_suffix = "" if reward_type == "sparse" else reward_type.title()
                control_suffix = "" if control_type == "ee" else control_type.title()
                obs_suffix = "" if obs_type == "ee" else "-" + obs_type.title()

                env_id = f"Panda{task}{control_suffix}{reward_suffix}-v3{obs_suffix}"

                register(
                    id=env_id,
                    entry_point=f"panda_gym.envs:Panda{task}Env",
                    kwargs={"reward_type": reward_type, "control_type": control_type, "obs_type": obs_type},
                    max_episode_steps=100 if task == "Stack" else 50,
                )

                ENV_IDS.append(env_id)

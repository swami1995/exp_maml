from gym.envs.registration import load
from .normalized_env import NormalizedActionWrapper
from .normalized_env import NormalizedActionWrapperProMP
from .normalized_env import NormalizedObservationWrapper

def mujoco_wrapper(entry_point, promp_env=False, humanoid=False, **kwargs):
    # Load the environment from its entry point
    env_cls = load(entry_point)
    env = env_cls(**kwargs)
    # Normalization wrapper
    if promp_env:
        env = NormalizedActionWrapperProMP(env)
        if humanoid:
        	env = NormalizedObservationWrapper(env)
    else:
        env = NormalizedActionWrapper(env)

    return env

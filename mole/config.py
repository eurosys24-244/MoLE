import platform

assert int(platform.python_version_tuple()[0]) >= 3 and int(platform.python_version_tuple()[1]) >= 6, "Python version must be >= 3.6"

PatrickStarMoEScope = False

MOLE_ACTIVATED = False

CHECK_EP = False

OVERLAP_SHARDS = -1

DROP_P = -1

STEP = 0

USE_FP16 = False

MOVING_AVG = 0.01

POLICY = [0,1,2,3,4,5,6,7,8]

DEEPSPEED_PROFILE = False

FORCE_BALANCED_ROUTING = False

DS_CPU_ADAM = False

SYNC_OVERFLOW = True

SkipBS0 = False

SAVE_ROUTING = False

output_dir = None

PROFILE_REAL = False

MAX_SKIP_STEPS = 4

cache_cpu_tensors = True

def set_info(step):
    global STEP
    STEP = step

from enum import Enum

class Scheduler_Type(Enum):
    DDIM = 1
    EULER = 2
    LCM = 3
    DDPM = 4
 
class Model_Type(Enum):
    SDXL = 1
    SDXL_Turbo = 2
    LCM_SDXL = 3
    SD15 = 4
    SD21 = 5
    SD21_Turbo = 6
    SD14 = 7

class Gradient_Averaging_Type(Enum):
    NONE = 1
    EACH_ITER = 2
    ON_END = 3

class Epsilon_Update_Type(Enum):
    NONE = 1
    OVERRIDE = 2
    OPTIMIZE = 3
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from src.eunms import Model_Type, Scheduler_Type, Gradient_Averaging_Type, Epsilon_Update_Type

@dataclass
class RunConfig:
    model_type : Model_Type = Model_Type.SDXL_Turbo

    scheduler_type : Scheduler_Type = Scheduler_Type.EULER

    prompt: str = ""

    num_inference_steps: int = 4

    num_inversion_steps: int = 100

    opt_lr: float = 0.1

    opt_iters: int = 0

    opt_none_inference_steps: bool = False

    guidance_scale: float = 0.0

    # pipe_inversion: DiffusionPipeline = None

    # pipe_inference: DiffusionPipeline = None

    save_gpu_mem: bool = False

    do_reconstruction: bool = True

    loss_kl_lambda: float = 10.0

    max_num_aprox_steps_first_step: int = 1

    num_aprox_steps: int = 10

    inversion_max_step: float = 1.0

    gradient_averaging_type: Gradient_Averaging_Type = Gradient_Averaging_Type.NONE

    gradient_averaging_first_step_range: tuple = (0, 10)

    gradient_averaging_step_range: tuple = (0, 10)

    noise_friendly_inversion: bool = False

    update_epsilon_type: Epsilon_Update_Type = Gradient_Averaging_Type.NONE

    #pip2pip zero

    lambda_ac: float = 20.0

    lambda_kl: float = 20.0
    
    num_reg_steps: int = 5

    num_ac_rolls: int = 5

    def __post_init__(self):
        pass
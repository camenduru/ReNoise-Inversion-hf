
import torch
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline, AutoPipelineForImage2Image

from src.eunms import Model_Type, Scheduler_Type
from src.euler_scheduler import MyEulerAncestralDiscreteScheduler
from src.lcm_scheduler import MyLCMScheduler
from src.ddpm_scheduler import MyDDPMScheduler
from src.sdxl_inversion_pipeline import SDXLDDIMPipeline
from src.sd_inversion_pipeline import SDDDIMPipeline
    
def scheduler_type_to_class(scheduler_type):
    if scheduler_type == Scheduler_Type.DDIM:
        return DDIMScheduler
    elif scheduler_type == Scheduler_Type.EULER:
        return MyEulerAncestralDiscreteScheduler
    elif scheduler_type == Scheduler_Type.LCM:
        return MyLCMScheduler
    elif scheduler_type == Scheduler_Type.DDPM:
        return MyDDPMScheduler
    else:
        raise ValueError("Unknown scheduler type")
    
def model_type_to_class(model_type):
    if model_type == Model_Type.SDXL:
        return StableDiffusionXLImg2ImgPipeline, SDXLDDIMPipeline
    elif model_type == Model_Type.SDXL_Turbo:
        return AutoPipelineForImage2Image, SDXLDDIMPipeline
    elif model_type == Model_Type.LCM_SDXL:
        return AutoPipelineForImage2Image, SDXLDDIMPipeline
    elif model_type == Model_Type.SD15:
        return StableDiffusionImg2ImgPipeline, SDDDIMPipeline
    elif model_type == Model_Type.SD14:
        return StableDiffusionImg2ImgPipeline, SDDDIMPipeline
    elif model_type == Model_Type.SD21:
        return StableDiffusionImg2ImgPipeline, SDDDIMPipeline
    elif model_type == Model_Type.SD21_Turbo:
        return StableDiffusionImg2ImgPipeline, SDDDIMPipeline
    else:
        raise ValueError("Unknown model type")
    
def model_type_to_model_name(model_type):
    if model_type == Model_Type.SDXL:
        return "stabilityai/stable-diffusion-xl-base-1.0"
    elif model_type == Model_Type.SDXL_Turbo:
        return "stabilityai/sdxl-turbo"
    elif model_type == Model_Type.LCM_SDXL:
        return "stabilityai/stable-diffusion-xl-base-1.0"
    elif model_type == Model_Type.SD15:
        return "runwayml/stable-diffusion-v1-5"
    elif model_type == Model_Type.SD14:
        return "CompVis/stable-diffusion-v1-4"
    elif model_type == Model_Type.SD21:
        return "stabilityai/stable-diffusion-2-1"
    elif model_type == Model_Type.SD21_Turbo:
        return "stabilityai/sd-turbo"
    else:
        raise ValueError("Unknown model type")

    
def model_type_to_size(model_type):
    if model_type == Model_Type.SDXL:
        return (1024, 1024)
    elif model_type == Model_Type.SDXL_Turbo:
        return (512, 512)
    elif model_type == Model_Type.LCM_SDXL:
        return (768, 768) #TODO: check
    elif model_type == Model_Type.SD15:
        return (512, 512)
    elif model_type == Model_Type.SD14:
        return (512, 512)
    elif model_type == Model_Type.SD21:
        return (512, 512)
    elif model_type == Model_Type.SD21_Turbo:
        return (512, 512)
    else:
        raise ValueError("Unknown model type")
    
def is_float16(model_type):
    if model_type == Model_Type.SDXL:
        return True
    elif model_type == Model_Type.SDXL_Turbo:
        return True
    elif model_type == Model_Type.LCM_SDXL:
        return True
    elif model_type == Model_Type.SD15:
        return False
    elif model_type == Model_Type.SD14:
        return False
    elif model_type == Model_Type.SD21:
        return False
    elif model_type == Model_Type.SD21_Turbo:
        return False
    else:
        raise ValueError("Unknown model type")
    
def is_sd(model_type):
    if model_type == Model_Type.SDXL:
        return False
    elif model_type == Model_Type.SDXL_Turbo:
        return False
    elif model_type == Model_Type.LCM_SDXL:
        return False
    elif model_type == Model_Type.SD15:
        return True
    elif model_type == Model_Type.SD14:
        return True
    elif model_type == Model_Type.SD21:
        return True
    elif model_type == Model_Type.SD21_Turbo:
        return True
    else:
        raise ValueError("Unknown model type")
    
def _get_pipes(model_type, device):
    model_name = model_type_to_model_name(model_type)
    pipeline_inf, pipeline_inv = model_type_to_class(model_type)

    if is_float16(model_type):
        pipe_inversion = pipeline_inv.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                safety_checker = None
            ).to(device)

        pipe_inference = pipeline_inf.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                safety_checker = None
            ).to(device)
    else:        
        pipe_inversion = pipeline_inv.from_pretrained(
                model_name,
                use_safetensors=True,
                safety_checker = None
            ).to(device)

        pipe_inference = pipeline_inf.from_pretrained(
                model_name,
                use_safetensors=True,
                safety_checker = None
            ).to(device)
    
    return pipe_inversion, pipe_inference
    
def get_pipes(model_type, scheduler_type, device="cuda"):
    # model_name = model_type_to_model_name(model_type)
    # pipeline_inf, pipeline_inv = model_type_to_class(model_type)
    scheduler_class = scheduler_type_to_class(scheduler_type)

    pipe_inversion, pipe_inference = _get_pipes(model_type, device)

    # pipe_inversion = pipeline_inv.from_pretrained(
    #         model_name,
    #         # torch_dtype=torch.float16,
    #         use_safetensors=True,
    #         # variant="fp16",
    #         safety_checker = None
    #     ).to("cuda")

    # pipe_inference = pipeline_inf.from_pretrained(
    #         model_name,
    #         # torch_dtype=torch.float16,
    #         use_safetensors=True,
    #         # variant="fp16",
    #         safety_checker = None
    #     ).to("cuda")
    
    pipe_inference.scheduler            = scheduler_class.from_config(pipe_inference.scheduler.config)
    pipe_inversion.scheduler            = scheduler_class.from_config(pipe_inversion.scheduler.config)
    pipe_inversion.scheduler_inference  = scheduler_class.from_config(pipe_inference.scheduler.config)

    if is_sd(model_type):
        pipe_inference.scheduler.add_noise = lambda init_latents, noise, timestep: init_latents
        pipe_inversion.scheduler.add_noise = lambda init_latents, noise, timestep: init_latents
        pipe_inversion.scheduler_inference.add_noise = lambda init_latents, noise, timestep: init_latents

    if model_type == Model_Type.LCM_SDXL:
        adapter_id = "latent-consistency/lcm-lora-sdxl"
        # load and fuse lcm lora
        pipe_inversion.load_lora_weights(adapter_id)
        # pipe_inversion.fuse_lora()
        pipe_inference.load_lora_weights(adapter_id)
        # pipe_inference.fuse_lora()

    return pipe_inversion, pipe_inference
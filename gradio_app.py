from __future__ import annotations

import gradio as gr
from PIL import Image
import torch

from src.eunms import Model_Type, Scheduler_Type, Gradient_Averaging_Type, Epsilon_Update_Type
from src.enums_utils import model_type_to_size, get_pipes
from src.config import RunConfig
from main import run as run_model


DESCRIPTION = '''# ReNoise: Real Image Inversion Through Iterative Noising
This is a demo for our ''ReNoise: Real Image Inversion Through Iterative Noising'' [paper](https://garibida.github.io/ReNoise-Inversion/). Code is available [here](https://github.com/garibida/ReNoise-Inversion)
Our ReNoise inversion technique can be applied to various diffusion models, including recent few-step ones such as SDXL-Turbo.
This demo preform real image editing using our ReNoise inversion. The input image is resize to size of 512x512, the optimal size of SDXL Turbo.
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_type = Model_Type.SDXL_Turbo
scheduler_type = Scheduler_Type.EULER
image_size = model_type_to_size(Model_Type.SDXL_Turbo)
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)

cache_size = 10
prev_configs = [None for i in range(cache_size)]
prev_inv_latents = [None for i in range(cache_size)]
prev_images = [None for i in range(cache_size)]
prev_noises = [None for i in range(cache_size)]

def main_pipeline(
        input_image: str,
        src_prompt: str,
        tgt_prompt: str,
        edit_cfg: float,
        number_of_renoising_iterations: int,
        inersion_strength: float,
        avg_gradients: bool,
        first_step_range_start: int,
        first_step_range_end: int,
        rest_step_range_start: int,
        rest_step_range_end: int,
        lambda_ac: float,
        lambda_kl: float,
        noise_correction: bool):

        global prev_configs, prev_inv_latents, prev_images, prev_noises

        update_epsilon_type = Epsilon_Update_Type.OPTIMIZE if noise_correction else Epsilon_Update_Type.NONE
        avg_gradients_type = Gradient_Averaging_Type.ON_END if avg_gradients else Gradient_Averaging_Type.NONE

        first_step_range = (first_step_range_start, first_step_range_end)
        rest_step_range = (rest_step_range_start, rest_step_range_end)

        config = RunConfig(model_type = model_type,
                    num_inference_steps = 4,
                    num_inversion_steps = 4, 
                    guidance_scale = 0.0,
                    max_num_aprox_steps_first_step = first_step_range_end+1,
                    num_aprox_steps = number_of_renoising_iterations,
                    inversion_max_step = inersion_strength,
                    gradient_averaging_type = avg_gradients_type,
                    gradient_averaging_first_step_range = first_step_range,
                    gradient_averaging_step_range = rest_step_range,
                    scheduler_type = scheduler_type,
                    num_reg_steps = 4,
                    num_ac_rolls = 5,
                    lambda_ac = lambda_ac,
                    lambda_kl = lambda_kl,
                    update_epsilon_type = update_epsilon_type,
                    do_reconstruction = True)
        config.prompt = src_prompt

        inv_latent = None
        noise_list = None
        for i in range(cache_size):
            if prev_configs[i] is not None and prev_configs[i] == config and prev_images[i] == input_image:
                print(f"Using cache for config #{i}")
                inv_latent = prev_inv_latents[i]
                noise_list = prev_noises[i]
                prev_configs.pop(i)
                prev_inv_latents.pop(i)
                prev_images.pop(i)
                prev_noises.pop(i)
                break

        original_image = Image.open(input_image).convert("RGB").resize(image_size)

        res_image, inv_latent, noise, all_latents = run_model(original_image,
                                    config,
                                    latents=inv_latent,
                                    pipe_inversion=pipe_inversion,
                                    pipe_inference=pipe_inference,
                                    edit_prompt=tgt_prompt,
                                    noise=noise_list,
                                    edit_cfg=edit_cfg)

        prev_configs.append(config)
        prev_inv_latents.append(inv_latent)
        prev_images.append(input_image)
        prev_noises.append(noise)
        
        if len(prev_configs) > cache_size:
            print("Popping cache")
            prev_configs.pop(0)
            prev_inv_latents.pop(0)
            prev_images.pop(0)
            prev_noises.pop(0)

        return res_image


with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    gr.HTML(
        '''<a href="https://huggingface.co/spaces/garibida/ReNoise-Inversion?duplicate=true">
        <img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>Duplicate the Space to run privately without waiting in queue''')

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input image",
                type="filepath",
                height=image_size[0],
                width=image_size[1]
            )
            src_prompt = gr.Text(
                label='Source Prompt',
                max_lines=1,
                placeholder='A kitten is sitting in a basket on a branch',
            )
            tgt_prompt = gr.Text(
                label='Target Prompt',
                max_lines=1,
                placeholder='A plush toy kitten is sitting in a basket on a branch',
            )
            with gr.Accordion("Advanced Options", open=False):
                edit_cfg = gr.Slider(
                    label='Denoise Classifier-Free Guidence Scale',
                    minimum=1.0,
                    maximum=3.5,
                    value=1.0,
                    step=0.1
                )
                number_of_renoising_iterations = gr.Slider(
                    label='Number of ReNoise Iterations',
                    minimum=0,
                    maximum=20,
                    value=9,
                    step=1
                )
                inersion_strength = gr.Slider(
                    label='Inversion Strength',
                    info="Indicates how much to invert the reference image. The number of denoising steps depends on the amount of noise initially added. When strength is 1, the image will be inverted to complete noise and the denoising process will run for the full number of steps (4). When strength is 0.5, the image will be inverted to half noise and the denoising process will run for 2 steps.",
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.25
                )
                avg_gradients = gr.Checkbox(
                    label="Preform Estimation Averaging",
                    info="IMPROVES RECONSTRUCTION. Averagin the estination over multiple ReNoise iterations can improve the quality of the reconstruction. The Next 4 sliders control the range of steps to average over. The first two sliders control the range of steps to average over for the first inversion step (t < 250). The last two sliders control the range of steps to average over for the rest of the inversion step (t > 250).",
                    value=True
                )
                first_step_range_start = gr.Slider(
                    label='First Estimation in Average (t < 250)',
                    minimum=0,
                    maximum=21,
                    value=0,
                    step=1
                )
                first_step_range_end = gr.Slider(
                    label='Last Estimation in Average (t < 250)',
                    minimum=0,
                    maximum=21,
                    value=5,
                    step=1
                )
                rest_step_range_start = gr.Slider(
                    label='First Estimation in Average (t > 250)',
                    minimum=0,
                    maximum=21,
                    value=8,
                    step=1
                )
                rest_step_range_end = gr.Slider(
                    label='Last Estimation in Average (t > 250)',
                    minimum=0,
                    maximum=21,
                    value=10,
                    step=1
                )
                num_reg_steps = 4
                num_ac_rolls = 5
                lambda_ac = gr.Slider(
                    label='Labmda AC',
                    info="IMPROVES EDITABILITY. The weight of the pair loss in the noise prediction regulariztion. This loss encourages the inversion to predict more editable noise. A higher value allows more significant changes to the image (higher editability), but may result in less faithful reconstructions.",
                    minimum=0.0,
                    maximum=50.0,
                    value=20.0,
                    step=1.0
                )
                lambda_kl = gr.Slider(
                    label='Labmda Patch KL',
                    info="IMPROVES EDITABILITY. This weight controls the strength of the patch-level KL divergence term in the noise prediction regularization.  While it encourages editable noise like the Labmda AC, it often has a less detrimental effect on reconstruction fidelity.",
                    minimum=0.0,
                    maximum=0.4,
                    value=0.065,
                    step=0.005
                )
                noise_correction = gr.Checkbox(
                    label="Preform Noise Correction",
                    info="IMPROVES RECONSTRUCTION. Performs noise correction to improve the reconstruction of the image.",
                    value=True
                )

            run_button = gr.Button('Edit')
        with gr.Column():
            # result = gr.Gallery(label='Result')
            result = gr.Image(
                label="Result",
                type="pil",
                height=image_size[0],
                width=image_size[1]
            )

            examples = [
                [
                    "example_images/kitten.jpg", #input_image
                    "A kitten is sitting in a basket on a branch", #src_prompt
                    "a lego kitten is sitting in a basket on a branch", #tgt_prompt
                    1.0, #edit_cfg
                    9, #number_of_renoising_iterations
                    1.0, #inersion_strength
                    True, #avg_gradients
                    0, #first_step_range_start
                    5, #first_step_range_end
                    8, #rest_step_range_start
                    10, #rest_step_range_end
                    20.0, #lambda_ac
                    0.055, #lambda_kl
                    False #noise_correction
                ],
                [
                    "example_images/kitten.jpg", #input_image
                    "A kitten is sitting in a basket on a branch", #src_prompt
                    "a brokkoli is sitting in a basket on a branch", #tgt_prompt
                    1.0, #edit_cfg
                    9, #number_of_renoising_iterations
                    1.0, #inersion_strength
                    True, #avg_gradients
                    0, #first_step_range_start
                    5, #first_step_range_end
                    8, #rest_step_range_start
                    10, #rest_step_range_end
                    20.0, #lambda_ac
                    0.055, #lambda_kl
                    False #noise_correction
                ],
                [
                    "example_images/kitten.jpg", #input_image
                    "A kitten is sitting in a basket on a branch", #src_prompt
                    "a dog is sitting in a basket on a branch", #tgt_prompt
                    1.0, #edit_cfg
                    9, #number_of_renoising_iterations
                    1.0, #inersion_strength
                    True, #avg_gradients
                    0, #first_step_range_start
                    5, #first_step_range_end
                    8, #rest_step_range_start
                    10, #rest_step_range_end
                    20.0, #lambda_ac
                    0.055, #lambda_kl
                    False #noise_correction
                ],
                [
                    "example_images/monkey.jpeg", #input_image
                    "a monkey sitting on a tree branch in the forest", #src_prompt
                    "a beaver sitting on a tree branch in the forest", #tgt_prompt
                    1.0, #edit_cfg
                    9, #number_of_renoising_iterations
                    1.0, #inersion_strength
                    True, #avg_gradients
                    0, #first_step_range_start
                    5, #first_step_range_end
                    8, #rest_step_range_start
                    10, #rest_step_range_end
                    20.0, #lambda_ac
                    0.055, #lambda_kl
                    True #noise_correction
                ],
                [
                    "example_images/monkey.jpeg", #input_image
                    "a monkey sitting on a tree branch in the forest", #src_prompt
                    "a raccoon sitting on a tree branch in the forest", #tgt_prompt
                    1.0, #edit_cfg
                    9, #number_of_renoising_iterations
                    1.0, #inersion_strength
                    True, #avg_gradients
                    0, #first_step_range_start
                    5, #first_step_range_end
                    8, #rest_step_range_start
                    10, #rest_step_range_end
                    20.0, #lambda_ac
                    0.055, #lambda_kl
                    True #noise_correction
                ],
                [
                    "example_images/lion.jpeg", #input_image
                    "a lion is sitting in the grass at sunset", #src_prompt
                    "a tiger is sitting in the grass at sunset", #tgt_prompt
                    1.0, #edit_cfg
                    9, #number_of_renoising_iterations
                    1.0, #inersion_strength
                    True, #avg_gradients
                    0, #first_step_range_start
                    5, #first_step_range_end
                    8, #rest_step_range_start
                    10, #rest_step_range_end
                    20.0, #lambda_ac
                    0.055, #lambda_kl
                    True #noise_correction
                ]
            ]

            gr.Examples(examples=examples,
                        inputs=[
                            input_image,
                            src_prompt,
                            tgt_prompt,
                            edit_cfg,
                            number_of_renoising_iterations,
                            inersion_strength,
                            avg_gradients,
                            first_step_range_start,
                            first_step_range_end,
                            rest_step_range_start,
                            rest_step_range_end,
                            lambda_ac,
                            lambda_kl,
                            noise_correction
                        ],
                        outputs=[
                            result
                        ],
                        fn=main_pipeline,
                        cache_examples=True)


    inputs = [
        input_image,
        src_prompt,
        tgt_prompt,
        edit_cfg,
        number_of_renoising_iterations,
        inersion_strength,
        avg_gradients,
        first_step_range_start,
        first_step_range_end,
        rest_step_range_start,
        rest_step_range_end,
        lambda_ac,
        lambda_kl,
        noise_correction
    ]
    outputs = [
        result
    ]
    run_button.click(fn=main_pipeline, inputs=inputs, outputs=outputs)

demo.queue(max_size=50).launch(share=False)
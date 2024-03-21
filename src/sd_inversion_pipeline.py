# Plug&Play Feature Injection

import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from random import randrange
import PIL
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import custom_bwd, custom_fwd
import torch.nn.functional as F


from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DDIMScheduler,
)
from diffusers.utils.torch_utils import randn_tensor

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipelineOutput,
    retrieve_timesteps,
    PipelineImageInput
)

from src.eunms import Scheduler_Type, Gradient_Averaging_Type, Epsilon_Update_Type

def _backward_ddim(x_tm1, alpha_t, alpha_tm1, eps_xt):
    """
    let a = alpha_t, b = alpha_{t - 1}
    We have a > b,
    x_{t} - x_{t - 1} = sqrt(a) ((sqrt(1/b) - sqrt(1/a)) * x_{t-1} + (sqrt(1/a - 1) - sqrt(1/b - 1)) * eps_{t-1})
    From https://arxiv.org/pdf/2105.05233.pdf, section F.
    """

    a, b = alpha_t, alpha_tm1
    sa = a**0.5
    sb = b**0.5

    return sa * ((1 / sb) * x_tm1 + ((1 / a - 1) ** 0.5 - (1 / b - 1) ** 0.5) * eps_xt)


class SDDDIMPipeline(StableDiffusionImg2ImgPipeline):
    # @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        strength: float = 1.0,
        num_inversion_steps: Optional[int] = 50,
        timesteps: List[int] = None,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        opt_lr: float = 0.001,
        opt_iters: int = 1,
        opt_none_inference_steps: bool = False,
        opt_loss_kl_lambda: float = 10.0,
        num_inference_steps: int = 50,
        num_aprox_steps: int = 100,
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

         # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None:
            image_embeds, negative_image_embeds = self.encode_image(ip_adapter_image, device, num_images_per_prompt)
            if self.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])

        # 4. Preprocess image
        image = self.image_processor.preprocess(image)

        # 5. set timesteps
        timesteps, num_inversion_steps = retrieve_timesteps(self.scheduler, num_inversion_steps, device, timesteps)
        timesteps, num_inversion_steps = self.get_timesteps(num_inversion_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        _, num_inference_steps = retrieve_timesteps(self.scheduler_inference, num_inference_steps, device, None)

        # 6. Prepare latent variables
        with torch.no_grad():
            latents = self.prepare_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
            )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

        # 7.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        prev_timestep = None
        self.prev_z = torch.clone(latents)
        self.prev_z4 = torch.clone(latents)
        self.z_0 = torch.clone(latents)
        g_cpu = torch.Generator().manual_seed(7865)
        self.noise = randn_tensor(self.z_0.shape, generator=g_cpu, device=self.z_0.device, dtype=self.z_0.dtype)


        all_latents = [latents.clone()]
        with self.progress_bar(total=num_inversion_steps) as progress_bar:
            for i, t in enumerate(reversed(timesteps)):

                z_tp1 = self.inversion_step(latents,
                                            t,
                                            prompt_embeds,
                                            added_cond_kwargs,
                                            prev_timestep=prev_timestep,
                                            num_aprox_steps=num_aprox_steps)

                if t in self.scheduler_inference.timesteps:
                    z_tp1 = self.optimize_z_tp1(z_tp1, 
                                                latents, 
                                                t, 
                                                prompt_embeds, 
                                                added_cond_kwargs, 
                                                nom_opt_iters=opt_iters, 
                                                lr=opt_lr, 
                                                opt_loss_kl_lambda=opt_loss_kl_lambda)
                                        
                prev_timestep = t
                latents = z_tp1
                    
                all_latents.append(latents.clone())

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None), all_latents
    
    def noise_regularization(self, e_t, noise_pred_optimal):
        for _outer in range(self.cfg.num_reg_steps):
            if self.cfg.lambda_kl>0:
                _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                # l_kld = self.kl_divergence(_var)
                l_kld = self.patchify_latents_kl_divergence(_var, noise_pred_optimal)
                l_kld.backward()
                _grad = _var.grad.detach()
                _grad = torch.clip(_grad, -100, 100)
                e_t = e_t - self.cfg.lambda_kl*_grad
            if self.cfg.lambda_ac>0:
                for _inner in range(self.cfg.num_ac_rolls):
                    _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                    l_ac = self.auto_corr_loss(_var)
                    l_ac.backward()
                    _grad = _var.grad.detach()/self.cfg.num_ac_rolls
                    e_t = e_t - self.cfg.lambda_ac*_grad
            e_t = e_t.detach()

        return e_t

    def auto_corr_loss(self, x, random_shift=True):
        B,C,H,W = x.shape
        assert B==1
        x = x.squeeze(0)
        # x must be shape [C,H,W] now
        reg_loss = 0.0
        for ch_idx in range(x.shape[0]):
            noise = x[ch_idx][None, None,:,:]
            while True:
                if random_shift: roll_amount = randrange(noise.shape[2]//2)
                else: roll_amount = 1
                reg_loss += (noise*torch.roll(noise, shifts=roll_amount, dims=2)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=roll_amount, dims=3)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        return reg_loss
    
    def kl_divergence(self, x):
        _mu = x.mean()
        _var = x.var()
        return _var + _mu**2 - 1 - torch.log(_var+1e-7)

    # @torch.no_grad()
    def inversion_step(
        self,
        z_t: torch.tensor,
        t: torch.tensor,
        prompt_embeds,
        added_cond_kwargs,
        prev_timestep: Optional[torch.tensor] = None,
        num_aprox_steps: int = 100
    ) -> torch.tensor:
        extra_step_kwargs = {}

        avg_range = self.cfg.gradient_averaging_first_step_range if t.item() < 250 else self.cfg.gradient_averaging_step_range

        # When doing more then one approximation step in the first step it adds artifacts
        if t.item() < 250:
            num_aprox_steps = min(self.cfg.max_num_aprox_steps_first_step, num_aprox_steps)

        approximated_z_tp1 = z_t.clone()
        nosie_pred_avg = None

        if self.cfg.num_reg_steps > 0:
            z_tp1_forward = self.scheduler.add_noise(self.z_0, self.noise, t.view((1))).detach()
            latent_model_input = torch.cat([z_tp1_forward] * 2) if self.do_classifier_free_guidance else z_tp1_forward
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                # predict the noise residual
                noise_pred_optimal = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=None,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0].detach()
        else:
            noise_pred_optimal = None

        for i in range(num_aprox_steps + 1):
            latent_model_input = torch.cat([approximated_z_tp1] * 2) if self.do_classifier_free_guidance else approximated_z_tp1
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=None,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

            if  i >= avg_range[0] and i < avg_range[1]:
                j = i - avg_range[0]
                if nosie_pred_avg is None:
                    nosie_pred_avg = noise_pred.clone()
                else:
                    nosie_pred_avg = j * nosie_pred_avg / (j + 1) + noise_pred / (j + 1)
                if self.cfg.gradient_averaging_type == Gradient_Averaging_Type.EACH_ITER:
                    noise_pred = nosie_pred_avg.clone()

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if i >= avg_range[0] or (self.cfg.gradient_averaging_type == Gradient_Averaging_Type.NONE and i > 0):
                noise_pred = self.noise_regularization(noise_pred, noise_pred_optimal)
            
            if self.cfg.scheduler_type == Scheduler_Type.EULER:
                approximated_z_tp1 = self.scheduler.inv_step(noise_pred, t, z_t, **extra_step_kwargs, return_dict=False)[0].detach()
            else:
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[prev_timestep]
                    if prev_timestep is not None
                    else self.scheduler.final_alpha_cumprod
                )
                approximated_z_tp1 = _backward_ddim(
                    x_tm1=z_t,
                    alpha_t=alpha_prod_t,
                    alpha_tm1=alpha_prod_t_prev,
                    eps_xt=noise_pred,
                )

        if self.cfg.gradient_averaging_type == Gradient_Averaging_Type.ON_END and nosie_pred_avg is not None:
            
            nosie_pred_avg = self.noise_regularization(nosie_pred_avg, noise_pred_optimal)
            if self.cfg.scheduler_type == Scheduler_Type.EULER:
                approximated_z_tp1 = self.scheduler.inv_step(nosie_pred_avg, t, z_t, **extra_step_kwargs, return_dict=False)[0].detach()
            else:
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[prev_timestep]
                    if prev_timestep is not None
                    else self.scheduler.final_alpha_cumprod
                )
                approximated_z_tp1 = _backward_ddim(
                    x_tm1=z_t,
                    alpha_t=alpha_prod_t,
                    alpha_tm1=alpha_prod_t_prev,
                    eps_xt=nosie_pred_avg,
                )

        if self.cfg.update_epsilon_type != Epsilon_Update_Type.NONE:
            latent_model_input = torch.cat([approximated_z_tp1] * 2) if self.do_classifier_free_guidance else approximated_z_tp1
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=None,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            self.scheduler.step_and_update_noise(noise_pred, t, approximated_z_tp1, z_t, return_dict=False, update_epsilon_type=self.cfg.update_epsilon_type)

        return approximated_z_tp1
    
    def detach_before_opt(self, z_tp1, t, prompt_embeds, added_cond_kwargs):
        z_tp1 = z_tp1.detach()
        t = t.detach()
        prompt_embeds = prompt_embeds.detach()
        return z_tp1, t, prompt_embeds, added_cond_kwargs
    
    def opt_z_tp1_single_step(
        self,
        z_tp1,
        z_t,
        t,
        prompt_embeds,
        added_cond_kwargs,
        lr=0.001,
        opt_loss_kl_lambda=10.0,
    ):
        l1_loss = torch.nn.L1Loss(reduction='sum')
        mse = torch.nn.MSELoss(reduction='sum')
        extra_step_kwargs = {}
        
        self.unet.requires_grad_(False)
        z_tp1, t, prompt_embeds, added_cond_kwargs = self.detach_before_opt(z_tp1, t, prompt_embeds, added_cond_kwargs)
        
        z_tp1 = torch.nn.Parameter(z_tp1, requires_grad=True)
        optimizer = torch.optim.SGD([z_tp1], lr=lr, momentum=0.9)

        optimizer.zero_grad()
        self.unet.zero_grad()
        latent_model_input = torch.cat([z_tp1] * 2) if self.do_classifier_free_guidance else z_tp1
        latent_model_input = self.scheduler_inference.scale_model_input(latent_model_input, t)

        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=self.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # perform guidance
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # # compute the previous noisy sample x_t -> x_t-1
        z_t_hat = self.scheduler_inference.step(noise_pred, t, z_tp1, **extra_step_kwargs, return_dict=False)[0]

        direct_loss = 0.5 * mse(z_t_hat, z_t.detach()) + 0.5 * l1_loss(z_t_hat, z_t.detach())
        kl_loss = torch.tensor([0]).to(z_t.device)
        loss = 1.0 * direct_loss + opt_loss_kl_lambda * kl_loss
        
        loss.backward()
        optimizer.step()
        print(f't: {t}\t total_loss: {format(loss.item(), ".3f")}\t\t direct_loss: {format(direct_loss.item(), ".3f")}\t\t kl_loss: {format(kl_loss.item(), ".3f")}')

        return z_tp1.detach()
    
    def optimize_z_tp1(
        self,
        z_tp1,
        z_t,
        t,
        prompt_embeds,
        added_cond_kwargs,
        nom_opt_iters=1,
        lr=0.001,
        opt_loss_kl_lambda=10.0,
    ):
        l1_loss = torch.nn.L1Loss(reduction='sum')
        mse = torch.nn.MSELoss(reduction='sum')
        extra_step_kwargs = {}
        
        self.unet.requires_grad_(False)
        z_tp1, t, prompt_embeds, added_cond_kwargs = self.detach_before_opt(z_tp1, t, prompt_embeds, added_cond_kwargs)
        
        z_tp1 = torch.nn.Parameter(z_tp1, requires_grad=True)
        optimizer = torch.optim.SGD([z_tp1], lr=lr, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, verbose=True, patience=5, cooldown=3)
        max_loss = 99999999999999

        z_tp1_forward = self.scheduler.add_noise(self.z_0, self.noise, t.view((1))).detach()
        z_tp1_best = None
        for i in range(nom_opt_iters):
            optimizer.zero_grad()
            self.unet.zero_grad()
            latent_model_input = torch.cat([z_tp1] * 2) if self.do_classifier_free_guidance else z_tp1
            latent_model_input = self.scheduler_inference.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=None,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # # compute the previous noisy sample x_t -> x_t-1
            z_t_hat = self.scheduler_inference.step(noise_pred, t, z_tp1, **extra_step_kwargs, return_dict=False)[0]

            direct_loss = 0.5 * mse(z_t_hat, z_t.detach()) + 0.5 * l1_loss(z_t_hat, z_t.detach())
            kl_loss = self.patchify_latents_kl_divergence(z_tp1, z_tp1_forward)
            loss = 1.0 * direct_loss + opt_loss_kl_lambda * kl_loss
            
            loss.backward()
            best = False
            if loss < max_loss:
                max_loss = loss
                z_tp1_best = torch.clone(z_tp1)
                best = True
            lr_scheduler.step(loss)
            if optimizer.param_groups[0]['lr'] < 9e-06:
                break
            optimizer.step()
            print(f't: {t}\t\t iter: {i}\t total_loss: {format(loss.item(), ".3f")}\t\t direct_loss: {format(direct_loss.item(), ".3f")}\t\t kl_loss: {format(kl_loss.item(), ".3f")}\t\t best: {best}')

        if z_tp1_best is not None:
            z_tp1 = z_tp1_best
        
        self.prev_z4 = torch.clone(z_tp1)

        return z_tp1.detach()

    def opt_inv(self,
                z_t,
                t,
                prompt_embeds,
                added_cond_kwargs,
                prev_timestep,
                nom_opt_iters=1,
                lr=0.001,
                opt_none_inference_steps=False,
                opt_loss_kl_lambda=10.0,
                num_aprox_steps=100):
        
        z_tp1 = self.inversion_step(z_t, t, prompt_embeds, added_cond_kwargs, num_aprox_steps=num_aprox_steps)

        if t in self.scheduler_inference.timesteps:
            z_tp1 = self.optimize_z_tp1(z_tp1, z_t, t, prompt_embeds, added_cond_kwargs, nom_opt_iters=nom_opt_iters, lr=lr, opt_loss_kl_lambda=opt_loss_kl_lambda)

        return z_tp1

    def latent2image(self, latents):
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        # cast back to fp16 if needed
        # if needs_upcasting:
        #     self.vae.to(dtype=torch.float16)
        
        return image
    
    def patchify_latents_kl_divergence(self, x0, x1):
        # devide x0 and x1 into patches (4x64x64) -> (4x4x4)
        PATCH_SIZE = 4
        NUM_CHANNELS = 4

        def patchify_tensor(input_tensor):
            patches = input_tensor.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
            patches = patches.contiguous().view(-1, NUM_CHANNELS, PATCH_SIZE, PATCH_SIZE)
            return patches
        
        x0 = patchify_tensor(x0)
        x1 = patchify_tensor(x1)

        kl = self.latents_kl_divergence(x0, x1).sum()
        # for i in range(x0.shape[0]):
        #     kl += self.latents_kl_divergence(x0[i], x1[i])
        return kl

    
    def latents_kl_divergence(self, x0, x1):
        EPSILON = 1e-6

        #{\displaystyle D_{\text{KL}}\left({\mathcal {N}}_{0}\parallel {\mathcal {N}}_{1}\right)={\frac {1}{2}}\left(\operatorname {tr} \left(\Sigma _{1}^{-1}\Sigma _{0}\right)-k+\left(\mu _{1}-\mu _{0}\right)^{\mathsf {T}}\Sigma _{1}^{-1}\left(\mu _{1}-\mu _{0}\right)+\ln \left({\frac {\det \Sigma _{1}}{\det \Sigma _{0}}}\right)\right).}
        x0 = x0.view(x0.shape[0], x0.shape[1], -1)
        x1 = x1.view(x1.shape[0], x1.shape[1], -1)
        mu0 = x0.mean(dim=-1)
        mu1 = x1.mean(dim=-1)
        var0 = x0.var(dim=-1)
        var1 = x1.var(dim=-1)
        kl = torch.log((var1 + EPSILON) / (var0 + EPSILON)) + (var0 + (mu0 - mu1)**2) / (var1 + EPSILON) - 1
        kl = torch.abs(kl).sum(dim=-1)
        # kl = torch.linalg.norm(mu0 - mu1) + torch.linalg.norm(var0 - var1)
        # kl *= 1000
        # sigma0 = torch.cov(x0)
        # sigma1 = torch.cov(x1)
        # inv_sigma1 = torch.inverse(sigma1.to(dtype=torch.float64)).to(dtype=x0.dtype)
        # k = x0.shape[1]
        # kl = 0.5 * (torch.trace(inv_sigma1 @ sigma0) - k + (mu1 - mu0).T @ inv_sigma1 @ (mu1 - mu0) + torch.log(torch.det(sigma1) / torch.det(sigma0)))
        return kl

    
class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)

        # dummy loss value
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        gt_grad, = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None

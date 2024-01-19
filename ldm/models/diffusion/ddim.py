"""SAMPLING ONLY."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'

        def to_torch(x): return x.clone().detach().to(
            torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(
            self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(
            np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(
            np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas',
                             np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps',
                             sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(
                        f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            # img shape (1, 4, 64, 64)
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(
                timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}

        # 1, 21, 41....981
        time_range = reversed(
            range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        # print(time_range)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # print(index, step)

            if mask is not None:
                assert x0 is not None
                # TODO: deterministic forward pass?
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            b, *_, device = *img.shape, img.device

            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(img, ts, cond)
            else:
                e_t = self.get_noise_pred_single(img, cond, ts, unconditional_conditioning,
                                                 unconditional_guidance_scale)

            self.score_correct(e_t, img, ts, cond,
                               score_corrector, corrector_kwargs)

            outs = self.prev_step(img, e_t, index, use_original_steps=ddim_use_original_steps,
                                  quantize_denoised=quantize_denoised,
                                  temperature=temperature, noise_dropout=noise_dropout,
                                  return_x0=True)
            img, pred_x0 = outs
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    def score_correct(self, e_t, x, t, context, score_corrector=None, corrector_kwargs=None):
        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(
                self.model, e_t, x, t, context, **corrector_kwargs)
        return e_t

    @torch.no_grad()
    def ddim_loop(self, latent, cond_embeddings, t_enc):
        all_latent = [latent]
        latent = latent.clone().detach()
        timesteps = self.ddim_timesteps[:t_enc]
        total_steps = timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")
        iterator = tqdm(timesteps, desc='Encoding image', total=total_steps)
        for i, step in enumerate(iterator):
            ts = torch.full((latent.shape[0],), step,
                            device=latent.device, dtype=torch.long)
            # print(ts, step)  # 1, 21, 41....981
            noise_pred, _ = self.model.apply_model(latent, ts, cond_embeddings)
            latent = self.next_step(
                noise_pred, step, latent, use_original_steps=True)
            all_latent.append(latent)
        return all_latent

    def next_step(self, e_t, timestep, x, use_original_steps=False, quantize_denoised=False):
        next_timestep = min(timestep + 1000 // len(self.ddim_timesteps), 999)
        b, *_, device = *x.shape, x.device
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        # select parameters corresponding to the currently considered timestep
        a_next = torch.full((b, 1, 1, 1), alphas[next_timestep], device=device)
        a_t = torch.full((b, 1, 1, 1), alphas[timestep], device=device)
        sqrt_one_minus_a_t = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[timestep], device=device)
        next_x0 = (x - sqrt_one_minus_a_t * e_t) / a_t.sqrt()
        # current prediction for x_0
        if quantize_denoised:
            next_x0, _, *_ = self.model.first_stage_model.quantize(next_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_next).sqrt() * e_t
        x_next = a_next.sqrt() * next_x0 + dir_xt
        return x_next

    def get_noise_pred_single(self, x, c, t, unconditional_conditioning=None, unconditional_guidance_scale=1.):
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        # unconditional_conditioning, c are all of shape(1, 77, 768)
        c_in = torch.cat([unconditional_conditioning, c]
                         )  # c_in.shape=(2, 77, 768)
        e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in)[0].chunk(2)
        # split it into two at 0 dim, e_t of (1,4,64,64)
        grad = (e_t - e_t_uncond)  # 1,4,64,64
        # grad = bilateralFilter(grad, 3)
        e_t = e_t_uncond + unconditional_guidance_scale * grad

        return e_t

    def prev_step(self, x, e_t, index, repeat_noise=False, use_original_steps=False,
                  quantize_denoised=False, temperature=1., noise_dropout=0., return_x0=False):
        b, *_, device = *x.shape, x.device

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * \
            noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        if return_x0:
            return x_prev, pred_x0
        else:
            return x_prev

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas

        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def ddim_inverse(self, latent, context, t_start, unconditional_guidance_scale=1.0,
                     unconditional_conditioning=None,
                     use_original_steps=False):

        iterator, total_steps = self.prepare_iterator(
            t_start, use_original_steps)

        x = latent
        for i, step in enumerate(iterator):
            # print(step) 981....1
            index = total_steps - i - 1
            ts = torch.full((x.shape[0],), step,
                            device=x.device, dtype=torch.long)

            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t, _ = self.model.apply_model(x, ts, context)
            else:
                e_t = self.get_noise_pred_single(x, context, ts,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
            x = self.prev_step(
                x, e_t, index, use_original_steps=use_original_steps)
        return x

    # @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        iterator, total_steps = self.prepare_iterator(
            t_start, use_original_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],),
                            step, dtype=torch.long).cuda()
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t, _ = self.model.apply_model(x_dec, ts, cond)
            else:
                e_t = self.get_noise_pred_single(x_dec, cond, ts,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
            x_dec = self.prev_step(
                x_dec, e_t, index, use_original_steps=use_original_steps)

        return x_dec

    def prepare_iterator(self, t_start, use_original_steps):
        timesteps = np.arange(
            self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")
        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        return iterator, total_steps

    @torch.no_grad()
    def ddim_replace_object(self, latent, ref_latents, src_cond, tgt_cond, t_start,
                            unconditional_guidance_scale=1.0,
                            unconditional_conditioning=None, use_original_steps=False,
                            corrector_fn=None,
                            quantize_denoised=False,
                            temperature=1.,
                            noise_dropout=0.,
                            score_corrector=None,
                            corrector_kwargs=None,
                            model_kwargs=None
                            ):
        iterator, total_steps = self.prepare_iterator(
            t_start, use_original_steps)

        x = latent.clone()
        x_ref = ref_latents
        for i, step in enumerate(iterator):
            # print(step, ts) 981....1
            index = total_steps - i - 1
            # print(index) 49,...0
            ts = torch.full((x.shape[0],), step,
                            device=x.device, dtype=torch.long)

            x = self.replace_object_single_step(x, x_ref[index], ts, src_cond, tgt_cond, index,
                                                unconditional_conditioning,
                                                unconditional_guidance_scale, use_original_steps,
                                                quantize_denoised=quantize_denoised,
                                                temperature=temperature, noise_dropout=noise_dropout,
                                                score_corrector=score_corrector,
                                                corrector_kwargs=corrector_kwargs, model_kwargs=model_kwargs)
            if corrector_fn is not None:
                x = corrector_fn(x, index)

        return x

    @torch.no_grad()
    def replace_object_single_step(self, x, x_ref, t, src, dst, index, unconditional_conditioning,
                                   unconditional_guidance_scale, use_original_steps,
                                   quantize_denoised=False,
                                   temperature=1.,
                                   noise_dropout=0.,
                                   score_corrector=None,
                                   corrector_kwargs=None,
                                   model_kwargs=None,
                                   ):

        e_ref, h_ref = self.model.apply_model(x_ref.clone(), t, src)
        if t > model_kwargs['t_combine']:
            e_t_uncond = self.model.apply_model_fusion(x, t, unconditional_conditioning,
                                                       h_ref=h_ref,
                                                       model_kwargs=model_kwargs)[0]
            e_t = self.model.apply_model_fusion(x, t, dst,
                                                h_ref=h_ref,
                                                model_kwargs=model_kwargs)[0]
        else:
            e_t_uncond = self.model.apply_model(
                x, t, unconditional_conditioning)[0]
            e_t = self.model.apply_model(x, t, dst)[0]
        grad = e_t - e_t_uncond
        e_t = e_t_uncond + unconditional_guidance_scale * grad
        self.score_correct(e_t, x, t, dst, score_corrector, corrector_kwargs)
        x_pred = self.prev_step(x, e_t, index, use_original_steps=use_original_steps,
                                quantize_denoised=quantize_denoised,
                                temperature=temperature, noise_dropout=noise_dropout)

        return x_pred

    def diffedit(self, latent, cond, t_start,
                 unconditional_guidance_scale=1.0,
                 unconditional_conditioning=None, use_original_steps=False,
                 corrector_fn=None,
                 quantize_denoised=False,
                 temperature=1.,
                 noise_dropout=0.,
                 score_corrector=None,
                 corrector_kwargs=None):
        b = latent.shape[0]

        img = latent

        iterator, total_steps = self.prepare_iterator(
            t_start, use_original_steps)

        for i, step in enumerate(iterator):

            index = total_steps - i - 1
            ts = torch.full((b,), step, dtype=torch.long).cuda()
            # print(step, index)

            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(img, ts, cond)
            else:
                e_t = self.get_noise_pred_single(img, cond, ts, unconditional_conditioning,
                                                 unconditional_guidance_scale)

            self.score_correct(e_t, img, ts, cond,
                               score_corrector, corrector_kwargs)

            outs = self.prev_step(img, e_t, index, use_original_steps=use_original_steps,
                                  quantize_denoised=quantize_denoised,
                                  temperature=temperature, noise_dropout=noise_dropout)

            img = outs

            if corrector_fn is not None:
                img = corrector_fn(img, index)

        return img

from ldm.util import instantiate_from_config
import argparse
import torch
import os
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import sys
import cv2

lldm_dir = os.path.abspath('./')
sys.path.append(lldm_dir)


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys: pass")
    #         print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys: pass")
    model.eval()
    return model


def get_coco_mask_dilate(name, root_dir, dilate_args=None):
    print(os.path.join(root_dir, 'masks/%s.png' % name))
    img_mask = cv2.imread(os.path.join(
        root_dir, 'masks/%s.png' % name), cv2.IMREAD_GRAYSCALE)
    img_mask = img_mask / 255

    if dilate_args is not None:
        kernel = np.ones((dilate_args[0], dilate_args[0]), dtype=np.uint8)
        img_mask = cv2.dilate(img_mask, kernel, dilate_args[1])

    img_mask = cv2.resize(img_mask, (64, 64), interpolation=cv2.INTER_LANCZOS4)
    img_mask[img_mask > 0.5] = 1
    img_mask[img_mask <= 0.5] = 0

    img_mask = torch.Tensor(img_mask).unsqueeze(0).unsqueeze(0).cuda()
    return img_mask


def load_img(path, W, H):
    image = Image.open(path).convert("RGB")
    image = image.resize((W, H), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def latent_to_image(model, latents):
    x_samples = model.decode_first_stage(latents)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
    x_samples = 255. * x_samples
    x_samples = x_samples.astype(np.uint8)
    return x_samples


def repeat_tensor(x, n, dim=0):
    dims = len(x.shape) * [1]
    dims[dim] = n
    return x.repeat(dims)


def load_model(opt):
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    return model


def make_dataset_txt(files):
    """
    :param path_files: the path of txt file that store the image paths
    :return: image paths
    """
    img_paths = []

    with open(files) as f:
        paths = f.readlines()

    for path in paths:
        path = path.strip()
        img_paths.append(path)

    return img_paths


def data_to_dict(data_set):
    tab = {}
    for data in data_set:
        data = data.split()
        img_id, prompt = data[0], data[1:]
        prompt = ' '.join(prompt)
        tab[img_id] = prompt
    return tab


parser = argparse.ArgumentParser()

parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)

parser.add_argument(
    "--ratio",
    type=float,
    default=1,
    help="encoding ratio",
)

parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)

parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--config",
    type=str,
    default="configs/stable-diffusion/v1-inference.yaml",
    help="path to config which constructs model",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="models/ldm/stable-diffusion-v1/model.ckpt",
    help="path to checkpoint of model",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--precision",
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast"
)

parser.add_argument(
    "--out_dir",
    type=str,
    help="the path of image to be edit",
    default='outputs/',
    required=False
)


def predict_mask(sampler, model, src, dst, uc, init_latent, noised_sample_num, ddim_steps, encode_ratio=0.5,
                 clamp_rate: float = 3.5):
    """
    the map value will be clamped to map.mean() * clamp_rate, then values will be scaled into 0~1, then term into binary(split at 0.5).
    so if a map value is large than map.mean() * clamp_rate * 0.5 will be encode to 1, less will be encode to 0.
    so the larger clamp rate is, less pixes will be encode to 1, the small clamp rate is, the more pixes will be encode to 1.
    """
    device = model.device
    repeated = repeat_tensor(init_latent, noised_sample_num)
    src = repeat_tensor(src, noised_sample_num)
    dst = repeat_tensor(dst, noised_sample_num)

    # add noise

    t_enc = int(encode_ratio * ddim_steps)

    time_step = sampler.ddim_timesteps[t_enc]

    t = torch.tensor([time_step] * 1).to(device)

    noised = sampler.stochastic_encode(
        repeated, torch.tensor([t_enc] * 1).to(device))

    noised = noised.squeeze(0)

    # pre on same noised
    pre_src = model.apply_model(noised, t, src)[0]
    pre_dst = model.apply_model(noised, t, dst)[0]

    # consider to add smooth method
    subed = (pre_src - pre_dst).abs_().mean(dim=[0, 1])
    max_v = subed.mean() * clamp_rate
    mask = subed.clamp(0, max_v) / max_v

    def to_binary(pix):
        if pix > 0.5:
            return 1.
        else:
            return 0.

    mask = mask.cpu().apply_(to_binary).to(device)
    return mask


def replace_object(sampler, model, tgt_img,
                   src_prompt,
                   dst_prompt,
                   encode_ratio,
                   ddim_steps,
                   model_kwargs,
                   scale: float = 7.5):
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning([""])
    src_cond = model.get_learned_conditioning([src_prompt])
    tgt_cond = model.get_learned_conditioning([dst_prompt])
    t_enc = int(ddim_steps * encode_ratio)

    ref_latent = model.get_first_stage_encoding(
        model.encode_first_stage(tgt_img))

    ref_latents = sampler.ddim_loop(ref_latent, src_cond, t_enc)

    init_latent = ref_latents[-1].clone()

    tgt_mask = model_kwargs['tgt_mask']

    def corrector_fn(x, index):
        if index / ddim_steps > model_kwargs['self_replace_step']:
            x = x * tgt_mask + (1 - tgt_mask) * ref_latents[index]
        return x

    recover_latent = sampler.ddim_replace_object(init_latent, ref_latents, src_cond, tgt_cond, t_enc,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 corrector_fn=corrector_fn,
                                                 model_kwargs=model_kwargs)

    images = latent_to_image(model, recover_latent)
    return images

from ldm.models.diffusion.ddim import DDIMSampler
import os
import sys
from contextlib import nullcontext
import pickle
import torch
from PIL import Image
from einops import repeat
from pytorch_lightning import seed_everything
from torch import autocast
import numpy as np
from common import parser, load_model, replace_object, \
    load_img, make_dataset_txt, get_coco_mask_dilate

lldm_dir = os.path.abspath('./')
sys.path.append(lldm_dir)
opt = parser.parse_args()
# prepare
model = load_model(opt)
device = torch.device("cuda")
model = model.to(device)
sampler = DDIMSampler(model)
seed_everything(opt.seed)
model.cond_stage_model = model.cond_stage_model.to(device)
precision_scope = autocast if opt.precision == "autocast" else nullcontext
sampler.make_schedule(ddim_num_steps=opt.ddim_steps,
                      ddim_eta=0.0, verbose=False)
# load data
img_list = make_dataset_txt('data/coco/object_list.txt')
with open('data/coco/data.pkl', 'rb') as f:
    data = pickle.load(f)
root_dir = 'data/coco_animals'
# define hyper-parameters
fusion_layers = {
    "encoder": [8, 9, 10, 11, 12],
    "middle": True,
    "decoder": [13, 14, 15, 16, 17]
}
attn_layers = {
    "encoder": [6, 7, 8, 9, 10, 11, 12],
    "decoder": [13, 14, 15, 16, 17, 18, 19],
}
t_combine = 0
self_replace_step = 0.5
masked_attn = True
# ========================================================================
out_dir = os.path.join(opt.out_dir, 'coco-obj')
os.makedirs(out_dir, exist_ok=True)

with torch.no_grad():
    with precision_scope(device.type):
        with model.ema_scope():
            for name in img_list[:2500]:
                name = str(name)
                img_path = os.path.join(root_dir, 'img_dir/%s.png' % name)
                tgt_image = repeat(
                    load_img(img_path, opt.W, opt.H).cuda(), '1 ... -> b ...', b=1)
                prompt = data[int(name)]['src']
                edit_prompt = 'a ' + data[int(name)]['new_obj']
                print(name, edit_prompt)

                tgt_mask = get_coco_mask_dilate(name, root_dir)
                model_kwargs = {}
                model_kwargs['fusion_layers'] = fusion_layers
                model_kwargs["t_combine"] = t_combine
                model_kwargs["tgt_mask"] = tgt_mask
                model_kwargs["attn_mask"] = {
                    'attn_mask': tgt_mask,
                    'words': [[1, 2], []]
                } if masked_attn else None

                model_kwargs["attn_layers"] = attn_layers
                model_kwargs["self_replace_step"] = self_replace_step

                res = replace_object(sampler, model, tgt_image,
                                     src_prompt=prompt,
                                     dst_prompt=edit_prompt,
                                     encode_ratio=opt.ratio,
                                     ddim_steps=opt.ddim_steps,
                                     scale=opt.scale,
                                     model_kwargs=model_kwargs,
                                     )
                Image.fromarray(res[0]).save(out_dir + '/%s.png' % name)

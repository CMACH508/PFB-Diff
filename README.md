## PFB-Diff — Official PyTorch Implementation

---

This repository contains the **official PyTorch implementation** of the paper:

**PFB-Diff: Progressive Feature Blending Diffusion for Text-driven Image Editing**

> **Abstract:** *Diffusion models have showcased their remarkable capability to synthesize diverse and high-quality images, sparking interest in their application for real image editing. However, existing diffusion-based approaches for local image editing often suffer from undesired artifacts due to the latent-level blending of the noised target images and diffusion latent variables, which lack the necessary semantics for maintaining image consistency. To address these issues, we propose PFB-Diff, a Progressive Feature Blending method for Diffusion-based image editing. Unlike previous methods, PFB-Diff seamlessly integrates text-guided generated content into the target image through multi-level feature blending. The rich semantics encoded in deep features and the progressive blending scheme from high to low levels ensure semantic coherence and high quality in edited images. Additionally, we introduce an attention masking mechanism in the cross-attention layers to confine the impact of specific words to desired regions, further improving the performance of background editing. PFB-Diff can effectively address various editing tasks, including object/background replacement and object attribute editing. Our method demonstrates its superior performance in terms of editing accuracy and image quality without the need for fine-tuning or training. *

## Installation

---

Install the dependencies:
```bash
conda create -n pfb-diff python=3.8
conda activate pfb-diff
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```
Before running [**pfbdiff.ipynb**](https://github.com/CMACH508/PFB-Diff/blob/main/pdfbdiff.ipynb), run the following command to add the virtual environment pfb-diff to the jupyter kernel:

```bash
python -m ipykernel install --user --name=pfb-diff
```

## Quick start

---

For a quick start, we recommend taking a look at the notebook: [**pfbdiff.ipynb**](https://github.com/CMACH508/PFB-Diff/blob/main/pdfbdiff.ipynb). The notebook contains end-to-end examples of the usage of PFB-Diff in image editing.

## Datasets and quantitative experiments

---

To obtain the COCO-animals-10k dataset, download images and masks from  [COCOA-10k](https://drive.google.com/file/d/17wOT2Du1oKMU8DtRWupR_m1mgWvnGl1I/view?usp=sharing). Unzip the file and put it into "data/coco-animals".

Download pre-trained weights,  [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)  and [xxmix9realistic ](https://civitai.com/models/47274/xxmix9realistic),  and put them into "models/ldm/stable-diffusion-v1/".

For **object editing**, run:

```
# collect latent codes
python edit_coco_object.py --out_dir <OUT_DIR> --ckpt <CHECKPOINT>
```

For **background editing**, run:

```
python edit_coco_background.py --out_dir <OUT_DIR> --ckpt <CHECKPOINT>
```

## Acknowledgement

---
This repository used some codes in  [dpm-solver ](https://github.com/LuChengTHU/dpm-solver).

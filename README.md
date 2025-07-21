## PFB-Diff — Official PyTorch Implementation

---

This repository contains the **official PyTorch implementation** of the paper:

**PFB-Diff: Progressive Feature Blending Diffusion for Text-driven Image Editing**

> **Abstract:** Diffusion models have demonstrated their ability to generate diverse and high-quality images, sparking considerable interest in their potential for real image editing applications. However, existing diffusion-based approaches for local image editing often suffer from undesired artifacts due to the latent-level blending of the noised target images and diffusion latent variables, which lack the necessary semantics for maintaining image consistency. To address these issues, we propose PFB-Diff, a Progressive Feature Blending method for Diffusion-based image editing. Unlike previous methods, PFB-Diff seamlessly integrates text-guided generated content into the target image through multi-level feature blending. The rich semantics encoded in deep features and the progressive blending scheme from high to low levels ensure semantic coherence and high quality in edited images. Additionally, we introduce an attention masking mechanism in the cross-attention layers to confine the impact of specific words to desired regions, further improving the performance of background editing and multi-object replacement. PFB-Diff can effectively address various editing tasks, including object/background replacement and object attribute editing. Our method demonstrates its superior performance in terms of editing accuracy and image quality without the need for fine-tuning or training.
---
[[Paper (Neural Networks)](https://www.sciencedirect.com/science/article/pii/S0893608024007019?via%3Dihub)]
[[Paper (arXiv)](https://arxiv.org/abs/2306.16894)]
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

**Prepare datasets**:

To obtain the COCO-animals-10k dataset, download images and masks from the following link: 

Link: https://pan.baidu.com/s/1BIlcqy1f6exLnPFIBhyamQ   Password: 8888 

Unzip the downloaded file and put it into "data/coco-animals".

**Prepare pre-trained models**:

Download pre-trained weights, [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/tree/main) and [xxmix9realistic ](https://civitai.com/models/47274/xxmix9realistic),  and put them into "models/ldm/stable-diffusion-v1/".

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

## Citation
If you find PFBDiff useful for your project or research, welcome to 🌟 this repo and cite our work using the following BibTeX:
```bibtex
@article{huang2025pfb,
  title={Pfb-diff: Progressive feature blending diffusion for text-driven image editing},
  author={Huang, Wenjing and Tu, Shikui and Xu, Lei},
  journal={Neural Networks},
  volume={181},
  pages={106777},
  year={2025},
  publisher={Elsevier}
}
```

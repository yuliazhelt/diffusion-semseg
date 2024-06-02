# Application of Generative Models in Discriminative Tasks

This work explores the adaptability of pre-trained text-to-image diffusion models for visual perception tasks, with a focus on semantic segmentation. The core hypothesis is that the
knowledge embedded within these generative models can be effectively transferred to enhance the
quality and efficiency for semantic segmentation tasks. The investigation centers on how these
models, trained to synthesize images based on textual prompts, might be repurposed to interpret
and segment real images.

The research focuses on [VPD](https://github.com/wl-zhao/VPD) approach with its extensions: [TADP](https://github.com/damaggu/TADP) and [MetaPrompts](https://github.com/fudan-zvg/meta-prompts). Additionally, employing custom imrovements as well as combining features from varioud approaches.

## Environment Setup
```
bash setup_environment.sh
```

## Dataset and Checkpoints Loading
Pre-trained StableDiffusion model: 
```
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -O checkpoints/v1-5-pruned-emaonly.ckpt 
```

ADE20K dataset:
```
mkdir data/ &&
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip -O data/ADEChallengeData2016.zip &&
unzip data/ADEChallengeData2016.zip -d data/
```

## Experiments
Experiment results and logs are available at [WandB project page](https://wandb.ai/yuliazhelt/thesis_vpd).


## Acknowledgements
This code is based on [VPD](https://github.com/wl-zhao/VPD), [stable-diffusion](https://github.com/CompVis/stable-diffusion), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [TADP](https://github.com/damaggu/TADP), [MetaPrompts](https://github.com/fudan-zvg/meta-prompts).
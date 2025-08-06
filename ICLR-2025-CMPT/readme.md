# RELEASE THE POWERS OF PROMPT TUNING: CROSS-MODALITY PROMPT TRANSFER

The implementation of "RELEASE THE POWERS OF PROMPT TUNING: CROSS-MODALITY PROMPT TRANSFER" in Python. 

This work is funded by ARC (Laureate Fellow grant FL190100149) and Australian Artificial Intelligence Institute (AAII), University of Technology Sydney (UTS).

Code for the ICLR 2025 publication. The full paper can be found [here](https://openreview.net/forum?id=SYnIf4LxAG).

## Contribution

+ We explore cross-modality prompt transfer as an effective approach for boosting prompt tunning and verify the feasibility through extensive experiments and in-depth analysis, addressing a critical research gap and opening new possibilities for data-scarce modalities.
+ We introduce a novel method for estimating prompt transferability by quantifying the modality and task gaps, which the existing in-modality methods have overlooked, as the gaps are enlarged in the cross-modality scenario. As a result, our method offers a more accurate solution compared to existing in-modality methods.
+ We further demonstrate the powers of cross-modality prompt transfer through Attention Transfer, which eases the modality gap and task gap by injecting target knowledge into the prompt and utilizing a top-transferable source prompt more effectively. As a result, attention transfer enables prompt tuning to perform comparably or even better than the best prompt-based benchmark.

## Overview
![Framework](github_images/Frozen%20and%20Projection.svg)



## Setup
```
conda create -n xmm python==3.9 pip
conda activate xmm
pip install torch==2.0.1 torchvision==0.15.2
pip install transformers==4.31.0
pip install seaborn
pip install tensorboard
pip install numpy==1.25.2
```
Open the folder as a PyCharm project is recommended after setting up.

## Dataset
Please follow https://github.com/ziplab/SPT.
Note:
+ Save the images in the .png format.
+ Put the entire dataset into the `./data` folder (e.g., `./data/vtab-1k/cifar`).

## Frozen Prompt Transfer & Projection Transfer & Attention Transfer
1. Open `main.py`
2. Modify the parameters under the `Parameters Need Your Modification` block, note that the function `train_utils.print_sys_info` determines your GPU device.
3. Run `main.py`
4. Note that you can modify the training hyperparameters in the config yaml files under the `./configs` folder.

## How to calculate MMD between source prompts and target images?
Simply run `./analysis/analyzer_cross_modality.py`

## Transferability
+ BASELINE-AVG_COS: modify `./transferability/baseline_main.py` line 22, variable `metric` to `COSINE_AVERAGE` and run.
+ BASELINE-MODEL_STIMULATION: modify `./transferability/baseline_main.py` line 22, variable `metric` to `MODEL_ACTIVATION` and run.
+ Gm Only: simply run `./transferability/rank_mmd_and_projection.py`
+ Gt Only: modify `./transferability/configs/UniversalProjector.yaml` line 32, variable `involve` to `False` and run `./transferability/universal_projector_main.py`
+ Gm and Gt: modify `./transferability/configs/UniversalProjector.yaml` line 32, variable `involve` to `True` and run `./transferability/universal_projector_main.py`
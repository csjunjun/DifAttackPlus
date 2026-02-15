# DifAttack++
## The official code for the paper 'DifAttack++: Query-Efficient Black-Box Adversarial Attack via Hierarchical Disentangled Feature Space in Cross–Domain'.


## Train autoencoders for image reconstruction and feature disentanglement:
set mode="train" in main.py
```
Python main.py
```

## Perform score-based black-box attack
set mode="test" in main.py
```
Python main.py
```
set testSensitivy=True to obtain the sensitivity of disentangled features in Fig.3 of the paper

## Acknowledgements
Part of the code is partially derived from ImageReconstruction [Github](https://github.com/SikanderBinMukaram/ImageReconstructionAutoEncoder/blob/main/ImageReconstruction.ipynb) and torchattacks [Github](https://github.com/Harry24k/adversarial-attacks-pytorch/tree/master).


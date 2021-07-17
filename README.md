# rsfMRI-VAE
This repository is the official Pytorch implementation of '[Representation Learning of Resting State fMRI with Variational Autoencoder](https://www.biorxiv.org/content/10.1101/2020.06.16.155937v2)'

## Environments

This code is developed and tested witn 

```
Python 2.7.17
Pytorch 1.2.0
```

## Training

To train the model in this paper, run this command:

```train
python fMRIVAE_Train.py --data-path path-to-your-data
```

## Evaluation
 
If you want to get latent variables of the trained model, run:

```eval1
python Example_Encoder.py
```

If you want to reconstruct images from the latent variables, run:

```eval2
python Example_Decoder.py
```

under construction. 

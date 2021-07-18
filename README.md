# rsfMRI-VAE
This repository is the official Pytorch implementation of '[Representation Learning of Resting State fMRI with Variational Autoencoder](https://www.biorxiv.org/content/10.1101/2020.06.16.155937v2)'

## Environments

This code is developed and tested with 

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

If you want to get latent variables of the trained model, change path inside the code `Example_Encoder.py` and run:

```eval1
python Example_Encoder.py
```

If you want to reconstruct images from the latent variables, change path inside the code `Example_Decoder.py` and run:

```eval2
python Example_Decoder.py
```


## License

Copyright 2021 Jung-Hoon Kim and Zhongming Liu
junghoon.kimok@gmail.com or zmliu@umich.edu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Reference

Kim, Jung-Hoon, et al. "Representation Learning of Resting State fMRI with Variational Autoencoder." bioRxiv (2020).

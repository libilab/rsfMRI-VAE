This document is prepared to explain how one can utilize sourcecodes (MATLAB and Python) to train and test VAE model from the scratch.
Following is the running order of codes that can reproduce findings from our paper [1]. 
In case one want to use pretrained model, ignore the procedures 1-5 and go to step 6 directly. The location of pretrained VAE model is /Trained_VAE/.
Below is the full list of platforms and their versions used here: MATLAB R2019b; Python 2.7.17; Pytorch v1.2.0.
The version of Fieldtrip MATLAB toolbox is unclear as Fieldtrip is not released with version numbers; please see version control in https://www.fieldtriptoolbox.org/reference/ft_version/.

1. Download dataset from HCP server 
   It will automatically download fMRI data (LR+RL, session1 + session2) from HCP server through Amazon Web service (AWS).
   Details for proper setting can be found at https://wiki.humanconnectome.org/display/PublicData/How+To+Connect+to+Connectome+Data+via+AWS
   Location of MATLAB code: /MATLAB/HCP_Downloader/Training_Data_Set_Downloader.m : Download training data
             /MATLAB/HCP_Downloader/Test_Data_Set_Downloader.m : Download testing data

   The subject list of testing and trainining+validation datasets used in [1] are saved as *.mat files in /MATLAB/ folder.
   For example, Sub_List_Sub500_Test.mat listed 500 subjects used in [1] and Direction_Sub500_Test.mat describes which fMRI data are used between L->R or R->L phase encoding directions

2. Minimally Preprocess dataset
   This code minimally preprocess downloaded HCP dataset (detrend + 0.01-0.1 bandpass filtering + Rescaling) 
   and modify the structure of data to fit as input of VAE model.
   It requires Fieldtrip MATLAB toolbox, which is freely available at https://www.fieldtriptoolbox.org.
   Location: /MATLAB/Preprocessing/VAE_Data_Preparation_Training_Set.m : training data
             /MATLAB/Preprocessing/VAE_Data_Preparation_Test_Set.m : testing data

3. Geometric reformatting
   This code prepares transformation matrix between grayordinate fMRI into 2D images.
   It requires the Fieldtrip MATLAB toolbox and gifti toolbox (available at http://www.artefact.tk/software/matlab/gifti/) as well, in addition to MNI brain templates and parcellation provided by HCP (already included here). 
   Outputs of the code are transformation/inverse-transformation matrices orignally used in our paper [1]. 
   Location: /MATLAB/Geometric_Reformatting/Dimension_Reduction_surface.m : Make a grid from the sphere model of MNI brain template.
             /MATLAB/Geometric_Reformatting/Geometric_Reformatting_Left_fMRI2Grid_NN.m : Generate transformation matrix of left hemisphere
             /MATLAB/Geometric_Reformatting/Geometric_Reformatting_Right_fMRI2Grid_NN.m : Generate transformation matrix of Right hemisphere

4. Preparation of dataset for VAE model
   This code applies geometric reformatting to fMRI data and save in H5 format (as input of VAE model)
   It requires Python and Pytorch. 
   Outputs of the code is Concatenated training+validation dataset (N=100+50) and testing dataset (individually saved, N=500) for VAE model. 
   Location: /Python/data_prep_Training_Validation.py : Reformat, concatenate, and save training+validation dataset (N=100+50).
             /Python/data_prep_Testing_Sess1.py : Reformat and save Session1 of testing dataset (N=500).
             /Python/data_prep_Testing_Sess2.py : Reformat and save Session2 of testing dataset (N=500).

5. Train VAE model
   This code train VAE model from the scratch, given the training set from step 4.
   Default setting is identical to the original paper [1].
   It requires Python and Pytorch. 
   Outputs of the code is learned parameters of VAE model. 
   Location: /Python/Finalv_train.py : Train VAE model.
   Finalv_model.py : The model architecture of beta-VAE.

6. Testing tool for trained VAE model
   This code utilize trained VAE model to encode/decode testing dataset.
   The pathway of input, output, VAE model should be specified accordingly.
   It requires Python and Pytorch. 
   Outputs of the code is latent variables or reconstructed fMRI data. 
   Location: /Python/Finalv_get_data.py : Test VAE model.
   Location: /Python/Example_Encoder.py : Exemplified code for encoding fMRI of testing dataset (fMRI --> Z).
   Location: /Python/Example_Decoder.py : Exemplified code for decoding Z of testing dataset (Z --> fMRI).


Copyright 2021 Jung-Hoon Kim and Zhongming Liu
junghoon.kimok@gmail.com or zmliu@umich.edu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


%%% Reference %%%

[1] Kim, Jung-Hoon, et al. "Representation Learning of Resting State fMRI with Variational Autoencoder." bioRxiv (2020).
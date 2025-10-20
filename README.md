# TestEnsemble
 
This repo implements combination methods, using only the outputs of base models for ensemble learning.

## Implementation of StackingNet, proposed in our paper "StackingNet: collective inference across independent AI foundation models".

The core advantages of StackingNet:
1. computationally efficient (only requires minimum computation to train)
2. privacy preservation (does not need any knowledge on base classifiers, except conditional independence)
3. applicable to both regression and classification
4. supports meta-combination, bias reduction, reliability ranking, and adversary pruning

```sh 
python regression_stackingnet_xxx.py
```  

```sh 
python classification_stackingnet.py
```  

### Update for StackingNet: Oct.9 code upload. Oct. 20 data upload.

## Implementation of SML-OVR, proposed in our paper "Black-Box Test-Time Ensemble".

SML-OVR utilizes only the base classifiers’ predictions on unlabeled test data, estimating the reliability of individual base classifiers and constructing a weighted ensemble that favors more accurate ones.

The core advantages of SML-OVR:
1. hyperparameter-free (zero hyperparameters to tune)
2. computationally efficient (only requires milliseconds to compute)
3. online adaptation (can be deployed for real-time online applications)
4. privacy preservation (does not need any knowledge on base classifiers, except conditional independence)

Currently, the implementation for text classification with large language models (LLMs) are provided. We provided the already generated results from the LLMs.

Run the following line to verify results on over 10 combination methods:

```sh 
python ensemble.py
```  

The classification results could also be generated from the LLMs on your own, using HuggingFace's transformers library. 

```sh 
python generate_classification.py
```  

### Update for SML-OVR: Accepted at CIM. Oct.20 paper upload.

## Contact and Citation information omitted for now

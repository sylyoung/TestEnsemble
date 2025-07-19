# TestEnsemble
 
Implementation of SML-OVR, proposed in our paper "Black-Box Test-Time Ensemble".

SML-OVR utilizes only the base classifiers’ predictions on unlabeled test data, estimating the reliability of individual base classifiers and constructing a weighted ensemble that favors more accurate ones.

The core advantages of SML-OVR:
1. hyperparameter-free (zero hyperparameters to tune)
2. computationally efficient (only requires milliseconds to compute)
3. online adaptation (can be deployed for real-time online applications)
4. privacy preservation (does not need any knowledge on base classifiers, except conditional independence)

Currently, the implementation for text classification with large language models (LLMs) are provided. We provided the already generated results from the LLMs.

```sh 
python ensemble.py
```  

The classification results could also be generated from the LLMs on your own, using HuggingFace's transformers library. 

```sh 
python generate_classification.py
```  

## Contact

Please contact me at syoungli@hust.edu.cn or lsyyoungll@gmail.com for any questions regarding the paper, and use Issues for any questions regarding the code.

## Citation

If you find this repo helpful, please cite our work:
```
@Article{Li2024,
  author  = {Li, Siyang and Wang, Ziwei and Chenhao, Liu and Wu, Dongrui},
  title   = {Black-Box Test-Time Ensemble},
  year    = {2025},
  month   = {under review,}
}
```

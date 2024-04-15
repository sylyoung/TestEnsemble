# TestEnsemble
 
Implementation of SML-OVR, proposed in our paper "Black-Box Test-Time Ensemble".

Currenly, the implementation for text classification with large language models (LLMs) are provided. We provided the already generated results from the LLMs.

```sh 
python ensemble.py
```  

The classification results could also be generated from the LLMs on your own, using HuggingFace's Transformer. 

```sh 
python generate_classification.py
```  

## Contact

Please contact me at syoungli@hust.edu.cn or lsyyoungll@gmail.com for any questions regarding the paper, and use Issues for any questions regarding the code.

## Citation

If you find this repo helpful, please cite our work:
```
@Article{Li2024,
  author  = {Li, Siyang and Wang, Ziwei and Wu, Dongrui},
  title   = {Black-Box Test-Time Ensemble},
  year    = {2024},
  month   = {under review,}
}
```

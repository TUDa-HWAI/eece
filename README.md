# Early-Exit with Class Exclusion for Efficient Inference of Neural Networks

## Train Model
```angular2html
python --cf train.py ./configs/{alexnet, resnet, vgg}/train/{model_name}_{dataset_name}_BC.yaml
```
BC for our method, CR for the traditional method.

## $\beta$  Search

```angular2html
python fine_tune.py --cf ./configs/{alexnet, resnet, vgg}/search/{model_name}_{dataset_name}_BC.yaml
```

## Test

```angular2html
python exit_text.py --cf ./configs/{alexnet, resnet, vgg}/test/{model_name}_{dataset_name}_BC.yaml
```
## Reference 
@INPROCEEDINGS{Wang2023EarlyExitWC,

  title={Early-Exit with Class Exclusion for Efficient Inference of Neural Networks},
  
  author={Jingcun Wang and Bing Li and Grace Li Zhang},
  
  booktitle={2024 IEEE 6th International Conference on AI Circuits and Systems (AICAS)},
  
  year={2024}
}

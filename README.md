# AIsystem Final Project


## 📝Report


## 👥Members


|<img src='https://avatars.githubusercontent.com/u/85860941?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/87682189?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/100858094?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/145880572?v=4' height=100 width=100px></img>|
| --- | --- | --- | --- |
| [서가은](https://github.com/gaeun0112) | [박서영](https://github.com/0o1ong) | [김정연](https://github.com/Kimtona) | [안민우](https://github.com/MWAhn991001) |
| BERT 코드 작성 및 실험<br/> & github 정리| CNN 코드 작성 및 실험 <br/> & Overleaf 작성 | ALU unit 코드 작성 및 실험 | Resnet50 코드 작성 및 실험 |


## <제목>
> "딥러닝 모델 task에 따른 최적의 활성화 함수 일반화 가능 여부 연구"


## 🔬Experiment
| Task | Image Classification | Text Classification |
| --- | --- | --- |
| Model | CNN, Resnet50 + **ALU** | BERT |
| Data | MNIST, CIFAR-10 | SST-2, CoLA |

## 🖱️ Usage
### 1. CNN, Resnet50, ALU unit
* CNN : Try code `CNN/CNN_Adam.ipynb`, `CNN/CNN_SGD.ipynb`
* Resnet50 : Try code `Resnet/ResNet50(Adam,_for_MNIST+CIFAR10).ipynb`, `Resnet/ResNet50(SGD,_for_MNIST+CIFAR10).ipynb`
* ALU unit : Try code `ALU_unit/AI시스템_KJY.ipynb`

### 2. BERT
* Install Torch `2.3.0`
'''
pip install -r requirments.txt
python  run_bert.py --dataset_name cola --num_epochs 1 --wandb_api_key [YOUR WANDB ACCESS TOKEN]
'''

import wandb
import random
import numpy as np
import argparse
from tqdm import tqdm
import time

from fairscale.nn.model_parallel.layers import (ColumnParallelLinear, ParallelEmbedding, RowParallelLinear)
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from datasets import load_dataset, DatasetDict, Dataset


parser = argparse.ArgumentParser()

parser.add_argument('--wandb_api_key', type=str, help="Write your wandb api key.")
parser.add_argument('--activation_function', type=str, default="relu", help="choice :  relu, gelu,  tanh, sigmoid, elu, leakyrelu, swish, swiglu, relu_2, penalized_tanh, glu")
parser.add_argument('--dataset_name', type=str, default="cola", help="choice : cola, sst2")
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('-project_name', type=str, default="aisystem_bert")

args = parser.parse_args()



# 소문자화 전처리
def lowercase_function(examples):
    examples['sentence'] = examples['sentence'].lower()
    return examples

# test 데이터셋 처리
def replace_test_dataset(dataset_dict, test_size=1000):
    train_dataset = dataset_dict['train']
    shuffled_indices = list(range(len(train_dataset)))
    random.seed(42)
    random.shuffle(shuffled_indices)

    new_test_indices = shuffled_indices[:test_size]
    new_train_indices = shuffled_indices[test_size:]

    new_test_dataset = train_dataset.select(new_test_indices)
    new_train_dataset = train_dataset.select(new_train_indices)

    new_dataset_dict = DatasetDict({
        'train': new_train_dataset,
        'validation': dataset_dict['validation'],
        'test': new_test_dataset
    })

    return new_dataset_dict

def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class ReLU2(nn.Module):
    def forward(self, z):
        return torch.max(torch.tensor(0.0), z**2)

class PenalizedTanh(nn.Module):
    def forward(self, x):
        return torch.where(x > 0, torch.tanh(x), 0.25 * torch.tanh(x))
    

def change_activation_function(model, activation_function):
    # 활성화 함수 매핑
    activation_functions = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'elu': nn.ELU(),
        'leakyrelu': nn.LeakyReLU(),
        'swish':nn.SiLU(),
        'swiglu' : SwiGLU(),
        'relu_2' : ReLU2(),
        'penalized_tanh' : PenalizedTanh(),
        "glu":nn.GLU(),
    }

    # 올바른 활성화 함수가 전달되었는지 확인
    if activation_function.lower() not in activation_functions:
        raise ValueError(f"Unsupported activation function: {activation_function}. Supported functions are: {list(activation_functions.keys())}")

    # 해당 활성화 함수로 변경
    for layer in model.bert.encoder.layer:
        layer.intermediate.intermediate_act_fn = activation_functions[activation_function.lower()]


# 학습 함수
def train(model, train_loader, val_loader, optimizer, device, num_epochs):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    wandb.config = {
        "learning_rate": optimizer.param_groups[0]['lr'],
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size
    }

    wandb.watch(model, log="all")  # 모델의 모든 파라미터와 그래디언트를 로깅합니다.

    layers_to_watch = [0, 3, 6, 9]
    train_global_step = 0
    val_global_step = 0

    for epoch in range(num_epochs):
        model.train()
        train_epoch_start_time = time.time()

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()

            # wandb를 사용하여 로깅
            if layers_to_watch is not None:
                for layer_idx in layers_to_watch:
                    layer = model.bert.encoder.layer[layer_idx]
                    layer_gradients = []
                    for name, param in layer.named_parameters():
                        if param.grad is not None:
                            # avg_grad = param.grad.abs().mean()
                            # layer_gradients.append(avg_grad)
                            grad_norm = torch.norm(param.grad, p=2)
                            layer_gradients.append(grad_norm)

                    if layer_gradients:
                        # total_avg_grad = torch.stack(layer_gradients).mean().item()
                        # wandb.log({f'Layer_{layer_idx+1}/avg_grad': total_avg_grad, 'global_step': train_global_step})
                        total_norm = torch.stack(layer_gradients).mean().item()
                        wandb.log({f'Layer_{layer_idx+1}/grad_norm': total_norm, 'global_step': train_global_step})

            classifier_gradients = []
            for param in model.classifier.parameters():
                if param.grad is not None:
                    # avg_grad = param.grad.abs().mean()
                    # classifier_gradients.append(avg_grad)
                    grad_norm = torch.norm(param.grad, p=2)
                    classifier_gradients.append(grad_norm)
            if classifier_gradients:
                # total_avg_grad = torch.stack(classifier_gradients).mean().item()
                # wandb.log({'Classifier/avg_grad': total_avg_grad, 'global_step': train_global_step})
                total_norm = torch.stack(classifier_gradients).mean().item()
                wandb.log({'Classifier/grad_norm': total_norm, 'global_step': train_global_step})

            train_global_step += 1
            optimizer.step()

            wandb.log({'Training Loss': loss.item(), 'global_step': train_global_step})

        train_epoch_end_time = time.time()
        wandb.log({'Training Time': train_epoch_end_time - train_epoch_start_time, 'epoch': epoch})

        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)

                _, preds = torch.max(logits, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())


                wandb.log({'Validation Loss': loss.item(), 'global_step': val_global_step})
                val_global_step += 1

            val_accuracy = accuracy_score(val_labels, val_preds)
            val_f1_score = f1_score(val_labels, val_preds, average='weighted')
            wandb.log({'Validation Accuracy': val_accuracy, 'Validation F1 Score': val_f1_score, 'epoch': epoch})

# 평가 함수
def evaluate(model, dataloader, device, dataset_name="test"):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    test_labels = []
    test_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(predictions.cpu().numpy())

    test_accuracy = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')

    print(f"{dataset_name} Accuracy: {test_accuracy}")
    print(f"{dataset_name} F1 Score: {test_f1}")

    wandb.log({
        'test_accuracy': test_accuracy,
        'test_f1': test_f1
    })


if __name__ == "__main__": 
    # wandb 로그인
    wandb.login(key = args.wandb_api_key)

    # random seed 고정
    seed = 42
    deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

    # 데이터셋 로드
    dataset = load_dataset('glue', args.dataset_name)

    lower_dataset = dataset.map(lowercase_function)

    new_dataset = replace_test_dataset(lower_dataset)

    # Tokenizing
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    encoded_dataset = new_dataset.map(tokenize_function, batched=True)
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


    # 데이터로더 생성
    train_dataloader = DataLoader(encoded_dataset['train'], batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(encoded_dataset['validation'], batch_size=args.batch_size)
    test_dataloader = DataLoader(encoded_dataset['test'], batch_size=args.batch_size)



    # 모델 선언
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.train()


    if args.activation_function not in ["gelu", "glu", "swiglu"]:
        change_activation_function(model, args.activation_function)
    elif args.activation_function in ["glu", "swiglu"]:
        for layer in model.bert.encoder.layer:
            layer.output.dense = nn.Linear(1536, 768)
            change_activation_function(model, args.activation_function)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)


    # wandb 초기화
    wandb.init(project=args.project_name)

    # 훈련 및 평가 실행
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(model, train_dataloader, eval_dataloader, optimizer, device, args.num_epochs)

    evaluate(model, test_dataloader, device, dataset_name="test")

    wandb.finish()  # wandb 세션을 종료.
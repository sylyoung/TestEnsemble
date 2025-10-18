# -*- coding: utf-8 -*-
# @Time    : 2024/3/1
# @Author  : ???
# @File    : generate_classification.py
# generate classification results using LLMs on text datasets
import sys, os, argparse, random
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import AutoConfig, LlamaConfig
from transformers import LlamaForCausalLM


def seed_everything(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)


def write_to_file(data, path):
    data = data.reshape(-1,)
    f = open(path, 'w')
    np.savetxt(f, data.astype(int))
    f.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM classification experiment')
    parser.add_argument('--seed', type=int, default=0, help='seed for everything')
    parser.add_argument('--dataset', type=str, default='SST', help='dataset name')
    parser.add_argument('--gpu_id', type=int, default=-1, help='gpuid')
    parser.add_argument('--model_id', type=int, default=0, help='model id')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for inference')

    args = parser.parse_args()
    seed_everything(args)

    print(args)

    try:
        device_id = str(args.gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        args.data_env = 'gpu'
    except:
        args.data_env = 'local'

    print('Using',  args.data_env)

    print('dataset', args.dataset)

    if args.model_id == 0:
        model_name = "Mistral-7B-v0.1"
    elif args.model_id == 1:
        model_name = "TinyLlama-1.1B-Chat-v1.0"
    elif args.model_id == 2:
        model_name = "jais-13b-chat"
    elif args.model_id == 3:
        model_name = "Llama-2-7b-chat-hf"
    elif args.model_id == 4:
        model_name = "mpt-7b"
    elif args.model_id == 5:
        model_name = "openchat_3.5"
    elif args.model_id == 6:
        model_name = "zephyr-7b-beta"
    elif args.model_id == 7:
        model_name = "phi-2"
    elif args.model_id == 8:
        model_name = "gemma-7b-it"
    elif args.model_id == 9:
        model_name = "neural-chat-7b-v3-1"

    model_path = "./" + model_name  # TODO supply your model repository path here

    batch_size = args.batch_size

    print(model_path)

    if args.dataset == 'WOS':
        prompt_left = "Classify the topic of the following text into one of seven classes: Computer Science, Electrical Engineering, Psychology, Mechanical Engineering, Civil Engineering, Medical Science, or Biochemistry.\nText: "
        prompt_right = "\nClass:"
    else:
        prompt_left = "Classify the sentiment of the following text into one of three classes: Negative, Neutral, or Positive.\nText: "
        prompt_right = "\nClass:"

    print('#' * 50)
    print(prompt_left)
    print('#' * 50)
    print(prompt_right)
    print('#' * 50)

    if model_name == 'Mistral-7B-v0.1' or model_name == 'falcon-7b' or model_name == 'mpt-7b' \
            or model_name == 'openchat_3.5' or model_name == 'zephyr-7b-beta' or model_name == 'neural-chat-7b-v3-1':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, device_map="auto"
        )
    elif model_name == 'gemma-7b-it':
        batch_size = 4
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, device_map="auto"
        )
    elif model_name == 'Llama-2-7b-chat-hf':
        batch_size = 1
        model = LlamaForCausalLM.from_pretrained(model_path)
        model = model.half().cuda()
    elif model_name == 'Qwen-7B-Chat' or model_name == 'jais-13b-chat':
        batch_size = 1
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
        )
    elif model_name == 'Baichuan-7B':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
        )
    elif model_name == 'phi-2':
        batch_size = 16
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model = model.cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model = model.cuda()


    if model_name == 'Qwen-7B-Chat' or model_name == 'jais-13b-chat' or model_name == 'Baichuan-7B':
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")


    if model_name == 'Mistral-7B-v0.1':
        tokenizer.pad_token = tokenizer.eos_token
    elif model_name == 'openai-gpt':
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer.pad_token = tokenizer.eos_token

    texts = []
    labels = []
    with open('./data/' + args.dataset + '/X.txt', 'r') as f:
        for line in f:
            texts.append(prompt_left + line + prompt_right)
    with open('./data/' + args.dataset + '/y.txt', 'r') as f:
        for line in f:
            labels.append(int(line))

    number_batches = len(texts) // batch_size
    if len(texts) % batch_size != 0:
        number_batches += 1

    preds = []
    cnt = 0
    for i in range(number_batches):
        if i != number_batches - 1:
            texts_batch = texts[cnt:cnt + batch_size]
        else:
            texts_batch = texts[cnt:]

        model_inputs = tokenizer(texts_batch, return_tensors="pt", truncation=True, max_length=500, padding=True).to("cuda")
        generated_ids = model.generate(**model_inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)

        all_prompt_and_generated = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        print('#' * 100)

        for j in range(len(all_prompt_and_generated)):
            prompt_and_generated = all_prompt_and_generated[j]

            prompt_and_generated = prompt_and_generated.lower().strip()
            if model_name == "gemma-7b-it":
                ind = prompt_and_generated.find('the text has a ')
                prompt_and_generated = prompt_and_generated[ind + 15:]
            else:
                ind = prompt_and_generated.find('class:')
                prompt_and_generated = prompt_and_generated[ind + 7:]

            geerated_text = prompt_and_generated.lower().strip()

            if args.dataset == 'SST-5':
                if geerated_text.startswith('very negative'):
                    preds.append(0)
                    print(str(cnt + j) + ': 0')
                elif geerated_text.startswith('negative'):
                    preds.append(1)
                    print(str(cnt + j) + ': 1')
                elif geerated_text.startswith('neutral'):
                    preds.append(2)
                    print(str(cnt + j) + ': 2')
                elif geerated_text.startswith('positive'):
                    preds.append(3)
                    print(str(cnt + j) + ': 3')
                elif geerated_text.startswith('very positive'):
                    preds.append(4)
                    print(str(cnt + j) + ': 4')
                else:
                    # try matching the first class output string
                    out = -1
                    for p in range(len(geerated_text)):
                        try:
                            # longer word sequence must be executed first
                            if geerated_text[:p+1].endswith('very negative'):
                                out = 0
                            elif geerated_text[:p+1].endswith('very positive'):
                                out = 4
                            elif geerated_text[:p+1].endswith('negative'):
                                out = 1
                            elif geerated_text[:p+1].endswith('neutral'):
                                out = 2
                            elif geerated_text[:p+1].endswith('positive'):
                                out = 3
                        except:
                            break
                    if out == -1:
                        print(str(cnt + j) + ': returning a random number')
                        preds.append(random.randint(0, 4))
                    else:
                        preds.append(out)
                        print(str(cnt + j) + ': ' + str(out))
            elif args.dataset == 'WOS':
                if geerated_text.startswith('computer science'):
                    preds.append(0)
                    print(str(cnt + j) + ': 0')
                elif geerated_text.startswith('electrical engineering'):
                    preds.append(1)
                    print(str(cnt + j) + ': 1')
                elif geerated_text.startswith('psychology'):
                    preds.append(2)
                    print(str(cnt + j) + ': 2')
                elif geerated_text.startswith('mechanical engineering'):
                    preds.append(3)
                    print(str(cnt + j) + ': 3')
                elif geerated_text.startswith('civil engineering'):
                    preds.append(4)
                    print(str(cnt + j) + ': 4')
                elif geerated_text.startswith('medical science'):
                    preds.append(4)
                    print(str(cnt + j) + ': 5')
                elif geerated_text.startswith('biochemistry'):
                    preds.append(4)
                    print(str(cnt + j) + ': 6')
                else:
                    # try matching the first class output string
                    out = -1
                    for p in range(len(geerated_text)):
                        try:
                            # longer word sequence must be executed first
                            if geerated_text[:p+1].endswith('computer science'):
                                out = 0
                            elif geerated_text[:p+1].endswith('electrical engineering'):
                                out = 1
                            elif geerated_text[:p+1].endswith('psychology'):
                                out = 2
                            elif geerated_text[:p+1].endswith('mechanical engineering'):
                                out = 3
                            elif geerated_text[:p+1].endswith('civil engineering'):
                                out = 4
                            elif geerated_text[:p+1].endswith('medical science'):
                                out = 5
                            elif geerated_text[:p+1].endswith('biochemistry'):
                                out = 6
                        except:
                            break
                    if out == -1:
                        print(str(cnt + j) + ': returning a random number')
                        preds.append(random.randint(0, 6))
                    else:
                        preds.append(out)
                        print(str(cnt + j) + ': ' + str(out))
            else:
                if geerated_text.startswith('negative'):
                    preds.append(0)
                    print(str(cnt + j) + ': 0')
                elif geerated_text.startswith('neutral'):
                    preds.append(1)
                    print(str(cnt + j) + ': 1')
                elif geerated_text.startswith('positive'):
                    preds.append(2)
                    print(str(cnt + j) + ': 2')
                else:
                    # try matching the first class output string
                    out = -1
                    for p in range(len(geerated_text)):
                        try:
                            if geerated_text[:p+1].endswith('negative'):
                                out = 0
                            elif geerated_text[:p+1].endswith('neutral'):
                                out = 1
                            elif geerated_text[:p+1].endswith('positive'):
                                out = 2
                        except:
                            break
                    if out == -1:
                        print(str(cnt + j) + ': returning a random number')
                        preds.append(random.randint(0, 2))
                    else:
                        preds.append(out)
                        print(str(cnt + j) + ': ' + str(out))
        print('#' * 100)
        cnt += batch_size
    preds = np.array(preds)
    write_to_file(path='./results/' + args.dataset + '/' + model_name + '.csv', data=preds)


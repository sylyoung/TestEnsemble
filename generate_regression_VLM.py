# -*- coding: utf-8 -*-
# @Time    : 2024/6/1
# @Author  : ???
# @File    : generate_regression.py
# generate regression results using VLMs on image-text datasets
import sys, os, argparse, random
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from transformers import AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import AutoConfig, LlamaConfig
from transformers import LlamaForCausalLM
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers.image_utils import load_image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from pathlib import Path


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

    ############################################################configs#################################################
    parser = argparse.ArgumentParser(description='VLM experiment')
    parser.add_argument('--seed', type=int, default=0, help='seed for everything')
    parser.add_argument('--root_dir', type=str, default="/mnt/data2/sylyoung/LLM/", help='model path root dir')
    parser.add_argument('--data_dir', type=str, default="/mnt/data2/sylyoung/Image/", help='model path root dir')
    parser.add_argument('--dataset', type=str, default='AgeDB', help='dataset name')
    parser.add_argument('--gpu_id', type=int, default=-1, help='gpuid')
    parser.add_argument('--model_id', type=int, default=0, help='model id')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for inference')
    args = parser.parse_args()
    seed_everything(args)

    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
    print(f"Available CUDA devices: {torch.cuda.device_count()}")
    print('torch.cuda.is_available()', torch.cuda.is_available())
    try:
        if torch.cuda.device_count() == 1:
            device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    except:
        device = 'local'
    print('device:', device)

    ############################################################model configs###########################################
    print('dataset', args.dataset)

    if args.model_id == 0:
        model_name = "paligemma-3b-mix-448"  # good
    elif args.model_id == 1:
        model_name = "blip-vqa-base"  # good
    elif args.model_id == 2:
        model_name = "h2ovl-mississippi-2b"  # good
    elif args.model_id == 3:
        model_name = "InternVL2-8B"  # good
    elif args.model_id == 4:
        model_name = "deepseek-vl-7b-chat"  # good
    elif args.model_id == 5:
        model_name = "llava-onevision-qwen2-7b-si-hf"  # good
    elif args.model_id == 6:
        model_name = "Molmo-7B-O-0924"  # good
    elif args.model_id == 7:
        model_name = "SmolVLM-Instruct"  # good
    elif args.model_id == 8:
        model_name = "Phi-3.5-vision-instruct"  # good
    elif args.model_id == 9:
        model_name = "SAIL-VL-2B"  # good

    #"POINTS-1-5-Qwen-2-5-7B-Chat"  # bad
    #"Qwen-VL-Chat"  #
    # "glm-4v-9b" #  error
    # "Eagle-X5-7B"  # fail
    # "Monkey-Chat"  # fail
    # "Aquila-VL-2B-llava-qwen"  # fail
    # "Ovis1.5-Llama3-8B"  # bad output text
    # "POINTS-1-5-Qwen-2-5-7B-Chat"  # fail
    # "Mantis-8B-Idefics2"  # bad
    # "glm-4v-9b"  # fail
    # "Idefics3-8B-Llama3"  # fail
    # "MiniCPM-Llama3-V-2_5"  # bad
    #  "deplot"

    model_path = args.root_dir + model_name

    batch_size = args.batch_size

    print(model_path)

    # prompt
    if args.dataset == 'AgeDB':
        prompt = "What is the age of the person in the image?\nAnswer in a single integer:"
    elif args.dataset == 'CFD-age':
        prompt = "Estimate the approximate age of this person (in years)?\nAnswer in a single integer:"
    elif args.dataset == 'CFD-afraid':
        prompt = "On a scale of 1 to 7, how fearful/afraid does the person pictured above appear? (1 = Not at all, 4 = Neutral, 7 = Extremely)\nAnswer in a single float:"
    elif args.dataset == 'CFD-angry':
        prompt = "On a scale of 1 to 7, how angry does the person pictured above appear? (1 = Not at all, 4 = Neutral, 7 = Extremely)\nAnswer in a single float:"
    elif args.dataset == 'CFD-attractive':
        prompt = "On a scale of 1 to 7, how attractive does the person pictured above appear? (1 = Not at all, 4 = Neutral, 7 = Extremely)\nAnswer in a single float:"
    elif args.dataset == 'CFD-babyfaced':
        prompt = "On a scale of 1 to 7, how baby-faced does the person pictured above appear? (1 = Not at all, 4 = Neutral, 7 = Extremely)\nAnswer in a single float:"
    elif args.dataset == 'CFD-disgusted':
        prompt = "On a scale of 1 to 7, how disgusted does the person pictured above appear? (1 = Not at all, 4 = Neutral, 7 = Extremely)\nAnswer in a single float:"
    # elif args.dataset == 'CFD-dominant':
    #     prompt = "On a scale of 1 to 7, how dominant does the person pictured above appear? (1 = Not at all, 4 = Neutral, 7 = Extremely)\nAnswer in a single float:"
    elif args.dataset == 'CFD-feminine':
        prompt = "On a scale of 1 to 7, how feminine does the person pictured above appear? (1 = Not at all, 4 = Neutral, 7 = Extremely)\nAnswer in a single float:"
    elif args.dataset == 'CFD-happy':
        prompt = "On a scale of 1 to 7, how happy does the person pictured above appear? (1 = Not at all, 4 = Neutral, 7 = Extremely)\nAnswer in a single float:"
    elif args.dataset == 'CFD-masculine':
        prompt = "On a scale of 1 to 7, how masculine does the person pictured above appear? (1 = Not at all, 4 = Neutral, 7 = Extremely)\nAnswer in a single float:"
    elif args.dataset == 'CFD-sad':
        prompt = "On a scale of 1 to 7, how sad does the person pictured above appear? (1 = Not at all, 4 = Neutral, 7 = Extremely)\nAnswer in a single float:"
    elif args.dataset == 'CFD-surprised':
        prompt = "On a scale of 1 to 7, how surprised does the person pictured above appear? (1 = Not at all, 4 = Neutral, 7 = Extremely)\nAnswer in a single float:"
    elif args.dataset == 'CFD-threatening':
        prompt = "On a scale of 1 to 7, how threatening does the person pictured above appear? (1 = Not at all, 4 = Neutral, 7 = Extremely)\nAnswer in a single float:"
    elif args.dataset == 'CFD-trustworthy':
        prompt = "On a scale of 1 to 7, how trustworthy does the person pictured above appear? (1 = Not at all, 4 = Neutral, 7 = Extremely)\nAnswer in a single float:"
    elif args.dataset == 'CFD-unusual':
        prompt = "On a scale of 1 to 7, how unusual does the person pictured above appear? (1 = Not at all, 4 = Neutral, 7 = Extremely)\nAnswer in a single float:"
    # elif args.dataset == 'CFD-warm':
    #     prompt = "On a scale of 1 to 7, how warm does the person pictured above appear? (1 = Not at all, 4 = Neutral, 7 = Extremely)\nAnswer in a single float:"
    # elif args.dataset == 'CFD-competent':
    #     prompt = "On a scale of 1 to 7, how competent does the person pictured above appear? (1 = Not at all, 4 = Neutral, 7 = Extremely)\nAnswer in a single float:"
    else:
        sys.exit(0)

    print('#' * 50)
    print(prompt)
    print('#' * 50)

    if model_name == 'Qwen-VL-Chat':
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True).eval()
    elif model_name == 'deepseek-vl-7b-chat':
        from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
        from deepseek_vl.utils.io import load_pil_images
        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer

        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device)
        vl_gpt = vl_gpt.to(torch.bfloat16).eval()
    elif model_name == "POINTS-1-5-Qwen-2-5-7B-Chat":
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     trust_remote_code=True,
                                                     torch_dtype=torch.float16,
                                                     device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        from wepoints.utils.images import Qwen2ImageProcessorForPOINTSV15
        image_processor = Qwen2ImageProcessorForPOINTSV15.from_pretrained(model_path)
    elif model_name == "blip-vqa-base":
        processor = BlipProcessor.from_pretrained(model_path,
                                                  trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16,
                                                  device_map=device
                                                  )
        model = BlipForQuestionAnswering.from_pretrained(model_path).to(device)
    elif model_name == "Mantis-8B-Idefics2":
        from transformers import AutoProcessor, AutoModelForVision2Seq
        processor = AutoProcessor.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True)  # do_image_splitting is False by default
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            device_map="auto"
        )
        generation_kwargs = {
            "max_new_tokens": 1024,
            "num_beams": 1,
            "do_sample": False
        }
    elif model_name == 'h2ovl-mississippi-2b':
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device).eval()
    elif model_name == 'SAIL-VL-2B':
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        def build_transform(input_size):
            MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
            return transform

        def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
            best_ratio_diff = float('inf')
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            return best_ratio

        def dynamic_preprocess(image, min_num=1, max_num=10, image_size=448, use_thumbnail=False):
            orig_width, orig_height = image.size
            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
                i * j <= max_num and i * j >= min_num)
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            target_aspect_ratio = find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size)

            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

            # resize the image
            resized_img = image.resize((target_width, target_height))
            processed_images = []
            for i in range(blocks):
                box = (
                    (i % (target_width // image_size)) * image_size,
                    (i // (target_width // image_size)) * image_size,
                    ((i % (target_width // image_size)) + 1) * image_size,
                    ((i // (target_width // image_size)) + 1) * image_size
                )
                # split the image
                split_img = resized_img.crop(box)
                processed_images.append(split_img)
            assert len(processed_images) == blocks
            if use_thumbnail and len(processed_images) != 1:
                thumbnail_img = image.resize((image_size, image_size))
                processed_images.append(thumbnail_img)
            return processed_images


        def load_image(image_file, input_size=448, max_num=10):
            image = Image.open(image_file).convert('RGB')
            transform = build_transform(input_size=input_size)
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values
    elif model_name == 'Ivy-VL-llava':
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, model_name,
                                                                              device_map="auto")  # Add any other thing you want to pass in llava_model_args

    elif model_name == 'InternVL2-8B':
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            # load_in_8bit=True,
            low_cpu_mem_usage=True,
            # use_flash_attn=True,
            trust_remote_code=True,
            device_map=device).eval()
    elif model_name == "llava-onevision-qwen2-7b-si-hf":
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device
        )
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_name == 'deplot':
        from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
        processor = Pix2StructProcessor.from_pretrained(model_path,
                                                        torch_dtype=torch.bfloat16,
                                                        device_map=device
                                                        )
        model = Pix2StructForConditionalGeneration.from_pretrained(model_path)
    elif model_name == 'Bunny-Llama-3-8B-V':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # float32 for cpu
            device_map='auto',
            trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True)
    elif model_name == "Eagle-X5-7B":
        from eagle.model.builder import load_pretrained_model
        from eagle.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images, \
            KeywordsStoppingCriteria
        from eagle.constants import DEFAULT_IMAGE_TOKEN

        from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, False,
                                                                               False)
        if model.config.mm_use_im_start_end:
            input_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            input_prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    elif model_name == "Valley-Eagle-7B":
        from Valley.valley_eagle_chat import ValleyEagleChat

        model = ValleyEagleChat(
            model_path=model_path,
            padding_side='left',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device
        )
    elif model_name == 'Ovis1.5-Llama3-8B':
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     torch_dtype=torch.bfloat16,
                                                     # multimodal_max_length=8192,
                                                     trust_remote_code=True).to(device)
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()
        conversation_formatter = model.get_conversation_formatter()
    # elif model_name == "cambrian-8b":
    #
    elif model_name == 'MiniCPM-V':
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        # For Nvidia GPUs support BF16 (like A100, H100, RTX3090)
        model = model.to(device=device, dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
    elif model_name == "MiniCPM-Llama3-V-2_5":
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                          torch_dtype=torch.float16)
        model = model.to(device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
    elif model_name == 'Molmo-7B-O-0924':
        processor = AutoProcessor.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16).eval()
    elif model_name == "Monkey-Chat":
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eod_id
    elif model_name == "glm-4v-9b":
        print()
    elif model_name == "Phi-3.5-vision-instruct":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation='eager'
        )
        processor = AutoProcessor.from_pretrained(model_path,
                                                  trust_remote_code=True,
                                                  num_crops=4
                                                  )
    # elif model_name == "Aquila-VL-2B-llava-qwen":
    #     from llava.model.builder import load_pretrained_model
    #     from llava.mm_utils import process_images, tokenizer_image_token
    #     from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    #     from llava.conversation import conv_templates
    #
    #     pretrained = "BAAI/Aquila-VL-2B-llava-qwen"
    #     model_name = "llava_qwen"
    #     device = device
    #     device_map = "auto"
    #     tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, model_name,
    #                                                                           device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    #     model.eval()
    elif model_name == "Idefics3-8B-Llama3":
        processor = AutoProcessor.from_pretrained(model_path)
        from transformers import AutoProcessor, AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(device)
    elif model_name == "SmolVLM-Instruct":
        from transformers import AutoProcessor, AutoModelForVision2Seq

        # Initialize processor and model
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )

        # Create input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            },
        ]
    elif model_name == "paligemma-3b-mix-448":
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_path).eval().to(device)
        processor = AutoProcessor.from_pretrained(model_path)
    else:
        processor = AutoProcessor.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16).eval()


    # image data
    if args.dataset == 'AgeDB':
        image_path_dir = args.data_dir + "AgeDB/"
    elif 'CFD' in args.dataset:
        image_path_dir = args.data_dir + "CFD/Images"
    else:
        sys.exit(0)

    print(image_path_dir)

    first_question = True

    if args.dataset == 'AgeDB':
        files = os.listdir(image_path_dir)
        # Filter and sort the files based on the numerical ID
        sorted_files = sorted(
            files,
            key=lambda x: int(x.split('_')[0])  # Extract and convert the ID to an integer
        )
    elif 'CFD' in args.dataset:
        def process_jpg_images(root_dir):
            """
            Process all JPG images in root_dir and its subdirectories in fixed lexicographical order.

            Args:
                root_dir (str): Path to the root directory containing images
            """
            jpg_files = []

            # Recursively walk through the directory tree
            for dirpath, _, filenames in os.walk(root_dir):
                for filename in filenames:
                    # Check for .jpg or .jpeg extensions (case-insensitive)
                    lower_filename = filename.lower()
                    if lower_filename.endswith(('.jpg', '.jpeg')):
                        full_path = os.path.join(dirpath, filename)
                        jpg_files.append(full_path)

            # Sort paths lexicographically for fixed order
            jpg_files.sort()
            return jpg_files

        sorted_files = process_jpg_images(image_path_dir)
    else:
        sys.exit(0)


    ##########################################################model configs & query#####################################
    cnt = 0
    for file in sorted_files:
        if args.dataset == 'AgeDB':
            image_path = image_path_dir + file
        else:
            image_path = file
        if not image_path.endswith(".jpg"):
            continue
        print(image_path)
        image = load_image(image_path)

        # 1st dialogue turn
        # query = tokenizer.from_list_format([
        #     # {'image': image},
        #     # {'text': prompt},
        #     {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
        #     {'text': '这是什么'},
        # ])
        # response, history = model.chat(tokenizer, query=query, history=None)
        # print(response)

        if model_name == 'Qwen-VL-Chat':
            # Preprocess the image and question
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

            # Perform inference
            with torch.no_grad():
                outputs = model.generate(**inputs)

            # Decode the response
            generated_text = processor.decode(outputs[0], skip_special_tokens=True)
            generated_text = str(generated_text)
            print(generated_text)
        elif model_name == 'deepseek-vl-7b-chat':
            conversation = [
                {
                    "role": "User",
                    "content": "<image_placeholder>" + prompt,
                    "images": [image_path]
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]

            # load images and prepare for inputs
            pil_images = load_pil_images(conversation)
            prepare_inputs = vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(vl_gpt.device)

            # run image encoder to get the image embeddings
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            # run the model to get the response
            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=4,
                do_sample=False,
                use_cache=True
            )

            generated_text = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            print(generated_text)
        elif model_name == "llava-onevision-qwen2-7b-si-hf":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                },
            ]
            prompt_ = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=image, text=prompt_, return_tensors='pt').to(0, torch.float16).to(device)
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            generated_text = processor.decode(output[0][2:], skip_special_tokens=True)
            print(generated_text)
        elif model_name == "POINTS-1-5-Qwen-2-5-7B-Chat":
            content = [
                dict(type='image', image=image_path),
                dict(type='text', text=prompt)
            ]
            messages = [
                {
                    'role': 'user',
                    'content': content
                }
            ]
            generation_config = {
                'max_new_tokens': 1024,
                'temperature': 0.0,
                'top_p': 0.0,
                'num_beams': 1,
            }
            model.generation_config.output_attentions = False
            generated_text = model.chat(
                messages,
                tokenizer,
                image_processor,
                generation_config
            )
            print(generated_text)
        elif model_name == 'MiniCPM-V':

            msgs = [{'role': 'user', 'content': prompt}]

            generated_text, context, _ = model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.7
            )
            print(generated_text)
        elif model_name == 'Bunny-Llama-3-8B-V':
            text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
            text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
            input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to(
                device)

            # image, sample images can be found in images folder
            image = Image.open(image_path)
            image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)

            # generate
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=100,
                use_cache=True,
                repetition_penalty=1.0  # increase this to avoid chattering
            )[0]

            generated_text = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
            print(generated_text)
        elif model_name == 'SAIL-VL-2B':
            # set the max number of tiles in `max_num`
            pixel_values = load_image(image_path, max_num=10).to(torch.bfloat16).to(device)
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            question = '<image>         ' + prompt

            # Get device from language model
            lang_model_device = next(model.language_model.parameters()).device

            # Set CUDA device context for tokenizer
            with torch.cuda.device(lang_model_device):
                generated_text = model.chat(tokenizer, pixel_values, question, generation_config)

            print(generated_text)
        elif model_name == 'Valley-Eagle-7B':

            request = {
                "chat_history": [
                    {'role': 'system',
                     'content': 'You are Valley, developed by ByteDance. Your are a helpfull Assistant.'},
                    {'role': 'user', 'content': prompt},
                ],
                "images": [image],
            }

            generated_text = model(request)
            print(generated_text)
        elif model_name == "Mantis-8B-Idefics2":
            # Note that passing the image urls (instead of the actual pil images) to the processor is also possible
            image1 = load_image(image_path)
            images = [image1]
            print('len(images)', len(images))
            print('prompt', prompt)

            ### Chat
            if first_question:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ]
                    }
                ]

            prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
            inputs = processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate
            generated_ids = model.generate(**inputs, **generation_kwargs)
            generated_text = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            generated_text = generated_text[0]
            print(generated_text)
            first_question = False
        elif model_name == "Eagle-X5-7B":
            from eagle.conversation import conv_templates
            conv_mode = "vicuna_v1"
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            image = Image.open(image_path).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)[0]
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

            input_ids = input_ids.to(device=device, non_blocking=True)
            image_tensor = image_tensor.to(dtype=torch.float16, device=device, non_blocking=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids.unsqueeze(0),
                    images=image_tensor.unsqueeze(0),
                    image_sizes=[image.size],
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.5,
                    num_beams=1,
                    max_new_tokens=256,
                    use_cache=True)

            generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(generated_text)
        elif model_name == 'deplot':
            inputs = processor(images=image, text=prompt,
                               return_tensors="pt")
            predictions = model.generate(**inputs, max_new_tokens=512)
            generated_text = processor.decode(predictions[0], skip_special_tokens=True)
            print(generated_text)
        elif model_name == 'h2ovl-mississippi-2b':
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            question = '<image>\n' + prompt

            # Set CUDA device context for tokenizer
            with torch.cuda.device(device):
                generated_text, history = model.chat(tokenizer, image_path, question, generation_config, history=None,
                                               return_history=True)
            print(generated_text)
        elif model_name == "blip-vqa-base":
            inputs = processor(image, prompt, return_tensors="pt").to(device)

            out = model.generate(**inputs)
            generated_text = processor.decode(out[0], skip_special_tokens=True)
            print(generated_text)
        elif model_name == 'Ivy-VL-llava':
            import copy

            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

            conv_template = "qwen_2_5"  # Make sure you use correct chat template for different models
            question = DEFAULT_IMAGE_TOKEN + prompt
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors="pt").unsqueeze(0).to(device)
            image_sizes = [image.size]

            cont = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )

            generated_text = tokenizer.batch_decode(cont, skip_special_tokens=True)

            print(generated_text)
        elif model_name == 'InternVL2-8B':
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
            question = '<image>\n' + prompt

            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
            IMAGENET_MEAN = (0.485, 0.456, 0.406)
            IMAGENET_STD = (0.229, 0.224, 0.225)

            def build_transform(input_size):
                MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
                transform = T.Compose([
                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                    T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD)
                ])
                return transform

            def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
                best_ratio_diff = float('inf')
                best_ratio = (1, 1)
                area = width * height
                for ratio in target_ratios:
                    target_aspect_ratio = ratio[0] / ratio[1]
                    ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                    if ratio_diff < best_ratio_diff:
                        best_ratio_diff = ratio_diff
                        best_ratio = ratio
                    elif ratio_diff == best_ratio_diff:
                        if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                            best_ratio = ratio
                return best_ratio

            def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
                orig_width, orig_height = image.size
                aspect_ratio = orig_width / orig_height

                # calculate the existing image aspect ratio
                target_ratios = set(
                    (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
                    i * j <= max_num and i * j >= min_num)
                target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

                # find the closest aspect ratio to the target
                target_aspect_ratio = find_closest_aspect_ratio(
                    aspect_ratio, target_ratios, orig_width, orig_height, image_size)

                # calculate the target width and height
                target_width = image_size * target_aspect_ratio[0]
                target_height = image_size * target_aspect_ratio[1]
                blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

                # resize the image
                resized_img = image.resize((target_width, target_height))
                processed_images = []
                for i in range(blocks):
                    box = (
                        (i % (target_width // image_size)) * image_size,
                        (i // (target_width // image_size)) * image_size,
                        ((i % (target_width // image_size)) + 1) * image_size,
                        ((i // (target_width // image_size)) + 1) * image_size
                    )
                    # split the image
                    split_img = resized_img.crop(box)
                    processed_images.append(split_img)
                assert len(processed_images) == blocks
                if use_thumbnail and len(processed_images) != 1:
                    thumbnail_img = image.resize((image_size, image_size))
                    processed_images.append(thumbnail_img)
                return processed_images

            def load_image_internVL(image_file, input_size=448, max_num=12):
                image = Image.open(image_file).convert('RGB')
                transform = build_transform(input_size=input_size)
                images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
                pixel_values = [transform(image) for image in images]
                pixel_values = torch.stack(pixel_values)
                return pixel_values

            pixel_values = load_image_internVL(image_path, max_num=12).to(torch.bfloat16).to(device)
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            generated_text = model.chat(tokenizer, pixel_values, question, generation_config)
            print(generated_text)
        elif model_name == 'Ovis1.5-Llama3-8B':
            query = f'<image>\n' + prompt
            prompt, input_ids = conversation_formatter.format_query(query)
            input_ids = torch.unsqueeze(input_ids, dim=0).to(device=model.device)
            attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id).to(device=model.device)
            pixel_values = [visual_tokenizer.preprocess_image(image).to(
                dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]
            with torch.inference_mode():
                gen_kwargs = dict(
                    max_new_tokens=1024,
                    do_sample=False,
                    top_p=None,
                    top_k=None,
                    temperature=None,
                    repetition_penalty=None,
                    eos_token_id=model.generation_config.eos_token_id,
                    pad_token_id=text_tokenizer.pad_token_id,
                    use_cache=True
                )
                output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
                generated_text = text_tokenizer.decode(output_ids, skip_special_tokens=True)
                print(generated_text)
        elif model_name == "MiniCPM-Llama3-V-2_5":

            msgs = [{'role': 'user', 'content': prompt}]

            generated_text = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,  # if sampling=False, beam_search will be used by default
                temperature=0.7,
                # system_prompt='' # pass system_prompt if needed
            )
            print(generated_text)
        elif model_name == 'Molmo-7B-O-0924':
            inputs = processor.process(
                images=image,
                text=prompt
            )
            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
            with torch.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
                output = model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                    tokenizer=processor.tokenizer
                )
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(generated_text)
        elif model_name == "Monkey-Chat":
            query = f'<img>{image_path}</img> {prompt}'

            input_ids = tokenizer(query, return_tensors='pt', padding='longest')
            attention_mask = input_ids.attention_mask
            input_ids = input_ids.input_ids

            pred = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                do_sample=False,
                num_beams=1,
                max_new_tokens=512,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,
                pad_token_id=tokenizer.eod_id,
                eos_token_id=tokenizer.eod_id,
            )
            generated_text = tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
            print(generated_text)
        elif model_name == "SmolVLM-Instruct":
            # Prepare inputs
            image1 = load_image(
                image_path)

            prompt_ = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt_, images=[image1], return_tensors="pt")
            inputs = inputs.to(device)

            # Generate outputs
            generated_ids = model.generate(**inputs, max_new_tokens=500)
            generated_texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            generated_text = generated_texts[0]
            print(generated_text)
        elif model_name == "glm-4v-9b":
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            image = Image.open(image_path).convert('RGB')
            inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": prompt}],
                                                   add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                                   return_dict=True)  # chat mode
            inputs = inputs.to(device)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=True,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )#.eval()#.to(device).eval()
            gen_kwargs = {"max_length": 100, "do_sample": True, "top_k": 1}
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                generated_text = tokenizer.decode(outputs[0])
                print(generated_text)
        elif model_name == "paligemma-3b-mix-448":
            # Instruct the model to create a caption in Spanish
            model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
            input_len = model_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
                generation = generation[0][input_len:]
                generated_text = processor.decode(generation, skip_special_tokens=True)
                print(generated_text)
        elif model_name == "Phi-3.5-vision-instruct":
            images = []
            placeholder = ""
            images.append(image)

            messages = [
                {"role": "user", "content": f"<|image_{1}|>\n" + prompt},
            ]

            prompt_ = processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = processor(prompt_, image, return_tensors="pt").to(device)

            generation_args = {
                "max_new_tokens": 1000,
                "temperature": 0.0,
                "do_sample": False,
            }

            generate_ids = model.generate(**inputs,
                                          eos_token_id=processor.tokenizer.eos_token_id,
                                          **generation_args
                                          )

            # remove input tokens
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            generated_text = processor.batch_decode(generate_ids,
                                              skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0]

            print(generated_text)
        # elif model_name == "Aquila-VL-2B-llava-qwen":
        #     image_tensor = process_images([image], image_processor, model.config)
        #     image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
        #
        #     conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        #     question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        #     import copy
        #     conv = copy.deepcopy(conv_templates[conv_template])
        #     conv.append_message(conv.roles[0], question)
        #     conv.append_message(conv.roles[1], None)
        #     prompt_question = conv.get_prompt()
        #
        #     input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
        #         0).to(device)
        #     image_sizes = [image.size]
        #
        #     cont = model.generate(
        #         input_ids,
        #         images=image_tensor,
        #         image_sizes=image_sizes,
        #         do_sample=False,
        #         temperature=0,
        #         max_new_tokens=4096,
        #     )
        #
        #     generated_text = tokenizer.batch_decode(cont, skip_special_tokens=True)
        #
        #     print(generated_text)
        elif model_name == "Idefics3-8B-Llama3":

            # Create inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ]
                }
            ]
            prompt_ = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt_, images=[image_path], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            generated_ids = model.generate(**inputs, max_new_tokens=500)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            print(generated_text)
        else:
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs)
            generated_text = processor.decode(outputs[0], skip_special_tokens=True)
            generated_text = str(generated_text)
            print(generated_text)

        if not os.path.isdir('./results/' + args.dataset + '/'):
            path = Path('./results/' + args.dataset + '/')
            path.mkdir(parents=True)

        with open('./results/' + args.dataset + '/' + model_name + '.log', 'a') as f:
            f.write(image_path + '\n')
            f.write(generated_text + '\n')
            f.write('#' * 50 + '\n')
        cnt += 1
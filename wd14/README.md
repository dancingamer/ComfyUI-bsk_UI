# ComfyUI WD 1.4 Tagger
1. WD14Tagger 节点新增功能（wd14tagger.py）
batch_dir 输入框：填入有效目录路径后，自动扫描该目录下所有图片（jpg/jpeg/png/webp/bmp/gif/tiff），逐张反推标签并生成同名 .txt 文件
trigger_word 输入框：触发词会添加到 txt 内容最前面；如果没有英文逗号结尾会自动补逗号（如输入 1girl → 输出 1girl, ...tags）
逻辑：batch_dir 有有效路径时优先执行批量模式，否则走原有的单图/张量模式
2. 独立 CLI 工具（wd14_tagger_cli.py）
完全不依赖 ComfyUI，可直接在命令行使用：

bash
复制
插入
新建文件
保存
插入到终端
应用
# 基本用法
python wd14_tagger_cli.py --dir /path/to/images

# 带触发词和递归扫描
python wd14_tagger_cli.py --dir /path/to/images --trigger_word "1girl" --recursive

# 指定模型和阈值
python wd14_tagger_cli.py --dir /path/to/images --model wd-v1-4-moat-tagger-v2 --threshold 0.35

# 覆盖已有txt
python wd14_tagger_cli.py --dir /path/to/images --overwrite

# 查看可用模型
python wd14_tagger_cli.py --list_models
主要参数：

参数	说明
--dir / -d	目标图片目录（必填）
--model / -m	模型名称
--threshold / -t	通用标签阈值
--character_threshold / -ct	角色标签阈值
--trigger_word / -tw	触发词，自动补逗号
--recursive / -r	递归扫描子目录
--overwrite / -o	覆盖已有 txt
--exclude_tags / -e	排除标签

A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) extension allowing the interrogation of booru tags from images.

Based on [SmilingWolf/wd-v1-4-tags](https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags) and [toriato/stable-diffusion-webui-wd14-tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger)  
All models created by [SmilingWolf](https://huggingface.co/SmilingWolf)

## Installation
1. `git clone https://github.com/pythongosssss/ComfyUI-WD14-Tagger` into the `custom_nodes` folder 
    - e.g. `custom_nodes\ComfyUI-WD14-Tagger`  
2. Open a Command Prompt/Terminal/etc
3. Change to the `custom_nodes\ComfyUI-WD14-Tagger` folder you just created 
    - e.g. `cd C:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-WD14-Tagger` or wherever you have it installed
4.  Install python packages
      - **Windows Standalone installation** (embedded python):   
       `../../../python_embeded/python.exe -s -m pip install -r requirements.txt`  
      - **Manual/non-Windows installation**   
        `pip install -r requirements.txt`

## Usage
Add the node via `image` -> `WD14Tagger|pysssss`  
![image](https://github.com/pythongosssss/ComfyUI-WD14-Tagger/assets/125205205/ee6756ae-73f6-4e9f-a3da-eb87a056eb87)  
Models are automatically downloaded at runtime if missing.  
![image](https://github.com/pythongosssss/ComfyUI-WD14-Tagger/assets/125205205/cc09ae71-1a38-44da-afec-90f470a4b47d)  
Supports tagging and outputting multiple batched inputs.  
- **model**: The interrogation model to use. You can try them out here [WaifuDiffusion v1.4 Tags](https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags). The newest model (as of writing) is `MOAT` and the most popular is `ConvNextV2`.  
- **threshold**: The score for the tag to be considered valid
- **character_threshold**: The score for the character tag to be considered valid
- **exclude_tags** A comma separated list of tags that should not be included in the results

Quick interrogation of images is also available on any node that is displaying an image, e.g. a `LoadImage`, `SaveImage`, `PreviewImage` node.  
Simply right click on the node (or if displaying multiple images, on the image you want to interrogate) and select `WD14 Tagger` from the menu  
![image](https://github.com/pythongosssss/ComfyUI-WD14-Tagger/assets/125205205/11733899-6163-49f6-a22b-8dd86d910de6)

Settings used for this are in the `settings` section of `pysssss.json`.

### Offline Use
Simplest way is to use it online, interrogate an image, and the model will be downloaded and cached, however if you want to manually download the models:
- Create a `models` folder (in same folder as the `wd14tagger.py`)
- Use URLs for models from the list in `pysssss.json`
- Download `model.onnx` and name it with the model name e.g. `wd-v1-4-convnext-tagger-v2.onnx`
- Download `selected_tags.csv` and name it with the model name e.g. `wd-v1-4-convnext-tagger-v2.csv`

## Requirements
`onnxruntime` (recommended, interrogation is still fast on CPU, included in requirements.txt)  
or `onnxruntime-gpu` (allows use of GPU, many people have issues with this, if you try I can't provide support for this)

## Changelog
- 2023-05-14 - Moved to own repo, add downloading models, support multiple inputs
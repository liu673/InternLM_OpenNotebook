
# Finetune简介
![img.png](note_picture%2Fimg.png)
 - 增量预训练微调
 - 指令跟随微调
## 指令跟随微调
指令跟随微调数据中会有 Input 和 Output 希望模型学会的是答案(Output)而不是问题(Input)，训练时只会对答案部分计算Loss。
![img_1.png](note_picture%2Fimg_1.png)
![img_2.png](note_picture%2Fimg_2.png)
![img_3.png](note_picture%2Fimg_3.png)
![img_4.png](note_picture%2Fimg_4.png)

## 增量预训练微调
增量数据微调最终要的不同在于：“让LLM知道什么时候开始一段话，什么时候结束一段话。”
![img_5.png](note_picture%2Fimg_5.png)

## LoRA、QLoRA
![img_6.png](note_picture%2Fimg_6.png)
![img_7.png](note_picture%2Fimg_7.png)

# XTune微调框架
## 简介
![img_8.png](note_picture%2Fimg_8.png)
![img_9.png](note_picture%2Fimg_9.png)
![img_10.png](note_picture%2Fimg_10.png)
## 数据引擎
![img_11.png](note_picture%2Fimg_11.png)
![img_12.png](note_picture%2Fimg_12.png)
![img_13.png](note_picture%2Fimg_13.png)

# XTune特色
![img_14.png](note_picture%2Fimg_14.png)

# 实战
## 配置
```shell
# 如果你是在 InternStudio 平台，则从本地 clone 一个已有 pytorch 2.0.1 的环境：
/root/share/install_conda_env_internlm_base.sh xtuner0.1.9
# 如果你是在其他平台：
conda create --name xtuner0.1.9 python=3.10 -y

# 激活环境
conda activate xtuner0.1.9
# 进入家目录 （~的意思是 “当前用户的home路径”）
cd ~
# 创建版本文件夹并进入，以跟随本教程
mkdir xtuner019 && cd xtuner019


# 拉取 0.1.9 的版本源码
git clone -b v0.1.9  https://github.com/InternLM/xtuner
# 无法访问github的用户请从 gitee 拉取:
# git clone -b v0.1.9 https://gitee.com/Internlm/xtuner

# 进入源码目录
cd xtuner

# 从源码安装 XTuner
pip install -e '.[all]'
```
## 微调
### 准备工作
```shell
# 列出所有内置配置
xtuner list-cfg
```
模型下载
```shell
ln -s /share/temp/model_repos/internlm-chat-7b ~/ft-oasst1/
```
> 以上是通过软链的方式，将模型文件挂载到家目录下，优势是： 
> - 节省拷贝时间，无需等待 
> - 节省用户开发机存储空间

数据下载
```shell
cd ~/ft-oasst1
# ...-guanaco 后面有个空格和英文句号啊
cp -r /root/share/temp/datasets/openassistant-guanaco .
```
![img_15.png](note_picture%2Fimg_15.png)

![img_16.png](note_picture%2Fimg_16.png)

### 开始微调
```shell
# 单卡
## 用刚才改好的config文件训练
xtuner train ./internlm_chat_7b_qlora_oasst1_e3_copy.py --deepspeed deepspeed_zero2

# 多卡
NPROC_PER_NODE=${GPU_NUM} xtuner train ./internlm_chat_7b_qlora_oasst1_e3_copy.py

# 若要开启 deepspeed 加速，增加 --deepspeed deepspeed_zero2 即可
```
![img_17.png](note_picture%2Fimg_17.png)
加速
![img_18.png](note_picture%2Fimg_18.png)
tmux
```shell
apt update -y
apt install tmux -y

tmux new -s finetune
# 返回页面 ctrl + B (松开)+ D
tmux attach -t finetune
# 回到tmux中
```
![img_19.png](note_picture%2Fimg_19.png)
![img_20.png](note_picture%2Fimg_20.png)

### 模型转换
将得到的 PTH 模型转换为 HuggingFace 模型，即：生成 Adapter 文件夹
```shell
xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH_file_dir} ${SAVE_PATH}
```
在本示例中，为：
```shell
mkdir hf
export MKL_SERVICE_FORCE_INTEL=1

xtuner convert pth_to_hf ./internlm_chat_7b_qlora_oasst1_e3_copy.py ./work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy/epoch_1.pth ./hf
```
![img_21.png](note_picture%2Fimg_21.png)

此时，hf 文件夹即为我们平时所理解的所谓 “LoRA 模型文件”
> 可以简单理解：LoRA 模型文件 = Adapter

## 部署与测试
### 模型合并
将 HuggingFace adapter 合并到大语言模型：
```shell
xtuner convert merge ./internlm-chat-7b ./hf ./merged --max-shard-size 2GB
# xtuner convert merge \
#     ${NAME_OR_PATH_TO_LLM} \
#     ${NAME_OR_PATH_TO_ADAPTER} \
#     ${SAVE_PATH} \
#     --max-shard-size 2GB
```
![img_22.png](note_picture%2Fimg_22.png)
### 模型对话
与合并后的模型对话：
```shell
# 加载 Adapter 模型对话（Float 16）
xtuner chat ./merged --prompt-template internlm_chat

# 4 bit 量化加载
# xtuner chat ./merged --bits 4 --prompt-template internlm_chat
```
![img_23.png](note_picture%2Fimg_23.png)

### Demo
```shell
python ./cli_demo.py
```
![img_24.png](note_picture%2Fimg_24.png)





























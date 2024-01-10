# 书生·浦语大模型趣味Demo
**基础作业：**
- 使用 InternLM-Chat-7B 模型生成 300 字的小故事（需截图）。
- 熟悉 hugging face 下载功能，使用 huggingface_hub python 包，下载 InternLM-20B 的 config.json 文件到本地（需截图下载过程）。

**进阶作业：**
- 完成浦语·灵笔的图文理解及创作部署（需截图）
- 完成 Lagent 工具调用 Demo 创作部署（需截图）

**模型介绍**

![img.png](note_picture%2Fimg.png)

![img_1.png](note_picture%2Fimg_1.png)

## Hugging Face 下载 InternLM-20B 
```shell
python hf_download.py 
```
![img_14.png](note_picture%2Fimg_14.png)

## InternLM-Chat-7B 智能对话 Demo
### InternLM-Chat-7B 模型的对话能力
```shell
python /root/code/InternLM/cli_demo.py
```
![img_2.png](note_picture%2Fimg_2.png)
### InternLM-Chat-7B 模型web对话
```shell
streamlit run web_demo.py --server.address 127.0.0.1 --server.port 6006
```
![img_3.png](note_picture%2Fimg_3.png)
![img_4.png](note_picture%2Fimg_4.png)
![img_5.png](note_picture%2Fimg_5.png)

## Lagent 智能体工具调用 Demo
```shell
streamlit run /root/code/lagent/examples/react_web_demo.py --server.address 127.0.0.1 --server.port 6006
```
![img_6.png](note_picture%2Fimg_6.png)
![img_7.png](note_picture%2Fimg_7.png)
![img_8.png](note_picture%2Fimg_8.png)

## 浦语·灵笔图文理解创作 Demo
```shell
python examples/web_demo.py  \
    --folder /root/model/Shanghai_AI_Laboratory/internlm-xcomposer-7b \
    --num_gpus 1 \
    --port 6006
```
![img_9.png](note_picture%2Fimg_9.png)
![img_10.png](note_picture%2Fimg_10.png)
![img_11.png](note_picture%2Fimg_11.png)
![img_12.png](note_picture%2Fimg_12.png)

## 多模态图片理解 Demo
![img_13.png](note_picture%2Fimg_13.png)



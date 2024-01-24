
# LMDeploy 大模型量化部署实践

## 大模型部署背景
### 大模型部署特点
![img.png](note_picture%2Fimg.png)

### 大模型部署挑战与方案
![img_1.png](note_picture%2Fimg_1.png)

## LMDeploy
### 简介
![img_2.png](note_picture%2Fimg_2.png)

### 推理性能
![img_3.png](note_picture%2Fimg_3.png)

### 量化
#### 为什么要做量化
![img_4.png](note_picture%2Fimg_4.png)

#### Weight Only量化
![img_5.png](note_picture%2Fimg_5.png)

![img_6.png](note_picture%2Fimg_6.png)

### 推理引擎TurboMind
![img_7.png](note_picture%2Fimg_7.png)

#### 持续批处理
![img_8.png](note_picture%2Fimg_8.png)

#### 有状态的推理
![img_9.png](note_picture%2Fimg_9.png)

#### Blocked k/v cache
![img_10.png](note_picture%2Fimg_10.png)

#### 高性能的cuda kernel
![img_11.png](note_picture%2Fimg_11.png)

### API server
![img_12.png](note_picture%2Fimg_12.png)

## 实践

### 环境准备
![img_13.png](note_picture%2Fimg_13.png)
```shell
# 解决 ModuleNotFoundError: No module named 'packaging' 问题
pip install packaging
# 使用 flash_attn 的预编译包解决安装过慢问题
pip install /root/share/wheels/flash_attn-2.4.2+cu118torch2.0cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

# 得安装这个版本的lmdeploy，不然容易后续端口转发的时候出现问题
pip install 'lmdeploy[all]==v0.1.0'
```

### 服务部署
#### 模型转换
**在线转换**
![img_14.png](note_picture%2Fimg_14.png)

**离线转换**
![img_15.png](note_picture%2Fimg_15.png)

**TurboMind 推理+命令行本地对话**
```shell
# TurboMind + Bash Local Chat
lmdeploy chat turbomind ./workspace
```
![img_16.png](note_picture%2Fimg_16.png)

**TurboMind推理+API服务**

通过下面命令启动服务。
```shell
# 35602
# ApiServer+Turbomind   api_server => AsyncEngine => TurboMind
lmdeploy serve api_server ./workspace \
	--server_name 0.0.0.0 \
	--server_port 35602 \
	--instance_num 64 \
	--tp 1
```
![img_18.png](note_picture%2Fimg_18.png)
然后，我们可以新开一个窗口，执行下面的 Client 命令。如果使用官方机器，可以打开 vscode 的 Terminal，执行下面的命令。
```shell
# ChatApiClient+ApiServer（注意是http协议，需要加http）
lmdeploy serve api_client http://localhost:35602
```
![img_17.png](note_picture%2Fimg_17.png)

**网页**
```shell
# Gradio+ApiServer。必须先开启 Server，此时 Gradio 为 Client
lmdeploy serve gradio http://0.0.0.0:23333 \
	--server_name 0.0.0.0 \
	--server_port 35602 \
	--restful_api True
```
![img_19.png](note_picture%2Fimg_19.png)









# 基于 InternLM 和 LangChain 搭建你的知识库
## 介绍
### 大模型局限性
 - 时效性
 - 专业能力有限（垂直领域）
 - 定制化成本高
   - (引入 Langchain 帮助大模型解决知识库方面的不足)

![img.png](note_picture%2Fimg.png)

### RAG VS Finetune
![img_1.png](note_picture%2Fimg_1.png)

### RAG
![img_2.png](note_picture%2Fimg_2.png)

## LangChain
Langchain 框架是一个开源工具，通过为各种LLM提供通用接口来简化应用程序的开发流程，帮助开发者自由构建 LLM 应用
![img_3.png](note_picture%2Fimg_3.png)

![img_4.png](note_picture%2Fimg_4.png)

## 构建向量数据库
  - 确定源文件类型，针对不同类型源文件选用不同的加载器
  - 由于单个文档往往超过模型上下限，所以要对加载的文档进行切分
  - 使用向量数据库来支持语义检索，需要将文档向量化存入向量数据库

![img_5.png](note_picture%2Fimg_5.png)

# 搭建知识库助手
![img_6.png](note_picture%2Fimg_6.png)
![img_7.png](note_picture%2Fimg_7.png)
![img_8.png](note_picture%2Fimg_8.png)

# Web Demo
 - Gradio
 - Streamlit

![img_9.png](note_picture%2Fimg_9.png)


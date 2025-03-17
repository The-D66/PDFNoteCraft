# PDFNoteCraft

一个用于处理PDF文件并自动生成中文读书笔记的工具。该工具可以提取PDF中的文本和图片，使用AI模型理解内容，最终生成结构化的读书笔记。

## 功能特点

- 自动提取PDF文件中的文本和图片
- 使用AI模型分析图片内容
- 使用大语言模型生成结构化读书笔记
- 支持并行处理多个PDF文件
- 缓存机制避免重复处理
- 可配置的处理时间窗口

## 安装

1. 克隆仓库
```bash
git clone https://github.com/yourusername/PDFNoteCraft.git
cd PDFNoteCraft
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置
```bash
cp config.example.json config.json
# 编辑 config.json 文件，替换API密钥和其他配置
```

## 使用方法

1. 将需要处理的PDF文件放入`pdf`文件夹
2. 运行程序
```bash
python main.py
```
3. 生成的读书笔记将保存在`markdown_note`文件夹中

## 配置文件说明

### 基础配置

- `pdf_folder`: PDF 文件所在的文件夹路径
- `output_folder`: 生成的 Markdown 笔记输出文件夹路径
- `cache_folder`: 临时缓存文件夹路径
- `log_file`: 日志文件路径

### 并行处理配置

`parallel_processes` 控制程序的并行处理能力：

- `pdf_files`: 同时处理的 PDF 文件数量
- `images`: 同时处理的图片数量
- `text_chunks`: 同时处理的文本块数量

### 模型配置

#### 图像理解模型 (image_model)

- `url`: DashScope API 的基础 URL
- `api_key`: DashScope API 密钥（需要替换为你自己的密钥）
- `name`: 使用的模型名称
- `available_hours`: 模型可用时间（24小时制，数组中的数字表示可用的小时）
- `prompt`: 图像理解提示词
- `max_retries`: API 调用失败时的最大重试次数
- `response_language`: 响应语言

#### 文本理解模型 (text_model)

- `url`: DeepSeek API 的基础 URL
- `api_key`: DeepSeek API 密钥（需要替换为你自己的密钥）
- `name`: 使用的模型名称
- `available_hours`: 模型可用时间（24小时制，数组中的数字表示可用的小时）
- `system_prompt`: 系统提示词，定义了模型的角色和任务要求
- `user_prompt`: 处理新文本块时的提示词
- `continue_prompt`: 处理后续文本块时的提示词（其中 `{previous_summary}` 会被替换为之前的总结）
- `final_summary_prompt`: 生成最终总结时的提示词
- `max_retries`: API 调用失败时的最大重试次数
- `response_language`: 响应语言

### 注意事项

1. API 密钥安全：
   - 不要将包含真实 API 密钥的 `config.json` 提交到版本控制系统
   - 确保 `config.json` 文件的权限设置正确，防止未授权访问

2. 并行处理：
   - 根据你的系统性能和 API 限制调整并行处理参数
   - 设置过高可能导致 API 调用失败或系统负载过重

3. 可用时间：
   - `available_hours` 用于控制 API 调用的时间窗口
   - 可以根据 API 服务商的计费策略和使用限制来调整

## 过滤规则

- 图片处理：程序会自动忽略小于20x20像素的图片，避免处理无意义的小图标或图形元素

## 许可

[MIT License](LICENSE) 
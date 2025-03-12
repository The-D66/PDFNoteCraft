import os
import time
import hashlib
import json
import re
import base64
import pickle
import asyncio
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
import pytz
import logging
import fitz  # PyMuPDF
import requests
import openai

# 设置日志
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

DEBUG = True


# 缓存相关函数
def get_pdf_hash(pdf_path):
  """计算PDF文件的哈希值，用于缓存目录命名"""
  hash_md5 = hashlib.md5()
  with open(pdf_path, "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
      hash_md5.update(chunk)
  return hash_md5.hexdigest()


def get_cache_dir(cache_base, pdf_path):
  """获取PDF对应的缓存目录"""
  pdf_hash = get_pdf_hash(pdf_path)
  cache_dir = os.path.join(cache_base, pdf_hash)
  os.makedirs(cache_dir, exist_ok=True)

  # 创建info.json记录原始文件信息
  info_file = os.path.join(cache_dir, "info.json")
  if not os.path.exists(info_file):
    pdf_info = {
        "original_path": pdf_path,
        "filename": os.path.basename(pdf_path),
        "created_time": datetime.now().isoformat(),
        "last_access": datetime.now().isoformat()
    }
    with open(info_file, 'w', encoding='utf-8') as f:
      json.dump(pdf_info, f, indent=2)
  else:
    # 更新访问时间
    with open(info_file, 'r', encoding='utf-8') as f:
      pdf_info = json.load(f)
    pdf_info["last_access"] = datetime.now().isoformat()
    with open(info_file, 'w', encoding='utf-8') as f:
      json.dump(pdf_info, f, indent=2)

  return cache_dir


def save_cache(cache_dir, cache_name, data):
  """保存数据到缓存文件"""
  cache_path = os.path.join(cache_dir, f"{cache_name}.pickle")
  with open(cache_path, 'wb') as f:
    pickle.dump(data, f)
  logging.info(f"已保存缓存: {cache_path}")
  return cache_path


def load_cache(cache_dir, cache_name):
  """从缓存加载数据，如果不存在则返回None"""
  cache_path = os.path.join(cache_dir, f"{cache_name}.pickle")
  if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
      data = pickle.load(f)
    logging.info(f"已加载缓存: {cache_path}")
    return data
  return None


def save_text_cache(cache_dir, text):
  """将文本内容保存到缓存中"""
  text_path = os.path.join(cache_dir, "extracted_text.txt")
  with open(text_path, 'w', encoding='utf-8') as f:
    f.write(text)
  logging.info(f"已保存文本缓存: {text_path}")


def load_text_cache(cache_dir):
  """从缓存中加载文本内容"""
  text_path = os.path.join(cache_dir, "extracted_text.txt")
  if os.path.exists(text_path):
    with open(text_path, 'r', encoding='utf-8') as f:
      text = f.read()
    logging.info(f"已加载文本缓存: {text_path}")
    return text
  return None


# 时间相关函数
def is_time_to_send(available_hours=None):
  """检查当前时间是否在可调用时间内"""
  if DEBUG:
    return True

  tz = pytz.timezone('Asia/Shanghai')
  now = datetime.now(tz)

  # 如果指定了可调用时间，则检查当前小时是否在可调用时间内
  if available_hours:
    return now.hour in available_hours

  # 默认规则：1:00~8:00
  return 1 <= now.hour < 8


def wait_until_available(available_hours):
  """等待直到当前时间在可调用时间内"""
  if is_time_to_send(available_hours):
    return

  tz = pytz.timezone('Asia/Shanghai')
  now = datetime.now(tz)

  # 找到下一个可调用的时间
  next_hour = None
  for hour in sorted(available_hours):
    if hour > now.hour:
      next_hour = hour
      break

  # 如果当天没有可调用时间了，则找第二天的第一个可调用时间
  if next_hour is None:
    next_hour = min(available_hours)
    next_time = (now + timedelta(days=1)
                ).replace(hour=next_hour, minute=0, second=0, microsecond=0)
  else:
    next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)

  sleep_seconds = (next_time - now).total_seconds()
  logging.info(f"等待至 {next_time} 以发送API请求")
  time.sleep(sleep_seconds)


# PDF处理相关函数
def extract_pdf_content(pdf_path, cache_dir):
  """提取PDF中的文本和图片，并缓存结果"""
  # 检查是否有缓存的文本
  cached_text = load_text_cache(cache_dir)
  cached_images = load_cache(cache_dir, "extracted_images")

  if cached_text && cached_images:
    return cached_text, cached_images

  # 没有缓存，从PDF提取
  doc = fitz.open(pdf_path)
  text = ""
  images = []

  for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    text += f"\n[第 {page_num + 1} 页文本]\n" + page.get_text()

    for img in page.get_images(full=True):
      xref = img[0]
      base_image = doc.extract_image(xref)
      image_bytes = base_image["image"]

      # 保存图片到缓存目录
      image_path = os.path.join(cache_dir, f"image_{page_num}_{xref}.png")
      with open(image_path, "wb") as img_file:
        img_file.write(image_bytes)

      images.append((page_num, image_path))

  # 保存提取结果到缓存
  save_text_cache(cache_dir, text)
  save_cache(cache_dir, "extracted_images", images)

  return text, images


async def process_image(image_path, page_num, model_config, cache_dir):
  """异步处理单张图片，返回图片总结"""
  # 检查是否有图片总结缓存
  cache_name = f"image_summary_{os.path.basename(image_path)}"
  cached_summary = load_cache(cache_dir, cache_name)

  if cached_summary:
    return page_num, cached_summary

  # 等待时间窗口
  wait_until_available(model_config["available_hours"])

  # 调用API进行图像理解
  try:
    client = openai.OpenAI(
        base_url=model_config["url"], api_key=model_config["api_key"]
    )

    with open(image_path, "rb") as img_file:
      response = client.chat.completions.create(
          model=model_config["name"],
          messages=[
              {
                  "role":
                      "user",
                  "content":
                      [
                          {
                              "type": "text",
                              "text": "这张图片中包含什么内容？请提供一个简短的总结。"
                          }, {
                              "type": "image_url",
                              "image_url":
                                  {
                                      "url":
                                          f"data:image/png;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"
                                  }
                          }
                      ]
              }
          ]
      )

    summary = response.choices[0].message.content

    # 缓存结果
    save_cache(cache_dir, cache_name, summary)

    return page_num, summary
  except Exception as e:
    logging.error(f"处理图片 {image_path} 失败: {str(e)}")
    return page_num, f"图像理解失败: {str(e)}"


# 文本处理相关函数
def estimate_token_count(text):
  """粗略估计文本中的token数量，按照平均4个字符一个token计算"""
  return len(text) / 4


def split_text_into_chunks(text, max_tokens=20000):
  """将文本分割成较小的块，每块不超过指定的token数量"""
  # ... [保持原有代码不变] ...
  chunks = []
  # 按段落分割文本
  paragraphs = re.split(r'\n\s*\n', text)

  current_chunk = ""
  current_tokens = 0

  for paragraph in paragraphs:
    paragraph_tokens = estimate_token_count(paragraph)

    # 如果单个段落超过最大token限制，则进一步分割
    if paragraph_tokens > max_tokens:
      words = paragraph.split()
      temp_paragraph = ""
      temp_tokens = 0

      for word in words:
        word_tokens = estimate_token_count(word + " ")
        if temp_tokens + word_tokens <= max_tokens:
          temp_paragraph += word + " "
          temp_tokens += word_tokens
        else:
          if current_tokens + temp_tokens <= max_tokens:
            current_chunk += temp_paragraph
            current_tokens += temp_tokens
          else:
            chunks.append(current_chunk)
            current_chunk = temp_paragraph
            current_tokens = temp_tokens

          temp_paragraph = word + " "
          temp_tokens = word_tokens

      if temp_paragraph:
        if current_tokens + temp_tokens <= max_tokens:
          current_chunk += temp_paragraph
          current_tokens += temp_tokens
        else:
          chunks.append(current_chunk)
          current_chunk = temp_paragraph
          current_tokens = temp_tokens
    else:
      # 检查添加此段落是否会超出限制
      if current_tokens + paragraph_tokens <= max_tokens:
        current_chunk += paragraph + "\n\n"
        current_tokens += paragraph_tokens
      else:
        # 如果会超出限制，则开始新的块
        chunks.append(current_chunk)
        current_chunk = paragraph + "\n\n"
        current_tokens = paragraph_tokens

  # 添加最后一个块
  if current_chunk:
    chunks.append(current_chunk)

  return chunks


async def process_text_chunk(
    client, model_name, chunk, previous_summary, cache_dir, chunk_index
):
  """异步处理文本块，返回总结"""
  # 检查缓存
  cache_name = f"text_summary_chunk_{chunk_index}"
  cached_summary = load_cache(cache_dir, cache_name)

  if cached_summary:
    return cached_summary

  # 生成总结
  try:
    messages = [{"role": "system", "content": "你是一个助手，专门为给定的文本生成中文读书笔记。"}]

    if previous_summary:
      prompt = f"下面是我之前总结的内容：\n\n{previous_summary}\n\n请基于上面的总结，继续为以下新的内容生成读书笔记，保持连贯性和一致性：\n\n{chunk}"
    else:
      prompt = f"请为以下文本撰写中文读书笔记: {chunk}"

    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model_name, messages=messages
    )
    summary = response.choices[0].message.content

    # 缓存结果
    save_cache(cache_dir, cache_name, summary)

    return summary
  except Exception as e:
    logging.error(f"生成总结失败: {e}")
    return f"生成总结失败: {str(e)}"


async def generate_final_summary(
    client, model_name, accumulated_summary, cache_dir
):
  """异步生成最终总结"""
  # 检查缓存
  cache_name = "final_summary"
  cached_summary = load_cache(cache_dir, cache_name)

  if cached_summary:
    return cached_summary

  try:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "你是一个助手，专门为给定的文本生成中文读书笔记。"
            }, {
                "role": "user",
                "content": f"请为以下内容撰写一个精炼的最终读书笔记: {accumulated_summary}"
            }
        ]
    )
    summary = response.choices[0].message.content

    # 缓存结果
    save_cache(cache_dir, cache_name, summary)

    return summary
  except Exception as e:
    logging.error(f"生成最终总结失败: {e}")
    return accumulated_summary  # 失败时返回原累积总结


async def process_pdf(pdf_path, config):
  """处理单个PDF文件的主要流程"""
  try:
    notes_path = os.path.join(
        config["output_folder"],
        os.path.basename(pdf_path).replace('.pdf', '.md')
    )
    # 先确认是否存在已输出的笔记
    if os.path.exists(notes_path):
      logging.info(f"{pdf_path} 的读书笔记已存在于 {notes_path}，跳过处理")
      return "skipped"

    # 准备缓存目录（检查是否存在缓存文件）
    cache_dir = get_cache_dir(config["cache_folder"], pdf_path)

    logging.info(f"正在处理 {pdf_path}")
    # 提取PDF内容
    text, images = extract_pdf_content(pdf_path, cache_dir)
    logging.info(f"已从 {pdf_path} 提取文本和图片")

    # 并行异步处理所有图片
    image_model_config = config["models"]["image_model"]
    text_model_config = config["models"]["text_model"]

    # 启动图片处理任务
    image_tasks = []
    for page_num, image_path in images:
      image_tasks.append(
          process_image(image_path, page_num, image_model_config, cache_dir)
      )

    # 异步等待所有图片处理完成
    image_results = await asyncio.gather(*image_tasks)

    # 整理图片结果
    image_summaries = {}
    for page_num, summary in image_results:
      image_summaries[page_num] = summary

    # 分批处理文本以避免超出token限制
    text_chunks = split_text_into_chunks(text)
    logging.info(f"已将文本分为 {len(text_chunks)} 个批次进行处理")

    # 初始化文本模型客户端
    text_client = openai.OpenAI(
        base_url=text_model_config["url"], api_key=text_model_config["api_key"]
    )

    # 等待文本处理时间窗口
    wait_until_available(text_model_config["available_hours"])

    # 逐批处理文本
    accumulated_summary = ""
    for i, chunk in enumerate(text_chunks):
      # 添加相关页面的图片总结到文本块
      chunk_with_images = chunk

      # 查找与当前文本块相关的图片
      # 假设按页码分布：每个块处理文档的一部分页面
      pages_per_chunk = max(1, len(images) // len(text_chunks))
      start_page = i * pages_per_chunk
      end_page = (i + 1) * pages_per_chunk if i < len(text_chunks
                                                     ) - 1 else float('inf')

      # 添加本文本块涉及页面的图片总结
      related_images = [
          f"[图片总结 (第 {page_num + 1} 页): {image_summaries.get(page_num, '无总结')}]"
          for page_num, _ in images if start_page <= page_num < end_page
      ]

      if related_images:
        chunk_with_images += "\n\n" + "\n".join(related_images)

      logging.info(f"处理第 {i+1}/{len(text_chunks)} 批文本")

      # 生成当前批次的总结
      summary_chunk = await process_text_chunk(
          text_client, text_model_config["name"], chunk_with_images,
          accumulated_summary, cache_dir, i
      )

      # 累积总结结果
      if accumulated_summary:
        accumulated_summary = f"{accumulated_summary}\n\n{summary_chunk}"
      else:
        accumulated_summary = summary_chunk

      # 保存中间结果
      save_cache(cache_dir, "accumulated_summary", accumulated_summary)

      # 防止API请求过于频繁
      await asyncio.sleep(5)

    # 如果文本比较长（有多个块），生成最终摘要
    final_summary = accumulated_summary
    if len(text_chunks) > 1:
      logging.info(f"生成最终总结")
      final_summary = await generate_final_summary(
          text_client, text_model_config["name"], accumulated_summary, cache_dir
      )

    # 保存读书笔记到Markdown文件
    with open(notes_path, 'w', encoding='utf-8') as f:
      f.write(final_summary)
    logging.info(f"已将 {pdf_path} 的读书笔记保存至 {notes_path}")

    # 撰写笔记完成后，根据DEBUG模式决定是否删除对应缓存目录
    if not DEBUG:
      import shutil
      shutil.rmtree(cache_dir)
    else:
      logging.info("DEBUG模式下保留缓存")
      
    return True
  except Exception as e:
    logging.error(f"处理 {pdf_path} 失败: {str(e)}")
    return False


async def main_async():
  """异步主函数"""
  # 加载配置文件
  config = None
  try:
    with open("config.json", 'r', encoding='utf-8') as f:
      config = json.load(f)
    logging.info(f"已成功加载配置文件")
  except Exception as e:
    logging.error(f"加载配置文件失败: {e}")
    return

  # 确保输出和缓存文件夹存在
  os.makedirs(config["output_folder"], exist_ok=True)
  os.makedirs(config["cache_folder"], exist_ok=True)

  # 获取所有PDF文件
  pdf_files = []
  for root, _, files in os.walk(config["pdf_folder"]):
    for file in files:
      if file.endswith('.pdf'):
        pdf_files.append(os.path.join(root, file))

  # 限制并行处理的数量
  semaphore = asyncio.Semaphore(config.get("parallel_processes", 2))

  async def process_with_semaphore(pdf_path):
    async with semaphore:
      return await process_pdf(pdf_path, config)

  # 并行处理所有PDF文件
  tasks = [process_with_semaphore(pdf_path) for pdf_path in pdf_files]
  results = await asyncio.gather(*tasks)

  # 统计成功/失败数量
  success_count = sum(1 for r in results if r is True)
  skipped_count = sum(1 for r in results if r == "skipped")
  failure_count = sum(1 for r in results if r is False)

  logging.info(
      f"处理完成! 成功: {success_count}, 跳过: {skipped_count}, 失败: {failure_count}"
  )


def main():
  """程序入口点"""
  asyncio.run(main_async())


if __name__ == "__main__":
  main()

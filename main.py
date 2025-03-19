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
from logging.handlers import RotatingFileHandler
import fitz  # PyMuPDF
import requests
import openai
from PIL import Image


def setup_logging(log_file):
  """设置日志配置，包括控制台和文件输出"""
  # 确保日志目录存在
  os.makedirs(os.path.dirname(log_file), exist_ok=True)

  # 获取根日志记录器并清除现有的处理器
  root_logger = logging.getLogger()
  for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

  # 创建格式化器
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

  # 设置根日志记录器级别
  root_logger.setLevel(logging.INFO)

  # 创建并配置文件处理器（使用 RotatingFileHandler 进行日志轮转）
  file_handler = RotatingFileHandler(
      log_file,
      maxBytes=10 * 1024 * 1024,  # 10MB
      backupCount=5,
      encoding='utf-8'
  )
  file_handler.setLevel(logging.ERROR)  # 文件只记录 ERROR 级别及以上的日志
  file_handler.setFormatter(formatter)

  # 创建并配置控制台处理器
  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.INFO)
  console_handler.setFormatter(formatter)

  # 将处理器添加到根日志记录器
  root_logger.addHandler(file_handler)
  root_logger.addHandler(console_handler)


DEBUG = False


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
  logging.info("已保存缓存: %s", cache_path)
  return cache_path


def load_cache(cache_dir, cache_name):
  """从缓存加载数据，如果不存在则返回None"""
  cache_path = os.path.join(cache_dir, f"{cache_name}.pickle")
  if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
      data = pickle.load(f)
    logging.info("已加载缓存: %s", cache_path)
    return data
  return None


def save_text_cache(cache_dir, text):
  """将文本内容保存到缓存中"""
  text_path = os.path.join(cache_dir, "extracted_text.txt")
  with open(text_path, 'w', encoding='utf-8') as f:
    f.write(text)
  logging.info("已保存文本缓存: %s", text_path)


def load_text_cache(cache_dir):
  """从缓存中加载文本内容"""
  text_path = os.path.join(cache_dir, "extracted_text.txt")
  if os.path.exists(text_path):
    with open(text_path, 'r', encoding='utf-8') as f:
      text = f.read()
    logging.info("已加载文本缓存: %s", text_path)
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
  logging.info("等待至 %s 以发送API请求", next_time)
  time.sleep(sleep_seconds)


async def async_wait_until_available(available_hours):
  """异步等待直到当前时间在可调用时间内"""
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
  logging.info("异步等待至 %s 以生成笔记", next_time)
  await asyncio.sleep(sleep_seconds)


# PDF处理相关函数
def extract_pdf_content(pdf_path, cache_dir, config=None):
  """提取PDF中的文本和图片，并缓存结果"""
  # 检查是否有缓存的文本
  cached_text = load_text_cache(cache_dir)
  cached_images = load_cache(cache_dir, "extracted_images")

  if cached_text and cached_images:
    return cached_text, cached_images

  # 没有缓存，从PDF提取
  doc = fitz.open(pdf_path)

  # 检查PDF页数是否超过限制
  if config and "max_pdf_pages" in config:
    max_pages = config["max_pdf_pages"]
    if len(doc) > max_pages:
      logging.warning("跳过处理：%s 页数(%d)超过限制(%d)", pdf_path, len(doc), max_pages)
      return f"页数超过限制（{len(doc)} > {max_pages}）", []

  text = ""
  images = []

  for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    text += f"\n[第 {page_num + 1} 页文本]\n" + page.get_text()

    for img in page.get_images(full=True):
      xref = img[0]
      base_image = doc.extract_image(xref)
      image_bytes = base_image["image"]

      # 检查图片尺寸
      width = base_image.get("width", 0)
      height = base_image.get("height", 0)

      # 忽略小于20x20像素的图片
      if width < 20 or height < 20:
        logging.info("忽略小图片：第 %d 页，大小 %dx%d 像素", page_num + 1, width, height)
        continue

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
  cache_name = f"image_summary_{os.path.basename(image_path)}"
  cached_summary = load_cache(cache_dir, cache_name)

  if cached_summary:
    return page_num, cached_summary

  # 检查图片大小是否满足要求
  try:
    with Image.open(image_path) as img:
      width, height = img.size
      # 忽略小于100x100像素的图片
      if width < 100 or height < 100:
        logging.info("跳过小图片处理: %s (大小 %dx%d)", image_path, width, height)
        return page_num, "图片太小，不进行处理"
      # 忽略大于10MB的图片
      file_size = os.path.getsize(image_path)
      if file_size > 10 * 1024 * 1024:  # 10MB
        logging.info(
            "跳过大图片处理: %s (大小 %.2fMB)", image_path, file_size / 1024 / 1024
        )
        return page_num, "图片太大，不进行处理"
  except Exception as e:
    logging.warning("检查图片大小失败: %s, 错误: %s", image_path, str(e))
    # 继续处理，不因检查失败而中断

  # 使用异步版本代替同步版本
  await async_wait_until_available(model_config["available_hours"])

  retries = 0
  last_error = None

  while retries < model_config["max_retries"]:
    try:
      client = openai.OpenAI(
          base_url=model_config["url"], api_key=model_config["api_key"]
      )

      with open(image_path, "rb") as img_file:
        image_data = img_file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

      logging.info("发送图片处理请求 (页 %d)", page_num + 1)
      response = await asyncio.to_thread(
          lambda: client.chat.completions.create(
              model=model_config["name"],
              messages=[
                  {
                      "role":
                          "user",
                      "content":
                          [
                              {
                                  "type":
                                      "text",
                                  "text":
                                      f"{model_config['prompt']} 请使用{model_config['response_language']}回答。"
                              }, {
                                  "type": "image_url",
                                  "image_url":
                                      {
                                          "url":
                                              f"data:image/png;base64,{image_base64}"
                                      }
                              }
                          ]
                  }
              ]
          )
      )

      summary = response.choices[0].message.content
      save_cache(cache_dir, cache_name, summary)
      logging.info("图片处理完成 (页 %d)", page_num + 1)
      return page_num, summary

    except Exception as e:
      last_error = str(e)
      retries += 1
      if retries < model_config["max_retries"]:
        wait_time = 2**retries  # 指数退避
        logging.error(
            "处理图片 %s 失败 (尝试 %d/%d): %s", image_path, retries,
            model_config["max_retries"], str(e)
        )
        await asyncio.sleep(wait_time)
      else:
        logging.error("处理图片 %s 最终失败: %s", image_path, str(e))

  return page_num, f"图像理解失败 (重试 {model_config['max_retries']} 次): {last_error}"


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
    client, model_name, chunk, previous_summary, cache_dir, chunk_index,
    text_model_config
):
  """异步处理文本块，返回总结"""
  cache_name = f"text_summary_chunk_{chunk_index}"
  cached_summary = load_cache(cache_dir, cache_name)

  if cached_summary:
    return cached_summary

  retries = 0
  last_error = None

  while retries < text_model_config["max_retries"]:
    try:
      messages = [
          {
              "role":
                  "system",
              "content":
                  f"{text_model_config['system_prompt']} 请使用{text_model_config['response_language']}回答。"
          }
      ]

      if previous_summary:
        prompt = text_model_config["continue_prompt"].replace(
            "{previous_summary}", previous_summary
        ) + f"\n\n{chunk}"
      else:
        prompt = f"{text_model_config['user_prompt']} {chunk}"

      messages.append({"role": "user", "content": prompt})

      # 使用非阻塞的客户端调用
      logging.info("发送文本处理请求 (块 %d, 重试 %d)", chunk_index, retries)
      response = await asyncio.to_thread(
          lambda: client.chat.completions.
          create(model=model_name, messages=messages)
      )

      summary = response.choices[0].message.content
      save_cache(cache_dir, cache_name, summary)
      logging.info("文本块 %d 处理完成", chunk_index)
      return summary

    except Exception as e:
      last_error = str(e)
      retries += 1
      if retries < text_model_config["max_retries"]:
        wait_time = 2**retries  # 指数退避
        logging.error(
            "生成总结失败 (块 %d, 尝试 %d/%d): %s", chunk_index, retries,
            text_model_config["max_retries"], str(e)
        )
        await asyncio.sleep(wait_time)
      else:
        logging.error("生成总结最终失败 (块 %d): %s", chunk_index, str(e))

  return f"生成总结失败 (块 {chunk_index}, 重试 {text_model_config['max_retries']} 次): {last_error}"


async def generate_final_summary(
    client, model_name, accumulated_summary, cache_dir, text_model_config
):
  """异步生成最终总结"""
  cache_name = "final_summary"
  cached_summary = load_cache(cache_dir, cache_name)

  if cached_summary:
    return cached_summary

  retries = 0
  last_error = None

  while retries < text_model_config["max_retries"]:
    try:
      logging.info("开始生成最终总结")
      response = await asyncio.to_thread(
          lambda: client.chat.completions.create(
              model=model_name,
              messages=[
                  {
                      "role": "system",
                      "content": text_model_config["system_prompt"]
                  }, {
                      "role":
                          "user",
                      "content":
                          f"{text_model_config['final_summary_prompt']} {accumulated_summary}"
                  }
              ]
          )
      )
      summary = response.choices[0].message.content
      save_cache(cache_dir, cache_name, summary)
      logging.info("最终总结生成完成")
      return summary

    except Exception as e:
      last_error = str(e)
      retries += 1
      if retries < text_model_config["max_retries"]:
        wait_time = 2**retries  # 指数退避
        logging.error(
            "生成最终总结失败 (尝试 %d/%d): %s", retries,
            text_model_config["max_retries"], str(e)
        )
        await asyncio.sleep(wait_time)
      else:
        logging.error("生成最终总结最终失败: %s", str(e))

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
      logging.info("%s 的读书笔记已存在于 %s，跳过处理", pdf_path, notes_path)
      return "skipped"

    # 准备缓存目录（检查是否存在缓存文件）
    cache_dir = get_cache_dir(config["cache_folder"], pdf_path)

    logging.info("正在处理 %s", pdf_path)
    # 提取PDF内容
    text, images = extract_pdf_content(pdf_path, cache_dir, config)
    logging.info("已从 %s 提取文本和图片", pdf_path)

    # 并行异步处理所有图片
    image_model_config = config["models"]["image_model"]
    text_model_config = config["models"]["text_model"]

    # 创建图片处理的信号量
    image_semaphore = asyncio.Semaphore(config["parallel_processes"]["images"])
    text_semaphore = asyncio.Semaphore(
        config["parallel_processes"]["text_chunks"]
    )

    # 启动图片处理任务，但不等待结果
    async def process_image_with_semaphore(image_path, page_num):
      async with image_semaphore:
        return await process_image(
            image_path, page_num, image_model_config, cache_dir
        )

    image_tasks = [
        process_image_with_semaphore(image_path, page_num)
        for page_num, image_path in images
    ]

    # 分批处理文本以避免超出token限制
    text_chunks = split_text_into_chunks(text)
    logging.info(f"已将文本分为 {len(text_chunks)} 个批次进行处理")

    # 初始化文本模型客户端
    text_client = openai.OpenAI(
        base_url=text_model_config["url"], api_key=text_model_config["api_key"]
    )

    # 等待文本处理时间窗口
    await async_wait_until_available(text_model_config["available_hours"])

    # 创建处理文本块的函数
    async def process_chunk_with_semaphore(
        chunk, chunk_index, previous_summary, image_results
    ):
      async with text_semaphore:
        # 查找与当前文本块相关的图片
        pages_per_chunk = max(1, len(images) // len(text_chunks))
        start_page = chunk_index * pages_per_chunk
        end_page = (chunk_index + 1) * pages_per_chunk if chunk_index < len(
            text_chunks
        ) - 1 else float('inf')

        # 获取已完成的图片总结
        completed_images = {
            page_num: summary
            for page_num, summary in image_results
            if start_page <= page_num < end_page
        }

        # 添加本文本块涉及页面的图片总结
        related_images = [
            f"[图片总结 (第 {page_num + 1} 页): {completed_images.get(page_num, '处理中...')}]"
            for page_num, _ in images if start_page <= page_num < end_page
        ]

        chunk_with_images = chunk
        if related_images:
          chunk_with_images += "\n\n" + "\n".join(related_images)

        return await process_text_chunk(
            text_client, text_model_config["name"], chunk_with_images,
            previous_summary, cache_dir, chunk_index, text_model_config
        )

    # 创建一个异步队列来存储完成的图片结果
    image_results_queue = asyncio.Queue()

    # 创建一个任务来收集图片处理结果
    async def collect_image_results():
      completed_results = []
      for task in asyncio.as_completed(image_tasks):
        result = await task
        completed_results.append(result)
        await image_results_queue.put(result)
      return completed_results

    # 启动收集图片结果的任务
    image_collector_task = asyncio.create_task(collect_image_results())

    # 并行处理文本块
    chunk_tasks = []
    previous_summaries = {}  # 用于存储每个块的前序总结

    # 创建所有文本块的处理任务
    for i, chunk in enumerate(text_chunks):
      previous_summary = previous_summaries.get(i - 1, "") if i > 0 else ""
      # 获取当前已完成的图片结果
      current_image_results = []
      while not image_results_queue.empty():
        current_image_results.append(await image_results_queue.get())

      task = process_chunk_with_semaphore(
          chunk, i, previous_summary, current_image_results
      )
      chunk_tasks.append(task)

    # 等待所有文本块处理完成
    chunk_results = await asyncio.gather(*chunk_tasks)

    # 等待所有图片处理完成
    all_image_results = await image_collector_task

    # 合并所有文本块的总结
    accumulated_summary = "\n\n".join(chunk_results)

    # 如果文本比较长（有多个块），生成最终摘要
    final_summary = accumulated_summary
    if len(text_chunks) > 1:
      logging.info(f"生成最终总结")
      final_summary = await generate_final_summary(
          text_client, text_model_config["name"], accumulated_summary,
          cache_dir, text_model_config
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

    # 设置日志
    setup_logging(config["log_file"])
    logging.info("已成功加载配置文件")
  except Exception as e:
    logging.error("加载配置文件失败: %s", str(e))
    return

  # 确保输出和缓存文件夹存在
  os.makedirs(config["output_folder"], exist_ok=True)
  os.makedirs(config["cache_folder"], exist_ok=True)

  # 获取所有PDF文件并去重
  pdf_files = set()
  for root, _, files in os.walk(config["pdf_folder"]):
    for file in files:
      if file.endswith('.pdf'):
        abs_path = os.path.abspath(os.path.join(root, file))
        pdf_files.add(abs_path)

  if not pdf_files:
    logging.info("未找到PDF文件")
    return

  logging.info("找到 %d 个PDF文件待处理", len(pdf_files))

  # 创建三个有限容量的队列分别用于文本提取、图片处理和笔记生成
  extraction_queue = asyncio.Queue(
      maxsize=config["parallel_processes"]["pdf_files"] * 40
  )  # 待提取的PDF
  image_queue = asyncio.Queue(
      maxsize=config["parallel_processes"]["images"] * 20
  )  # 待处理的图片任务
  note_queue = asyncio.Queue(
      maxsize=config["parallel_processes"]["text_chunks"] * 80
  )  # 待生成笔记的文档

  # 创建信号量控制并发
  extraction_semaphore = asyncio.Semaphore(
      config["parallel_processes"]["pdf_files"]
  )
  image_semaphore = asyncio.Semaphore(config["parallel_processes"]["images"])

  # 检查配置中是否有并行笔记生成的配置，如果没有，默认使用与pdf_files相同的值
  parallel_notes = config["parallel_processes"].get(
      "notes", config["parallel_processes"]["pdf_files"]
  )
  logging.info("并行笔记生成数量: %d", parallel_notes)

  note_semaphore = asyncio.Semaphore(parallel_notes)

  # 存储处理结果
  doc_results = {}
  # 存储每个PDF的图片总数和已处理图片数
  doc_image_progress = {}

  # 用于跟踪文档的所有图片是否已处理完成
  async def check_and_queue_for_note_generation(pdf_path):
    """检查PDF的所有图片是否已处理完成，如果完成则将其加入笔记生成队列"""
    if pdf_path in doc_image_progress:
      total_images, processed_images = doc_image_progress[pdf_path]
      if processed_images >= total_images:
        logging.info("%s 的全部图片处理完成，加入笔记生成队列", pdf_path)
        await note_queue.put(pdf_path)
        # 防止重复加入队列
        doc_image_progress[pdf_path] = (total_images, -1)

  # 文本提取工作者
  async def extraction_worker():
    while True:
      try:
        pdf_path = await extraction_queue.get()
        if pdf_path is None:  # 结束信号
          extraction_queue.task_done()
          break

        async with extraction_semaphore:
          # 检查是否已存在笔记
          notes_path = os.path.join(
              config["output_folder"],
              os.path.basename(pdf_path).replace('.pdf', '.md')
          )
          if os.path.exists(notes_path):
            logging.info("%s 的读书笔记已存在于 %s，跳过处理", pdf_path, notes_path)
            extraction_queue.task_done()
            continue

          # 准备缓存目录
          cache_dir = get_cache_dir(config["cache_folder"], pdf_path)

          # 提取内容
          logging.info("开始提取 %s 的内容", pdf_path)
          text, images = extract_pdf_content(pdf_path, cache_dir, config)
          logging.info("已从 %s 提取文本和图片", pdf_path)

          # 检查是否为页数超标的文件
          if isinstance(text, str) and text.startswith("页数超过限制"):
            logging.warning("%s 页数超过限制，跳过笔记生成", pdf_path)
            # 记录为已处理但不加入笔记生成队列
            doc_results[pdf_path] = {
                "text": text,
                "images": {},
                "cache_dir": cache_dir,
                "processing": False,  # 标记为已处理完成
                "total_images": 0
            }
            extraction_queue.task_done()
            continue

          # 初始化文档结果
          doc_results[pdf_path] = {
              "text": text,
              "images": {},
              "cache_dir": cache_dir,
              "processing": True,
              "total_images": len(images)
          }

          # 记录图片数量
          doc_image_progress[pdf_path] = (len(images), 0)

          # 如果没有图片，直接加入笔记生成队列
          if len(images) == 0:
            logging.info("%s 没有图片，直接加入笔记生成队列", pdf_path)
            await note_queue.put(pdf_path)
          else:
            # 将图片任务加入队列
            logging.info("将 %s 的 %d 张图片加入处理队列", pdf_path, len(images))
            for page_num, image_path in images:
              await image_queue.put((pdf_path, image_path, page_num))

        extraction_queue.task_done()

      except Exception as e:
        logging.error("提取工作者出错: %s", str(e))
        extraction_queue.task_done()

  # 图片处理工作者
  async def image_worker():
    while True:
      try:
        task = await image_queue.get()
        if task is None:  # 结束信号
          image_queue.task_done()
          break

        pdf_path, image_path, page_num = task

        # 只处理正在进行中的文档
        if pdf_path in doc_results and doc_results[pdf_path]["processing"]:
          logging.info("开始处理图片: %s 第 %d 页", pdf_path, page_num + 1)
          result = await process_image(
              image_path, page_num, config["models"]["image_model"],
              doc_results[pdf_path]["cache_dir"]
          )

          # 保存结果
          doc_results[pdf_path]["images"][page_num] = result[1]

          # 更新进度
          if pdf_path in doc_image_progress:
            total, processed = doc_image_progress[pdf_path]
            doc_image_progress[pdf_path] = (total, processed + 1)
            logging.info(
                "完成图片处理: %s 第 %d 页 (%d/%d)", pdf_path, page_num + 1,
                processed + 1, total
            )

            # 检查是否所有图片都已处理完成
            await check_and_queue_for_note_generation(pdf_path)

        image_queue.task_done()

      except Exception as e:
        logging.error("图片处理工作者出错: %s", str(e))
        # 更新失败计数但仍然标记为处理完成
        if pdf_path in doc_image_progress:
          total, processed = doc_image_progress[pdf_path]
          doc_image_progress[pdf_path] = (total, processed + 1)
          await check_and_queue_for_note_generation(pdf_path)
        image_queue.task_done()

  # 笔记生成工作者
  async def note_worker():
    while True:
      try:
        pdf_path = await note_queue.get()
        if pdf_path is None:  # 结束信号
          note_queue.task_done()
          break

        # 只处理正在进行中的文档
        if pdf_path in doc_results and doc_results[pdf_path]["processing"]:
          # 再次检查文本是否因页数超标而被跳过
          text = doc_results[pdf_path].get("text", "")
          if isinstance(text, str) and text.startswith("页数超过限制"):
            logging.warning("%s 页数超过限制，跳过笔记生成", pdf_path)
            doc_results[pdf_path]["processing"] = False
            note_queue.task_done()
            continue

          # 检查是否可以生成笔记（是否在允许的时间窗口内）
          text_model_config = config["models"]["text_model"]
          if not is_time_to_send(text_model_config["available_hours"]):
            # 如果不在时间窗口内，将任务重新放回队列末尾，并暂时让出控制权
            # 这样可以处理其他可能已经在时间窗口内的PDF
            logging.info("%s 不在可处理时间窗口内，重新入队等待下一次检查", pdf_path)
            await note_queue.put(pdf_path)
            note_queue.task_done()
            # 暂停一段时间再继续处理队列中的下一个任务
            await asyncio.sleep(60)  # 60秒后再尝试处理队列
            continue

          logging.info("开始生成 %s 的笔记", pdf_path)
          # 直接处理，不使用信号量
          await process_doc(pdf_path, config, doc_results[pdf_path]["images"])
          logging.info("完成 %s 的笔记生成", pdf_path)

          # 清理
          doc_results[pdf_path]["processing"] = False
          if not DEBUG:
            try:
              import shutil
              if os.path.exists(doc_results[pdf_path]["cache_dir"]):
                shutil.rmtree(doc_results[pdf_path]["cache_dir"])
            except Exception as e:
              logging.error("清理缓存失败: %s", str(e))

        note_queue.task_done()

      except Exception as e:
        logging.error("笔记生成工作者出错: %s", str(e))
        if pdf_path in doc_results:
          doc_results[pdf_path]["processing"] = False
        note_queue.task_done()

  # 启动工作者
  extraction_workers = [
      asyncio.create_task(extraction_worker())
      for _ in range(config["parallel_processes"]["pdf_files"])
  ]
  image_workers = [
      asyncio.create_task(image_worker())
      for _ in range(config["parallel_processes"]["images"])
  ]
  note_workers = [
      asyncio.create_task(note_worker())
      for _ in range(parallel_notes)  # 使用笔记并行数量
  ]

  # 将所有PDF加入提取队列
  for pdf_path in pdf_files:
    await extraction_queue.put(pdf_path)

  # 发送结束信号
  for _ in range(len(extraction_workers)):
    await extraction_queue.put(None)

  # 等待提取队列处理完成
  await extraction_queue.join()

  # 发送图片队列结束信号
  for _ in range(len(image_workers)):
    await image_queue.put(None)

  # 等待图片队列处理完成
  await image_queue.join()

  # 发送笔记队列结束信号
  for _ in range(len(note_workers)):
    await note_queue.put(None)

  # 等待笔记队列处理完成
  await note_queue.join()

  # 等待所有工作者结束
  await asyncio.gather(*extraction_workers)
  await asyncio.gather(*image_workers)
  await asyncio.gather(*note_workers)

  # 统计处理结果
  success_count = sum(1 for v in doc_results.values() if not v["processing"])
  skipped_count = len(pdf_files) - len(doc_results)
  total_count = len(pdf_files)

  logging.info(
      f"处理完成! 成功: {success_count}, 跳过: {skipped_count}, 总计: {total_count}"
  )

  # 启动一个定期检查时间窗口的任务
  async def time_window_checker():
    while True:
      try:
        # 每5分钟检查一次
        await asyncio.sleep(300)

        # 检查是否有未完成的PDF文档需要处理
        text_model_config = config["models"]["text_model"]

        # 如果进入了可用时间窗口，则检查是否有需要处理但尚未加入笔记队列的文档
        if is_time_to_send(text_model_config["available_hours"]):
          logging.info("进入可用时间窗口，检查未处理的文档")
          for pdf_path, doc_info in doc_results.items():
            # 跳过页数超标的文件
            text = doc_info.get("text", "")
            if isinstance(text, str) and text.startswith("页数超过限制"):
              continue

            # 如果文档处理中，且图片处理已完成，但尚未生成笔记
            if (
                doc_info["processing"] and pdf_path in doc_image_progress and
                doc_image_progress[pdf_path][0]
                == doc_image_progress[pdf_path][1]
            ):

              # 将文档加入笔记生成队列（如果尚未在队列中）
              if not any(item == pdf_path for item in list(note_queue._queue)):
                logging.info("时间窗口已到，将 %s 加入笔记生成队列", pdf_path)
                await note_queue.put(pdf_path)
      except Exception as e:
        logging.error("时间窗口检查任务出错: %s", str(e))

  # 启动时间窗口检查任务
  time_check_task = asyncio.create_task(time_window_checker())

  # 等待所有任务完成
  # ... existing code ...

  # 取消时间窗口检查任务
  time_check_task.cancel()
  try:
    await time_check_task
  except asyncio.CancelledError:
    pass


async def process_doc(pdf_path, config, image_results):
  """处理单个文献的文本内容"""
  try:
    notes_path = os.path.join(
        config["output_folder"],
        os.path.basename(pdf_path).replace('.pdf', '.md')
    )
    cache_dir = get_cache_dir(config["cache_folder"], pdf_path)

    # 获取文本内容
    text, _ = extract_pdf_content(pdf_path, cache_dir, config)

    # 检查是否是页数超标的文件
    if isinstance(text, str) and text.startswith("页数超过限制"):
      logging.warning("%s 页数超过限制，跳过笔记生成", pdf_path)
      return False

    # 分批处理文本
    text_chunks = split_text_into_chunks(text)
    logging.info("已将文本分为 %d 个批次进行处理", len(text_chunks))

    # 初始化文本模型客户端
    text_model_config = config["models"]["text_model"]

    # 等待文本处理时间窗口
    await async_wait_until_available(text_model_config["available_hours"])

    # 处理文本块 - 完全并行处理
    max_concurrent = config["parallel_processes"]["text_chunks"]
    chunk_tasks = []
    previous_summaries = {}

    # 创建一个独立的客户端实例用于并行处理
    async def process_chunk_parallel(chunk_index, chunk):
      # 为每个任务创建单独的客户端实例，避免共享导致的锁
      text_client = openai.OpenAI(
          base_url=text_model_config["url"],
          api_key=text_model_config["api_key"]
      )

      previous_summary = previous_summaries.get(
          chunk_index - 1, ""
      ) if chunk_index > 0 else ""

      # 获取当前块相关的图片结果
      pages_per_chunk = max(1, len(image_results) // len(text_chunks))
      start_page = chunk_index * pages_per_chunk
      end_page = (chunk_index + 1) * pages_per_chunk if chunk_index < len(
          text_chunks
      ) - 1 else float('inf')

      # 添加相关图片总结
      related_images = [
          f"[图片总结 (第 {page_num + 1} 页): {image_results.get(page_num, '处理中...')}]"
          for page_num in
          range(start_page, int(min(end_page, len(image_results))))
      ]

      chunk_with_images = chunk
      if related_images:
        chunk_with_images += "\n\n" + "\n".join(related_images)

      result = await process_text_chunk(
          text_client, text_model_config["name"], chunk_with_images,
          previous_summary, cache_dir, chunk_index, text_model_config
      )

      # 存储结果供后续块使用
      previous_summaries[chunk_index] = result
      return chunk_index, result

    # 创建所有文本块的处理任务
    tasks = [
        process_chunk_parallel(i, chunk) for i, chunk in enumerate(text_chunks)
    ]

    # 使用 asyncio.gather 并行执行所有任务
    results = await asyncio.gather(*tasks)

    # 按原始顺序整理结果
    results.sort(key=lambda x: x[0])
    chunk_results = [result for _, result in results]

    # 合并所有文本块的总结
    accumulated_summary = "\n\n".join(chunk_results)

    # 如果需要，生成最终总结
    final_summary = accumulated_summary
    if len(text_chunks) > 1:
      logging.info("生成最终总结")
      # 为最终总结创建新的客户端实例
      final_client = openai.OpenAI(
          base_url=text_model_config["url"],
          api_key=text_model_config["api_key"]
      )
      final_summary = await generate_final_summary(
          final_client, text_model_config["name"], accumulated_summary,
          cache_dir, text_model_config
      )

    # 保存读书笔记
    with open(notes_path, 'w', encoding='utf-8') as f:
      f.write(final_summary)
    logging.info("已将 %s 的读书笔记保存至 %s", pdf_path, notes_path)

    # 清理缓存
    if not DEBUG:
      import shutil
      shutil.rmtree(cache_dir)

    return True

  except Exception as e:
    logging.error("处理文献 %s 失败: %s", pdf_path, str(e))
    return False


def main():
  """程序入口点"""
  asyncio.run(main_async())


if __name__ == "__main__":
  main()

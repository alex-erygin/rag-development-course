#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Постраничная конвертация PDF -> Markdown через OpenRouter PDF Inputs.

https://openrouter.ai/docs/features/multimodal/pdfs

Особенности:
- Разбивает исходный PDF на отдельные страницы (в памяти)
- Отправляет каждую страницу как одно-страничный PDF в OpenRouter (file-data)
- Полученный Markdown по каждой странице сохраняет в директорию "pages" (page-001.md, page-002.md, ...)
- Дополнительно формирует объединенный Markdown-файл со всем содержимым (в порядке страниц)

Все параметры заданы хардкодом в секции HARDCODED CONFIG. CLI-параметров и переменных окружения не требуется.

Требования:
- OpenRouter API ключ
- Подключенные зависимости (см. requirements.txt): PyPDF2, requests

ВНИМАНИЕ: Использование движка "mistral-ocr" тарифицируется. Движок "pdf-text" бесплатный для текстовых PDF.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PyPDF2 import PdfReader, PdfWriter


# ===================== HARDCODED CONFIG =====================
# Обязательный API-ключ OpenRouter (замените плейсхолдер на свой ключ)
OPENROUTER_API_KEY = "sk-"

# Путь к исходному PDF (локальный)
SOURCE_PDF_PATH = (Path(__file__).parent / "NASA Technical Standard-System Safety.pdf").resolve()

# Директория для постраничных Markdown-файлов
OUTPUT_PAGES_DIR = (Path(__file__).parent / "pages - NASA Technical Standard-System Safety").resolve()

# Итоговый объединенный Markdown-файл
OUTPUT_COMBINED_MD = (Path(__file__).parent / "NASA Technical Standard-System Safety.pdf.md").resolve()

# Модель и движок парсинга PDF (через plugin "file-parser")
MODEL = "anthropic/claude-opus-4.1" #"google/gemini-2.5-pro"  #"google/gemini-2.5-pro"  #"anthropic/claude-opus-4.1" #"qwen/qwen3-max"#"deepseek/deepseek-chat-v3.1" #"openai/gpt-5" #"anthropic/claude-sonnet-4.5" #"google/gemini-2.5-pro"  #"z-ai/glm-4.6"  # Примеры: google/gemma-3-27b-it, openai/gpt-4o
ENGINE = "pdf-text"  # варианты: "pdf-text" | "mistral-ocr" | "native"

# Параметры генерации/запроса
TEMPERATURE = 0.0
MAX_TOKENS: Optional[int] = None
TITLE_HEADER: Optional[str] = None
REFERER_HEADER: Optional[str] = None
REQUEST_TIMEOUT_SEC = 600

# Диапазон страниц для обработки (1-based, включительно). None = весь документ
PAGE_START: Optional[int] = 101
PAGE_END: Optional[int] = 101

# Промпт для постраничной конвертации
DEFAULT_PAGE_PROMPT = """You are a document-to-Markdown converter.
Process ONLY the attached single-page PDF and output a clean, well-structured Markdown for THIS PAGE ONLY:
- Preserve headings hierarchy (#, ##, ###) when evident.
- Convert bullet/numbered lists.
- Convert simple tables to Markdown tables when possible.
- If the page includes diagrams, flowcharts, architecture schematics, screenshots, UI mockups, or graphs, recreate them as high-fidelity ASCII art using monospaced characters, preserving layout, proportions, labels, connectors, legends, and captions.
- Render each ASCII image inside a fenced code block with no language tag (```) or with 'text'. Do NOT use Mermaid or any other diagram languages. Do NOT embed bitmap images or HTML.
- Immediately under each ASCII image, add a detailed description starting with "Image description:", enumerating key elements, their relationships, all text labels (transcribed), and notable visual features.
- Do NOT use fenced code blocks for regular text, lists, or tables. Use plain Markdown for all non-diagram content to ensure optimal rendering in Obsidian.
- Keep inline formatting (bold, italics) if clearly present.
- Do NOT include content from other pages.
- Do NOT add commentary or explanations beyond what appears on this page. Exception: include the required "Описание изображения:" paragraph under each ASCII image. Output ONLY the Markdown content for this page.
- Translate all text to Russian Language
"""
# ============================================================


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def encode_single_page_to_data_url(reader: PdfReader, page_index: int) -> str:
    """
    Возвращает data URL (base64) одно-страничного PDF для страницы page_index.
    """
    writer = PdfWriter()
    writer.add_page(reader.pages[page_index])
    with io.BytesIO() as buf:
        writer.write(buf)
        data = buf.getvalue()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:application/pdf;base64,{b64}"


def build_messages(prompt_text: str, filename: str, file_data: str) -> List[Dict[str, Any]]:
    """
    Формирует messages для OpenRouter с текстом промпта и file-блоком PDF.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "file",
                    "file": {
                        "filename": filename,
                        "file_data": file_data,
                    },
                },
            ],
        }
    ]


def call_openrouter(
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    engine: Optional[str],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    title_header: Optional[str] = None,
    referer: Optional[str] = None,
    timeout_sec: int = 600,
) -> Tuple[str, Dict[str, Any]]:
    """
    Вызывает OpenRouter Chat Completions API с плагином file-parser (при необходимости).
    Возвращает текст ассистента (Markdown) и сырой JSON-ответ.
    """
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if referer:
        headers["HTTP-Referer"] = referer
    if title_header:
        headers["X-Title"] = title_header

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    if engine:
        payload["plugins"] = [{"id": "file-parser", "pdf": {"engine": engine}}]

    resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=timeout_sec)

    # JSON разбор независимо от статуса
    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"OpenRouter response is not JSON. Status={resp.status_code}, Body={resp.text[:500]}")

    if resp.status_code >= 400 or "error" in data:
        err = data.get("error") or data
        raise RuntimeError(f"OpenRouter error: {json.dumps(err, ensure_ascii=False)}")

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"No choices in OpenRouter response: {json.dumps(data)[:500]}")

    message = choices[0].get("message", {})
    content = message.get("content")

    # Контент может быть строкой или массивом блоков
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                txt = part.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        content_text = "\n".join(parts).strip()
    elif isinstance(content, str):
        content_text = content.strip()
    else:
        content_text = (str(content) if content is not None else "").strip()

    if not content_text:
        raise RuntimeError("Empty content returned from OpenRouter.")

    return content_text, data


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    # Проверка ключа
    api_key = OPENROUTER_API_KEY
    if not api_key or "REPLACE_ME" in api_key:
        raise SystemExit("Пожалуйста, задайте OPENROUTER_API_KEY в коде скрипта (переменная OPENROUTER_API_KEY).")

    # Проверка исходного PDF
    pdf_path = SOURCE_PDF_PATH
    if not pdf_path.exists():
        raise SystemExit(f"Исходный PDF не найден: {pdf_path}")

    # Читаем PDF
    reader = PdfReader(str(pdf_path))
    num_pages = len(reader.pages)
    print(f"Найдено страниц: {num_pages}")

    # Вычисление и валидация диапазона страниц
    start_page = PAGE_START or 1
    end_page = PAGE_END or num_pages
    if start_page < 1:
        start_page = 1
    if end_page > num_pages:
        end_page = num_pages
    if start_page > end_page:
        raise SystemExit(f"Некорректный диапазон страниц: start={start_page}, end={end_page}, всего={num_pages}")
    selected_count = end_page - start_page + 1
    print(f"Будет обработано страниц: {selected_count} (диапазон {start_page}-{end_page})")

    combined_parts: List[str] = []
    OUTPUT_PAGES_DIR.mkdir(parents=True, exist_ok=True)

    for page_num in range(start_page, end_page + 1):
        idx = page_num - 1
        print(f"[{page_num}/{num_pages}] Обработка страницы...")

        # Подготовка одно-страничного PDF (data URL)
        data_url = encode_single_page_to_data_url(reader, idx)

        # Формируем сообщения и вызываем OpenRouter
        filename = f"{pdf_path.stem}-page-{page_num:03}.pdf"
        messages = build_messages(DEFAULT_PAGE_PROMPT, filename, data_url)

        md_text, _raw = call_openrouter(
            api_key=api_key,
            model=MODEL,
            messages=messages,
            engine=ENGINE,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            title_header=TITLE_HEADER,
            referer=REFERER_HEADER,
            timeout_sec=REQUEST_TIMEOUT_SEC,
        )

        # Сохранение страницы в pages/page-XXX.md
        page_md_path = OUTPUT_PAGES_DIR / f"page-{page_num:03}.md"
        save_text(page_md_path, md_text)
        print(f"Сохранено: {page_md_path}")

        # Добавляем в общий Markdown (с разделителем страниц)
        combined_parts.append(f"<!-- Page {page_num} -->\n\n{md_text}\n")

    # Сохраняем объединенный файл
    combined_text = "\n\n".join(combined_parts).strip() + "\n"
    model_suffix = str(MODEL).replace("/", "-")
    combined_md_with_model = OUTPUT_COMBINED_MD.with_name(f"{OUTPUT_COMBINED_MD.stem}.{model_suffix}{OUTPUT_COMBINED_MD.suffix}")
    save_text(combined_md_with_model, combined_text)
    print(f"Объединенный Markdown сохранен: {combined_md_with_model}")


if __name__ == "__main__":
    main()

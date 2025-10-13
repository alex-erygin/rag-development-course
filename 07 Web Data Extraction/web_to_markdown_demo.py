"""
Демонстрация извлечения данных из веб-страниц в формат Markdown с использованием Crawl4AI

Этот скрипт показывает, как:
1. Извлекать контент из одной веб-страницы
2. Извлекать данные из нескольких страниц
3. Сохранять результаты в файлы markdown
"""

import asyncio
import os
from pathlib import Path
from datetime import datetime
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter


async def extract_single_page(url: str, output_dir: str = "output"):
    """
    Извлекает контент из одной веб-страницы и сохраняет в markdown
    
    Args:
        url: URL веб-страницы для извлечения
        output_dir: Директория для сохранения результатов
    """
    print(f"\n{'='*60}")
    print(f"Извлечение данных из: {url}")
    print(f"{'='*60}\n")
    
    # Создаем директорию для результатов (включая родительские директории)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Настройка браузера
    browser_config = BrowserConfig(
        headless=True,  # Запуск без GUI
        verbose=True    # Подробный вывод
    )
    
    # Настройка процесса извлечения
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,  # Не использовать кеш
        word_count_threshold=10,       # Минимальное количество слов
    )
    
    # Создаем crawler и извлекаем данные
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url=url,
            config=run_config
        )
        
        if result.success:
            # Генерируем имя файла из URL
            filename = url.replace('https://', '').replace('http://', '').replace('/', '_')
            if len(filename) > 50:
                filename = filename[:50]
            filename = f"{filename}.md"
            
            output_path = Path(output_dir) / filename
            
            # Сохраняем markdown
            with open(output_path, 'w', encoding='utf-8') as f:
                # Добавляем метаданные
                f.write(f"# Извлечено из: {url}\n\n")
                f.write(f"**Дата извлечения:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Заголовок страницы:** {result.metadata.get('title', 'Не найден')}\n\n")
                f.write("---\n\n")
                
                # Основной контент
                f.write(result.markdown)
            
            print(f"✓ Успешно извлечено!")
            print(f"  - Размер контента: {len(result.markdown)} символов")
            print(f"  - Сохранено в: {output_path}")
            print(f"  - Заголовок: {result.metadata.get('title', 'Не найден')}")
            
            return output_path
        else:
            print(f"✗ Ошибка при извлечении: {result.error_message}")
            return None

async def extract_with_clean_markdown(url: str, output_dir: str = "output"):
    """
    Извлекает контент с использованием "чистого" markdown (удаление шума)
    
    Args:
        url: URL веб-страницы
        output_dir: Директория для сохранения
    """
    print(f"\n{'='*60}")
    print(f"Извлечение с очисткой контента из: {url}")
    print(f"{'='*60}\n")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    browser_config = BrowserConfig(
        headless=True,
        verbose=True,
        browser_type="undetected",
        extra_args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-web-security"
        ]
    )
    
    # Используем fit_markdown для получения очищенного контента
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=1,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=0.35,
                threshold_type="fixed",
                min_word_threshold=0
            )
        )
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config, magic=True)
        
        if result.success:
            filename = "cleaned_" + url.replace('https://', '').replace('http://', '').replace('/', '_')[:40] + ".md"
            output_path = Path(output_dir) / filename
            
            # Сохраняем два варианта: сырой и очищенный markdown
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# Очищенный контент\n\n")
                f.write(f"**Источник:** {url}\n\n")
                f.write("---\n\n")
                
                # Используем fit_markdown при наличии, иначе fallback на raw_markdown/markdown
                md = result.markdown
                used_content = None
                try:
                    if hasattr(md, "fit_markdown") and md.fit_markdown and len(md.fit_markdown.strip()) > 100:
                        used_content = md.fit_markdown
                    elif hasattr(md, "raw_markdown") and md.raw_markdown and len(md.raw_markdown.strip()) > 0:
                        used_content = md.raw_markdown
                    elif isinstance(md, str) and len(md.strip()) > 0:
                        used_content = md
                except Exception:
                    pass
                if not used_content:
                    used_content = str(md) if md is not None else ""
                f.write(used_content)
            
            print(f"✓ Очищенный контент сохранен!")
            print(f"  - Файл: {output_path}")
            print(f"  - Размер: {len(used_content)} символов")
            
            return output_path
        else:
            print(f"✗ Ошибка: {result.error_message}")
            return None
        
async def main():
    """
    Главная функция с примерами использования
    """
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ИЗВЛЕЧЕНИЯ ВЕБ-ДАННЫХ В MARKDOWN")
    print("="*60)
    
    # Создаем директорию для результатов
    output_dir = "07 Web Data Extraction/extracted_content"
    
    # Пример 1: Извлечение из одной страницы
    await extract_single_page(
        url="https://ru.wikipedia.org/wiki/%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0_%D1%82%D1%80%D1%91%D1%85_%D1%82%D0%B5%D0%BB",
        output_dir=output_dir
    )

    # Пример 2: Извлечение с очисткой контента
    print("\n\n--- ПРИМЕР 2: Очищенный контент ---")
    await extract_with_clean_markdown(
        url="https://ru.wikipedia.org/wiki/%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0_%D1%82%D1%80%D1%91%D1%85_%D1%82%D0%B5%D0%BB",
        output_dir=output_dir
    )
        
    # Итоговая статистика
    print("\n\n" + "="*60)
    print("ЗАВЕРШЕНО!")
    print("="*60)
    print(f"Результаты сохранены в: {output_dir}")
    print("\nДля просмотра результатов откройте файлы .md в вашем редакторе")


if __name__ == "__main__":
    # Запускаем демонстрацию
    asyncio.run(main())

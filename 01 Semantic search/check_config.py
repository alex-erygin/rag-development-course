#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для проверки и отображения текущей конфигурации
"""

import os
from config import *

def check_configuration():
    """Проверка и отображение текущей конфигурации"""
    print("🔧 ПРОВЕРКА КОНФИГУРАЦИИ СИСТЕМЫ ВЕКТОРНОГО ПОИСКА")
    print("=" * 60)
    
    # Проверка путей
    print("\n📁 ПУТИ И ДИРЕКТОРИИ:")
    print(f"  KB_DIRECTORY: {KB_DIRECTORY}")
    print(f"  CHROMA_DB_PATH: {CHROMA_DB_PATH}")
    
    # Проверка существования директории KB
    if os.path.exists(KB_DIRECTORY):
        kb_files = [f for f in os.listdir(KB_DIRECTORY) if f.endswith('.txt')]
        print(f"  ✅ Директория {KB_DIRECTORY} существует")
        print(f"  📄 Найдено {len(kb_files)} текстовых файлов: {kb_files}")
    else:
        print(f"  ❌ Директория {KB_DIRECTORY} не существует")
    
    # Проверка существования базы данных
    if os.path.exists(CHROMA_DB_PATH):
        print(f"  ✅ База данных ChromaDB существует")
    else:
        print(f"  ⚠️  База данных ChromaDB не существует (будет создана автоматически)")
    
    # Настройки модели
    print("\n🤖 НАСТРОЙКИ ЛОКАЛЬНОЙ МОДЕЛИ:")
    print(f"  EMBEDDING_MODEL_ENDPOINT: {EMBEDDING_MODEL_ENDPOINT}")
    print(f"  EMBEDDING_MODEL_ID: {EMBEDDING_MODEL_ID}")
    print(f"  EMBEDDING_API_KEY: {EMBEDDING_API_KEY}")
    
    # Настройки ChromaDB
    print("\n🗄️  НАСТРОЙКИ CHROMADB:")
    print(f"  COLLECTION_NAME: {COLLECTION_NAME}")
    
    # Настройки поиска
    print("\n🔍 НАСТРОЙКИ ПОИСКА:")
    print(f"  DEFAULT_SEARCH_RESULTS: {DEFAULT_SEARCH_RESULTS}")
    print(f"  MAX_CONTENT_PREVIEW_LENGTH: {MAX_CONTENT_PREVIEW_LENGTH}")
    
    # Тестовые запросы
    print("\n🧪 ТЕСТОВЫЕ ЗАПРОСЫ ДЛЯ ДЕМОНСТРАЦИИ:")
    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"  {i}. {query}")
    
    # Настройки логирования
    print("\n📝 НАСТРОЙКИ ЛОГИРОВАНИЯ:")
    print(f"  LOG_LEVEL: {LOG_LEVEL}")
    print(f"  LOG_FORMAT: {LOG_FORMAT}")
    
    print("\n" + "=" * 60)
    print("✅ Проверка конфигурации завершена!")

def main():
    """Основная функция"""
    check_configuration()

if __name__ == "__main__":
    main() 
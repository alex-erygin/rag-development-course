#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Конфигурационный файл для системы векторного поиска
"""

# Пути к директориям и файлам
KB_DIRECTORY = "KB"                                    # Директория с текстовыми файлами
CHROMA_DB_PATH = "./chroma_db"                        # Путь к базе данных ChromaDB

# Настройки локальной модели
EMBEDDING_MODEL_ENDPOINT = "http://127.0.0.1:1234/v1" # Адрес локальной модели
EMBEDDING_MODEL_ID = "text-embedding-nomic-embed-text-v1.5"  # Идентификатор модели эмбеддингов
EMBEDDING_API_KEY = "dummy"                           # API ключ (для локальной модели не важен)

# Настройки ChromaDB
COLLECTION_NAME = "automotive_knowledge"               # Название коллекции в ChromaDB

# Настройки поиска
DEFAULT_SEARCH_RESULTS = 3                            # Количество результатов по умолчанию
MAX_CONTENT_PREVIEW_LENGTH = 200                      # Максимальная длина предварительного просмотра

# Тестовые запросы для демонстрации
DEMO_QUERIES = [
    "проблемы с двигателем",
    "температура мотора", 
    "безопасность при ремонте",
    "охлаждение двигателя",
    "диагностика поломок"
]

# Настройки логирования
LOG_LEVEL = "INFO"                                    # Уровень логирования
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s" 
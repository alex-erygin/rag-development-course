#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для демонстрации техники Query Expansion в RAG-системах
Сравнивает стандартный RAG с RAG, использующим расширение запросов через LLM
"""

import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI
from query_expansion_visualization import visualize_query_expansion

# Конфигурационные параметры
ORIGINAL_CHROMA_DB_PATH = "../02 Embeddings Data Retrieval/chroma_db_embeddings"
COLLECTION_NAME = "knowledge_base_embeddings"
N_RESULTS_PER_QUERY = 5
MAX_ALTERNATIVE_QUERIES = 5
OPENAI_BASE_URL = "http://127.0.0.1:1234/v1"
OPENAI_API_KEY = "dummy"
MODEL_NAME = "google/gemma-3-12b"

def initialize_chromadb():
    """Инициализация ChromaDB и подключение к существующей коллекции"""
    try:
        client = chromadb.PersistentClient(path=ORIGINAL_CHROMA_DB_PATH)
        embedding_function = SentenceTransformerEmbeddingFunction()
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        return client, collection
    except Exception as e:
        print(f"❌ Ошибка подключения к ChromaDB: {e}")
        print("💡 Убедитесь, что запущен модуль '02 Embeddings Data Retrieval' для создания базы данных")
        return None, None

def search_documents(collection, query: str, n_results: int = N_RESULTS_PER_QUERY):
    """Поиск документов в ChromaDB по запросу"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        return results
    except Exception as e:
        print(f"❌ Ошибка поиска: {e}")
        return None

def augment_multiple_query(query, model=MODEL_NAME):
    """Генерация альтернативных запросов с помощью LLM"""
    openai_client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY
    )
    
    messages = [
        {
            "role": "system",
            "content": "Вы - полезный помощник-эксперт по финансовой отчетности. Пользователи задают вопросы об годовом отчете. "
            "Предложите до пяти дополнительных связанных вопросов, которые помогут им найти нужную информацию для предоставленного вопроса. "
            "Предлагайте только короткие вопросы без сложных предложений. Предложите разнообразные вопросы, охватывающие разные аспекты темы. "
            "Убедитесь, что это полные вопросы, связанные с исходным вопросом. "
            "Выводите по одному вопросу на строку. Не нумеруйте вопросы."
        },
        {"role": "user", "content": query}
    ]

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content
        # Разбиваем на отдельные вопросы и фильтруем пустые строки
        questions = [q.strip() for q in content.split("\n") if q.strip()]
        return questions[:MAX_ALTERNATIVE_QUERIES]
    except Exception as e:
        print(f"❌ Ошибка генерации альтернативных запросов: {e}")
        return []

def rag(query, retrieved_documents, model=MODEL_NAME):
    """Функция RAG для генерации ответа на основе найденных документов"""
    information = "\n\n".join(retrieved_documents)

    openai_client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY
    )
    
    messages = [
        {
            "role": "system",
            "content": "Вы - полезный помощник-эксперт. Ваши пользователи задают вопросы об информации, содержащейся в технических документах. "
            "Вам будет показан вопрос пользователя и соответствующая информация из документов. Ответьте на вопрос пользователя, используя только эту информацию"
        },
        {"role": "user", "content": f"Question: {query}. \n Information: {information}"}
    ]
    
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        print(f"❌ Ошибка при обращении к LLM: {e}")
        return "Не удалось получить ответ от LLM"

def deduplicate_documents(all_documents, all_metadatas):
    """Удаление дубликатов документов на основе содержимого"""
    seen_content = set()
    unique_documents = []
    unique_metadatas = []
    
    for doc, meta in zip(all_documents, all_metadatas):
        if doc not in seen_content:
            seen_content.add(doc)
            unique_documents.append(doc)
            unique_metadatas.append(meta)
    
    return unique_documents, unique_metadatas

def print_search_results(results, title="НАЙДЕННЫЕ ДОКУМЕНТЫ"):
    """Вывод результатов поиска в консоль"""
    print("=" * 60)
    print(title)
    print("=" * 60)
    
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        print(f"\n{i}. Файл: {meta['source_file']}, Чанк: {meta['chunk_index']}")
        print(f"   Содержание: {doc[:200]}{'...' if len(doc) > 200 else ''}")

def basic_rag(collection, query):
    """Стандартный RAG без расширения запросов"""
    print(f"\n📊 ЧАСТЬ 1: Стандартный RAG")
    print("=" * 60)
    print(f"❓ Запрос: {query}")
    
    # Поиск документов
    print(f"\n🔍 Поиск релевантных документов...")
    results = search_documents(collection, query, N_RESULTS_PER_QUERY)
    
    if not results:
        return None
    
    documents = results['documents'][0]
    print(f"✅ Найдено документов: {len(documents)}")
    
    # Показываем найденные документы
    print_search_results(results, "НАЙДЕННЫЕ ДОКУМЕНТЫ (Стандартный RAG)")
    
    # Генерация ответа
    print(f"\n🤖 Генерация ответа с помощью LLM...")
    response = rag(query, documents)
    
    print("\n" + "=" * 60)
    print("ОТВЕТ LLM (Стандартный RAG)")
    print("=" * 60)
    print(response)
    
    return {
        'query': query,
        'documents': documents,
        'response': response,
        'num_documents': len(documents)
    }

def expanded_rag(collection, query):
    """RAG с расширением запросов"""
    print(f"\n📈 ЧАСТЬ 2: RAG с Query Expansion")
    print("=" * 60)
    print(f"❓ Оригинальный запрос: {query}")
    
    # Генерация альтернативных запросов
    print(f"\n🔄 Генерация альтернативных запросов...")
    alternative_queries = augment_multiple_query(query)
    
    if not alternative_queries:
        print("❌ Не удалось сгенерировать альтернативные запросы")
        return None
    
    print(f"✅ Сгенерировано запросов: {len(alternative_queries)}")
    print("\n🔄 Сгенерированные запросы:")
    for i, alt_query in enumerate(alternative_queries, 1):
        print(f"   {i}. {alt_query}")
    
    # Объединяем оригинальный запрос с альтернативными
    all_queries = [query] + alternative_queries
    print(f"\n🔍 Поиск документов для {len(all_queries)} запросов...")
    
    # Поиск документов для каждого запроса
    all_documents = []
    all_metadatas = []
    
    for i, search_query in enumerate(all_queries, 1):
        print(f"   Поиск {i}/{len(all_queries)}: {search_query[:50]}{'...' if len(search_query) > 50 else ''}")
        results = search_documents(collection, search_query, N_RESULTS_PER_QUERY)
        
        if results:
            all_documents.extend(results['documents'][0])
            all_metadatas.extend(results['metadatas'][0])
    
    # Дедупликация документов
    unique_documents, unique_metadatas = deduplicate_documents(all_documents, all_metadatas)
    
    print(f"✅ Всего найдено документов: {len(all_documents)}")
    print(f"✅ Уникальных документов после дедупликации: {len(unique_documents)}")
    
    # Показываем найденные документы (первые 10 для краткости)
    mock_results = {
        'documents': [unique_documents[:10]], 
        'metadatas': [unique_metadatas[:10]]
    }
    print_search_results(mock_results, "НАЙДЕННЫЕ ДОКУМЕНТЫ (Query Expansion, первые 10)")
    
    # Генерация ответа
    print(f"\n🤖 Генерация ответа с помощью LLM на основе {len(unique_documents)} документов...")
    response = rag(query, unique_documents)
    
    print("\n" + "=" * 60)
    print("ОТВЕТ LLM (Query Expansion)")
    print("=" * 60)
    print(response)
    
    return {
        'query': query,
        'alternative_queries': alternative_queries,
        'documents': unique_documents,
        'response': response,
        'num_documents': len(unique_documents),
        'total_found': len(all_documents)
    }

def compare_results(basic_result, expanded_result):
    """Сравнение результатов двух подходов"""
    print(f"\n📋 СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    if not basic_result or not expanded_result:
        print("❌ Не удалось выполнить сравнение - один из методов не сработал")
        return
    
    print(f"📊 Стандартный RAG:")
    print(f"   • Количество документов: {basic_result['num_documents']}")
    print(f"   • Длина ответа: {len(basic_result['response'])} символов")
    
    print(f"\n📈 RAG с Query Expansion:")
    print(f"   • Альтернативных запросов: {len(expanded_result['alternative_queries'])}")
    print(f"   • Всего найдено документов: {expanded_result['total_found']}")
    print(f"   • Уникальных документов: {expanded_result['num_documents']}")
    print(f"   • Длина ответа: {len(expanded_result['response'])} символов")
    
    improvement = expanded_result['num_documents'] / basic_result['num_documents']
    print(f"\n📈 Улучшение покрытия: {improvement:.1f}x больше документов")
    
    print(f"\n💡 Выводы:")
    print(f"   • Query Expansion позволил найти в {improvement:.1f} раза больше релевантных документов")
    print(f"   • Это может привести к более полному и точному ответу")
    print(f"   • Особенно эффективно для сложных или многоаспектных вопросов")

def main():
    """Основная функция скрипта"""
    print("🚀 Демонстрация Query Expansion в RAG-системах")
    print("=" * 60)
    
    # Инициализация ChromaDB
    print("\n🗄️ Подключение к базе данных...")
    client, collection = initialize_chromadb()
    
    if not collection:
        return
    
    print("✅ Успешно подключились к ChromaDB")
    
    # Тестовый запрос
    query = "Какие планы по развитию и использованию ИИ?"
    
    # Выполнение стандартного RAG
    basic_result = basic_rag(collection, query)
    
    # Выполнение RAG с Query Expansion
    expanded_result = expanded_rag(collection, query)
    
    # Сравнение результатов
    compare_results(basic_result, expanded_result)
    
    # Создание визуализаций
    print(f"\n🎨 Создание визуализаций...")
    try:
        visualization_result = visualize_query_expansion(query)
        if visualization_result:
            print("✅ Визуализации успешно созданы!")
        else:
            print("⚠️ Не удалось создать визуализации")
    except Exception as e:
        print(f"⚠️ Ошибка при создании визуализаций: {e}")
        print("💡 Убедитесь, что установлены все необходимые библиотеки (matplotlib, umap-learn)")
    
    print(f"\n🎉 Демонстрация завершена!")
    print("💡 Попробуйте изменить запрос в коде для экспериментов с другими вопросами")

if __name__ == "__main__":
    main()

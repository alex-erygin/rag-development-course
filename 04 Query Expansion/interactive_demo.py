#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Интерактивная демонстрация Query Expansion в RAG-системах
Позволяет пользователю вводить собственные запросы и сравнивать результаты
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

def process_query(collection, query, show_details=True, create_visualization=False):
    """Обработка одного запроса с обоими методами"""
    print(f"\n{'='*80}")
    print(f"🔍 ОБРАБОТКА ЗАПРОСА: {query}")
    print(f"{'='*80}")
    
    # Стандартный RAG
    print(f"\n📊 Стандартный RAG:")
    print("-" * 40)
    
    basic_results = search_documents(collection, query, N_RESULTS_PER_QUERY)
    if not basic_results:
        print("❌ Ошибка поиска в стандартном RAG")
        return None, None
    
    basic_documents = basic_results['documents'][0]
    basic_response = rag(query, basic_documents)
    
    print(f"✅ Найдено документов: {len(basic_documents)}")
    if show_details:
        print(f"📄 Первые 3 документа:")
        for i, doc in enumerate(basic_documents[:3], 1):
            print(f"   {i}. {doc[:100]}{'...' if len(doc) > 100 else ''}")
    
    print(f"\n🤖 Ответ: {basic_response}")
    
    # RAG с Query Expansion
    print(f"\n📈 RAG с Query Expansion:")
    print("-" * 40)
    
    # Генерация альтернативных запросов
    alternative_queries = augment_multiple_query(query)
    if not alternative_queries:
        print("❌ Не удалось сгенерировать альтернативные запросы")
        return None, None
    
    print(f"🔄 Сгенерированные запросы ({len(alternative_queries)}):")
    for i, alt_query in enumerate(alternative_queries, 1):
        print(f"   {i}. {alt_query}")
    
    # Поиск для всех запросов
    all_queries = [query] + alternative_queries
    all_documents = []
    all_metadatas = []
    
    for search_query in all_queries:
        results = search_documents(collection, search_query, N_RESULTS_PER_QUERY)
        if results:
            all_documents.extend(results['documents'][0])
            all_metadatas.extend(results['metadatas'][0])
    
    # Дедупликация
    unique_documents, unique_metadatas = deduplicate_documents(all_documents, all_metadatas)
    expanded_response = rag(query, unique_documents)
    
    print(f"✅ Всего найдено: {len(all_documents)} документов")
    print(f"✅ Уникальных: {len(unique_documents)} документов")
    
    if show_details:
        print(f"📄 Первые 3 уникальных документа:")
        for i, doc in enumerate(unique_documents[:3], 1):
            print(f"   {i}. {doc[:100]}{'...' if len(doc) > 100 else ''}")
    
    print(f"\n🤖 Ответ: {expanded_response}")
    
    # Краткое сравнение
    improvement = len(unique_documents) / len(basic_documents) if len(basic_documents) > 0 else 1
    print(f"\n📊 Сравнение:")
    print(f"   • Стандартный RAG: {len(basic_documents)} документов")
    print(f"   • Query Expansion: {len(unique_documents)} документов")
    print(f"   • Улучшение: {improvement:.1f}x")
    
    return {
        'basic': {'documents': basic_documents, 'response': basic_response},
        'expanded': {'documents': unique_documents, 'response': expanded_response, 'queries': alternative_queries}
    }

def show_menu():
    """Показать меню опций"""
    print(f"\n{'='*60}")
    print("🎯 ИНТЕРАКТИВНАЯ ДЕМОНСТРАЦИЯ QUERY EXPANSION")
    print("="*60)
    print("Выберите действие:")
    print("1. Ввести свой запрос")
    print("2. Использовать примеры запросов")
    print("3. Показать информацию о базе данных")
    print("4. Выход")
    print("-" * 60)

def show_sample_queries():
    """Показать примеры запросов"""
    samples = [
        "Какова чистая прибыль Сбербанка в 2024 году?",
        "Сколько клиентов у Сбербанка?",
        "Какие технологические достижения у Сбера?",
        "Как изменился кредитный портфель?",
        "Что такое GigaChat и сколько у него пользователей?",
        "Какова рентабельность капитала банка?",
        "Какие ESG-инициативы реализует Сбер?"
    ]
    
    print(f"\n📝 ПРИМЕРЫ ЗАПРОСОВ:")
    print("-" * 40)
    for i, query in enumerate(samples, 1):
        print(f"{i}. {query}")
    
    try:
        choice = input(f"\nВыберите номер запроса (1-{len(samples)}) или нажмите Enter для возврата: ").strip()
        if choice and choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(samples):
                return samples[idx]
    except:
        pass
    
    return None

def show_database_info(collection):
    """Показать информацию о базе данных"""
    try:
        count = collection.count()
        print(f"\n📊 ИНФОРМАЦИЯ О БАЗЕ ДАННЫХ:")
        print("-" * 40)
        print(f"📁 Путь: {ORIGINAL_CHROMA_DB_PATH}")
        print(f"📚 Коллекция: {COLLECTION_NAME}")
        print(f"📄 Количество документов: {count}")
        print(f"🔍 Документов на запрос: {N_RESULTS_PER_QUERY}")
        print(f"🔄 Макс. альтернативных запросов: {MAX_ALTERNATIVE_QUERIES}")
        print(f"🤖 Модель LLM: {MODEL_NAME}")
        print(f"🌐 API сервер: {OPENAI_BASE_URL}")
    except Exception as e:
        print(f"❌ Ошибка получения информации: {e}")

def main():
    """Основная функция интерактивного режима"""
    print("🚀 Запуск интерактивной демонстрации Query Expansion...")
    
    # Инициализация
    print("\n🗄️ Подключение к базе данных...")
    client, collection = initialize_chromadb()
    
    if not collection:
        print("❌ Не удалось подключиться к базе данных. Завершение работы.")
        return
    
    print("✅ Успешно подключились к ChromaDB")
    
    while True:
        show_menu()
        
        try:
            choice = input("Ваш выбор (1-4): ").strip()
            
            if choice == "1":
                # Пользовательский запрос
                query = input("\n💬 Введите ваш запрос: ").strip()
                if query:
                    show_details = input("Показать детали поиска? (y/n, по умолчанию y): ").strip().lower()
                    show_details = show_details != 'n'
                    create_viz = input("Создать визуализацию? (y/n, по умолчанию n): ").strip().lower()
                    create_viz = create_viz == 'y'
                    
                    result = process_query(collection, query, show_details, create_viz)
                    
                    if create_viz and result:
                        print(f"\n🎨 Создание визуализаций...")
                        try:
                            visualization_result = visualize_query_expansion(query)
                            if visualization_result:
                                print("✅ Визуализации успешно созданы!")
                                print("💾 Файлы сохранены в текущей директории:")
                                print("   • query_expansion_basic.png")
                                print("   • query_expansion_expanded.png")
                            else:
                                print("⚠️ Не удалось создать визуализации")
                        except Exception as e:
                            print(f"⚠️ Ошибка при создании визуализаций: {e}")
                            print("💡 Убедитесь, что установлены все необходимые библиотеки (matplotlib, umap-learn)")
                else:
                    print("❌ Пустой запрос")
            
            elif choice == "2":
                # Примеры запросов
                query = show_sample_queries()
                if query:
                    show_details = input("Показать детали поиска? (y/n, по умолчанию y): ").strip().lower()
                    show_details = show_details != 'n'
                    create_viz = input("Создать визуализацию? (y/n, по умолчанию n): ").strip().lower()
                    create_viz = create_viz == 'y'
                    
                    result = process_query(collection, query, show_details, create_viz)
                    
                    if create_viz and result:
                        print(f"\n🎨 Создание визуализаций...")
                        try:
                            visualization_result = visualize_query_expansion(query)
                            if visualization_result:
                                print("✅ Визуализации успешно созданы!")
                                print("💾 Файлы сохранены в текущей директории:")
                                print("   • query_expansion_basic.png")
                                print("   • query_expansion_expanded.png")
                            else:
                                print("⚠️ Не удалось создать визуализации")
                        except Exception as e:
                            print(f"⚠️ Ошибка при создании визуализаций: {e}")
                            print("💡 Убедитесь, что установлены все необходимые библиотеки (matplotlib, umap-learn)")
            
            elif choice == "3":
                # Информация о БД
                show_database_info(collection)
            
            elif choice == "4":
                # Выход
                print("\n👋 До свидания!")
                break
            
            else:
                print("❌ Неверный выбор. Попробуйте снова.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Программа прервана пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Произошла ошибка: {e}")

if __name__ == "__main__":
    main()

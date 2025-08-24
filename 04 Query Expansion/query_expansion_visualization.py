#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для визуализации Query Expansion в RAG-системах
Создает визуализации для сравнения стандартного RAG и RAG с расширением запросов
"""

import os
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
import umap
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Используем backend без GUI
from openai import OpenAI

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
            include=["documents", "metadatas", "embeddings"]
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

def get_all_embeddings(collection):
    """Получение всех embeddings из коллекции"""
    results = collection.get(
        include=["embeddings", "documents", "metadatas"]
    )
    return results

def calculate_distances(query_embedding, document_embeddings):
    """Вычисление расстояний между query и document embeddings"""
    distances = []
    for doc_emb in document_embeddings:
        dist = cosine_distances([query_embedding], [doc_emb])[0][0]
        distances.append(dist)
    return distances

def create_visualization(embeddings_2d, labels, distances=None, query_text="", 
                        alternative_queries=None, visualization_type="basic", 
                        found_by_query=None):
    """Создание визуализации с помощью matplotlib"""
    plt.figure(figsize=(14, 10))
    
    # Разделяем точки по типам
    query_indices = [i for i, label in enumerate(labels) if label == "QUERY"]
    doc_indices = [i for i, label in enumerate(labels) if label == "DOC"]
    found_doc_indices = [i for i, label in enumerate(labels) if label == "FOUND_DOC"]
    
    # Для всех точек
    all_x = [point[0] for point in embeddings_2d]
    all_y = [point[1] for point in embeddings_2d]
    
    # Документы (темно-синие точки)
    if doc_indices:
        doc_x = [all_x[i] for i in doc_indices]
        doc_y = [all_y[i] for i in doc_indices]
        plt.scatter(doc_x, doc_y, c='darkblue', alpha=0.6, s=50, label='Документы', marker='o')
    
    # Найденные документы с цветовой кодировкой по запросам
    if found_doc_indices:
        found_x = [all_x[i] for i in found_doc_indices]
        found_y = [all_y[i] for i in found_doc_indices]
        
        if visualization_type == "expanded" and found_by_query:
            # Разные цвета для документов, найденных разными запросами
            colors = ['orange', 'green', 'purple', 'brown', 'pink', 'gray']
            for query_idx, doc_indices_for_query in found_by_query.items():
                if doc_indices_for_query:
                    color = colors[query_idx % len(colors)]
                    query_name = f"Запрос {query_idx + 1}" if query_idx > 0 else "Оригинальный запрос"
                    
                    x_coords = [found_x[i] for i in doc_indices_for_query if i < len(found_x)]
                    y_coords = [found_y[i] for i in doc_indices_for_query if i < len(found_y)]
                    
                    if x_coords and y_coords:
                        sizes = [80 + (1-distances[i])*120 if distances and i < len(distances) else 100 
                                for i in doc_indices_for_query if i < len(found_x)]
                        plt.scatter(x_coords, y_coords, c=color, alpha=0.8, s=sizes, 
                                  label=query_name, marker='o')
        else:
            # Стандартная визуализация (оранжевые точки)
            if distances and len(distances) >= len(found_doc_indices):
                sizes = [80 + (1-dist)*120 for dist in distances[:len(found_doc_indices)]]
            else:
                sizes = [100] * len(found_doc_indices)
            plt.scatter(found_x, found_y, c='orange', alpha=0.8, s=sizes, 
                       label='Найденные документы', marker='o')
    
    # Запрос (красная звезда)
    if query_indices:
        query_x = [all_x[i] for i in query_indices]
        query_y = [all_y[i] for i in query_indices]
        plt.scatter(query_x, query_y, c='red', alpha=1.0, s=400, label='Оригинальный запрос', 
                   marker='*', edgecolors='black', linewidth=2)
    
    # Добавляем аннотации
    if query_indices:
        for i in query_indices:
            plt.annotate('ЗАПРОС', (all_x[i], all_y[i]), 
                        xytext=(15, 15), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                        fontsize=10, ha='left', fontweight='bold')
    
    # Настройка заголовка в зависимости от типа визуализации
    if visualization_type == "basic":
        title = f'Стандартный RAG\nЗапрос: "{query_text}"'
        subtitle = f"Найдено документов: {len(found_doc_indices)}"
    else:
        title = f'RAG с Query Expansion\nОригинальный запрос: "{query_text}"'
        alt_queries_text = ""
        if alternative_queries:
            alt_queries_text = f"\nАльтернативные запросы: {len(alternative_queries)}"
        subtitle = f"Найдено уникальных документов: {len(found_doc_indices)}{alt_queries_text}"
    
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.text(0.02, 0.98, subtitle, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Сохраняем график
    filename = f'query_expansion_{visualization_type}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 График сохранен как '{filename}'")
    return filename

def visualize_basic_rag(collection, query):
    """Визуализация стандартного RAG"""
    print(f"\n📊 Создание визуализации стандартного RAG...")
    
    # Получаем embedding для запроса
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0]
    
    # Поиск документов
    results = search_documents(collection, query, N_RESULTS_PER_QUERY)
    if not results:
        return None
    
    # Получаем все embeddings из коллекции
    all_data = get_all_embeddings(collection)
    all_embeddings = all_data['embeddings']
    all_documents = all_data['documents']
    
    # Подготавливаем данные для UMAP
    embeddings_for_umap = [query_embedding] + all_embeddings
    labels = ["QUERY"] + ["DOC"] * len(all_embeddings)
    
    # Применяем UMAP
    reducer = umap.UMAP(n_components=2, random_state=42, transform_seed=42)
    embeddings_2d = reducer.fit_transform(embeddings_for_umap)
    
    # Находим индексы найденных документов
    found_docs_content = results['documents'][0]
    found_indices = []
    for found_doc in found_docs_content:
        try:
            doc_index = all_documents.index(found_doc)
            found_indices.append(doc_index)
        except ValueError:
            continue
    
    # Создаем метки для визуализации
    visual_labels = ["QUERY"] + ["DOC"] * len(all_embeddings)
    for idx in found_indices:
        if idx < len(visual_labels) - 1:
            visual_labels[idx + 1] = "FOUND_DOC"
    
    # Вычисляем расстояния
    found_embeddings = results['embeddings'][0]
    distances = calculate_distances(query_embedding, found_embeddings)
    
    # Создаем визуализацию
    filename = create_visualization(embeddings_2d, visual_labels, distances=distances, 
                                  query_text=query, visualization_type="basic")
    
    return {
        'results': results,
        'distances': distances,
        'filename': filename,
        'embeddings_2d': embeddings_2d,
        'reducer': reducer,
        'query_embedding': query_embedding,
        'all_data': all_data
    }

def visualize_expanded_rag(collection, query, basic_data=None):
    """Визуализация RAG с Query Expansion"""
    print(f"\n📈 Создание визуализации RAG с Query Expansion...")
    
    # Используем данные из базового поиска, если они есть
    if basic_data:
        query_embedding = basic_data['query_embedding']
        all_data = basic_data['all_data']
        reducer = basic_data['reducer']
        embeddings_2d = basic_data['embeddings_2d']
    else:
        # Получаем embedding для запроса
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])[0]
        
        # Получаем все embeddings из коллекции
        all_data = get_all_embeddings(collection)
        
        # Подготавливаем данные для UMAP
        all_embeddings = all_data['embeddings']
        embeddings_for_umap = [query_embedding] + all_embeddings
        
        # Применяем UMAP
        reducer = umap.UMAP(n_components=2, random_state=42, transform_seed=42)
        embeddings_2d = reducer.fit_transform(embeddings_for_umap)
    
    # Генерируем альтернативные запросы
    alternative_queries = augment_multiple_query(query)
    if not alternative_queries:
        print("❌ Не удалось сгенерировать альтернативные запросы")
        return None
    
    print(f"✅ Сгенерировано {len(alternative_queries)} альтернативных запросов")
    
    # Поиск документов для всех запросов
    all_queries = [query] + alternative_queries
    all_documents_found = []
    all_metadatas_found = []
    found_by_query = {}  # Отслеживаем, какие документы найдены каким запросом
    
    for i, search_query in enumerate(all_queries):
        results = search_documents(collection, search_query, N_RESULTS_PER_QUERY)
        if results:
            query_docs = results['documents'][0]
            query_metas = results['metadatas'][0]
            
            # Запоминаем индексы документов для этого запроса
            start_idx = len(all_documents_found)
            all_documents_found.extend(query_docs)
            all_metadatas_found.extend(query_metas)
            end_idx = len(all_documents_found)
            
            found_by_query[i] = list(range(start_idx, end_idx))
    
    # Дедупликация документов
    unique_documents, unique_metadatas = deduplicate_documents(all_documents_found, all_metadatas_found)
    
    # Находим индексы уникальных документов в общем массиве
    all_documents = all_data['documents']
    found_indices = []
    for unique_doc in unique_documents:
        try:
            doc_index = all_documents.index(unique_doc)
            found_indices.append(doc_index)
        except ValueError:
            continue
    
    # Создаем метки для визуализации
    visual_labels = ["QUERY"] + ["DOC"] * len(all_documents)
    for idx in found_indices:
        if idx < len(visual_labels) - 1:
            visual_labels[idx + 1] = "FOUND_DOC"
    
    # Получаем embeddings для уникальных документов и вычисляем расстояния
    unique_embeddings = []
    for unique_doc in unique_documents:
        try:
            doc_index = all_documents.index(unique_doc)
            unique_embeddings.append(all_data['embeddings'][doc_index])
        except ValueError:
            continue
    
    distances = calculate_distances(query_embedding, unique_embeddings)
    
    # Создаем визуализацию
    filename = create_visualization(embeddings_2d, visual_labels, distances=distances, 
                                  query_text=query, alternative_queries=alternative_queries,
                                  visualization_type="expanded", found_by_query=found_by_query)
    
    return {
        'unique_documents': unique_documents,
        'alternative_queries': alternative_queries,
        'distances': distances,
        'filename': filename,
        'total_found': len(all_documents_found),
        'unique_found': len(unique_documents)
    }

def compare_visualizations(basic_result, expanded_result, query):
    """Создание сравнительной визуализации"""
    print(f"\n📋 Создание сравнительной визуализации...")
    
    if not basic_result or not expanded_result:
        print("❌ Не удалось создать сравнительную визуализацию")
        return
    
    # Создаем комбинированную визуализацию
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Здесь можно добавить код для создания side-by-side сравнения
    # Пока что просто выводим статистику
    
    plt.suptitle(f'Сравнение методов RAG\nЗапрос: "{query}"', fontsize=16, fontweight='bold')
    
    # Левая панель - статистика базового RAG
    ax1.text(0.1, 0.9, "📊 Стандартный RAG", fontsize=14, fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.1, 0.8, f"• Найдено документов: {len(basic_result['results']['documents'][0])}", 
             fontsize=12, transform=ax1.transAxes)
    ax1.text(0.1, 0.7, f"• Среднее расстояние: {np.mean(basic_result['distances']):.3f}", 
             fontsize=12, transform=ax1.transAxes)
    ax1.text(0.1, 0.6, f"• Мин. расстояние: {min(basic_result['distances']):.3f}", 
             fontsize=12, transform=ax1.transAxes)
    ax1.text(0.1, 0.5, f"• Макс. расстояние: {max(basic_result['distances']):.3f}", 
             fontsize=12, transform=ax1.transAxes)
    
    # Правая панель - статистика расширенного RAG
    ax2.text(0.1, 0.9, "📈 RAG с Query Expansion", fontsize=14, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.1, 0.8, f"• Альтернативных запросов: {len(expanded_result['alternative_queries'])}", 
             fontsize=12, transform=ax2.transAxes)
    ax2.text(0.1, 0.7, f"• Всего найдено: {expanded_result['total_found']}", 
             fontsize=12, transform=ax2.transAxes)
    ax2.text(0.1, 0.6, f"• Уникальных документов: {expanded_result['unique_found']}", 
             fontsize=12, transform=ax2.transAxes)
    ax2.text(0.1, 0.5, f"• Среднее расстояние: {np.mean(expanded_result['distances']):.3f}", 
             fontsize=12, transform=ax2.transAxes)
    
    # Убираем оси
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Сохраняем
    filename = 'query_expansion_comparison.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Сравнительная визуализация сохранена как '{filename}'")
    return filename

def visualize_query_expansion(query):
    """Основная функция для создания визуализаций Query Expansion"""
    print("🚀 Создание визуализаций Query Expansion")
    print("=" * 60)
    
    # Инициализация ChromaDB
    print("\n🗄️ Подключение к базе данных...")
    client, collection = initialize_chromadb()
    
    if not collection:
        print("❌ Не удалось подключиться к базе данных")
        return None
    
    print("✅ Успешно подключились к ChromaDB")
    
    # Создаем визуализацию стандартного RAG
    basic_result = visualize_basic_rag(collection, query)
    
    # Создаем визуализацию расширенного RAG
    expanded_result = visualize_expanded_rag(collection, query, basic_result)
    
    # Выводим статистику
    if basic_result and expanded_result:
        print(f"\n📊 РЕЗУЛЬТАТЫ ВИЗУАЛИЗАЦИИ")
        print("=" * 60)
        print(f"🔍 Запрос: {query}")
        print(f"📊 Стандартный RAG: {len(basic_result['results']['documents'][0])} документов")
        print(f"📈 Query Expansion: {expanded_result['unique_found']} уникальных документов")
        improvement = expanded_result['unique_found'] / len(basic_result['results']['documents'][0])
        print(f"📈 Улучшение: {improvement:.1f}x больше документов")
        print(f"💾 Созданы файлы:")
        print(f"   • {basic_result['filename']}")
        print(f"   • {expanded_result['filename']}")
    
    return {
        'basic': basic_result,
        'expanded': expanded_result,
        'query': query
    }

if __name__ == "__main__":
    # Тестовый запрос
    query = "Какие планы по развитию и использованию ИИ?"
    visualize_query_expansion(query)

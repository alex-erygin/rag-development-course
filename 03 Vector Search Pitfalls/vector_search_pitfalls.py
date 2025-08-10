#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для демонстрации визуализации расстояний между embeddings в векторной БД
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

# Конфигурационные параметры
CHROMA_DB_PATH = "../02 Embeddings Data Retrieval/chroma_db_embeddings"
COLLECTION_NAME = "knowledge_base_embeddings"

def load_text_files(directory: str) -> list:
    """Загрузка текстовых файлов из указанной директории"""
    import glob
    text_files = glob.glob(os.path.join(directory, "*.md"))
    documents = []
    
    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    documents.append({
                        'content': content,
                        'filename': os.path.basename(file_path)
                    })
        except Exception as e:
            print(f"Ошибка чтения файла {file_path}: {e}")
    
    return documents

def split_documents_into_chunks(documents: list, chunk_size: int = 256) -> list:
    """Разбиение документов на чанки заданного размера"""
    from langchain.text_splitter import SentenceTransformersTokenTextSplitter
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=0, 
        tokens_per_chunk=chunk_size
    )
    
    chunks = []
    for doc in documents:
        # Разбиваем документ на чанки
        doc_chunks = token_splitter.split_text(doc['content'])
        for i, chunk in enumerate(doc_chunks):
            chunks.append({
                'content': chunk,
                'source_file': doc['filename'],
                'chunk_index': i
            })
    
    return chunks

def add_chunks_to_chromadb(collection, chunks: list):
    """Добавление чанков в ChromaDB"""
    # Подготавливаем данные для добавления
    documents = [chunk['content'] for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            'source_file': chunk['source_file'],
            'chunk_index': chunk['chunk_index']
        } 
        for chunk in chunks
    ]
    
    # Добавляем чанки в коллекцию
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    
    print(f"Добавлено {len(chunks)} чанков в базу данных")

def connect_to_chromadb():
    """Подключение к существующей ChromaDB"""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        # Проверяем, есть ли в коллекции документы
        if collection.count() == 0:
            print("Коллекция пуста. Загружаем документы...")
            # Загружаем документы из папки KB
            kb_directory = os.path.join("..", "02 Embeddings Data Retrieval", "KB")
            documents = load_text_files(kb_directory)
            print(f"Загружено {len(documents)} документов")
            
            # Разбиваем на чанки
            chunks = split_documents_into_chunks(documents, 256)
            print(f"Получено {len(chunks)} чанков")
            
            # Добавляем в коллекцию
            if chunks:
                add_chunks_to_chromadb(collection, chunks)
            else:
                print("Нет документов для добавления в коллекцию")
        else:
            print(f"Коллекция содержит {collection.count()} документов")
        return client, collection
    except Exception as e:
        print(f"Коллекция {COLLECTION_NAME} не найдена. Создаем новую...")
        # Создаем новую коллекцию
        embedding_function = SentenceTransformerEmbeddingFunction()
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        
        # Загружаем документы из папки KB
        kb_directory = os.path.join("..", "02 Embeddings Data Retrieval", "KB")
        documents = load_text_files(kb_directory)
        print(f"Загружено {len(documents)} документов")
        
        # Разбиваем на чанки
        chunks = split_documents_into_chunks(documents, 256)
        print(f"Получено {len(chunks)} чанков")
        
        # Добавляем в коллекцию
        if chunks:
            add_chunks_to_chromadb(collection, chunks)
        else:
            print("Нет документов для добавления в коллекцию")
        
        return client, collection

def get_all_embeddings(collection):
    """Получение всех embeddings из коллекции"""
    # Получаем все документы и их embeddings
    results = collection.get(
        include=["embeddings", "documents", "metadatas"]
    )
    return results

def calculate_distances(query_embedding, document_embeddings):
    """Вычисление расстояний между query и document embeddings"""
    distances = []
    for doc_emb in document_embeddings:
        # Вычисляем косинусное расстояние
        dist = cosine_distances([query_embedding], [doc_emb])[0][0]
        distances.append(dist)
    return distances

def create_matplotlib_visualization(embeddings_2d, labels, distances=None, query_text="", visualization_type="all"):
    """Создание красивой визуализации с помощью matplotlib"""
    plt.figure(figsize=(12, 8))
    
    # Разделяем точки по типам
    query_indices = [i for i, label in enumerate(labels) if label == "QUERY"]
    doc_indices = [i for i, label in enumerate(labels) if label == "DOC"]
    found_doc_indices = [i for i, label in enumerate(labels) if label == "FOUND_DOC"]
    
    # Для всех точек
    all_x = [point[0] for point in embeddings_2d]
    all_y = [point[1] for point in embeddings_2d]
    
    # Создаем scatter plot
    # Документы (темно-синие точки)
    if doc_indices:
        doc_x = [all_x[i] for i in doc_indices]
        doc_y = [all_y[i] for i in doc_indices]
        plt.scatter(doc_x, doc_y, c='darkblue', alpha=0.6, s=50, label='Документы', marker='o')
    
    # Найденные документы (оранжевые кружочки)
    if found_doc_indices:
        found_x = [all_x[i] for i in found_doc_indices]
        found_y = [all_y[i] for i in found_doc_indices]
        # Размер точек в зависимости от расстояния (обратная зависимость) - уменьшенные размеры
        if distances and len(distances) >= len(found_doc_indices):
            sizes = [30 + (1-dist)*100 for dist in distances[:len(found_doc_indices)]]
        else:
            sizes = [100] * len(found_doc_indices)
        plt.scatter(found_x, found_y, c='orange', alpha=0.8, s=sizes, label='Найденные документы', marker='o')
    
    # Запрос (красная точка)
    if query_indices:
        query_x = [all_x[i] for i in query_indices]
        query_y = [all_y[i] for i in query_indices]
        plt.scatter(query_x, query_y, c='red', alpha=1.0, s=300, label='Запрос', marker='*', edgecolors='black', linewidth=2)
    
    # Добавляем подписи к точкам запроса и найденным документам
    if query_indices:
        for i in query_indices:
            plt.annotate('ЗАПРОС', (all_x[i], all_y[i]), 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                        fontsize=10, ha='left', fontweight='bold')
    
    if found_doc_indices and distances:
        for i, (idx, dist) in enumerate(zip(found_doc_indices, distances)):
            if i < len(distances):
                plt.annotate(f'Документ {i+1}\nРасстояние: {dist:.3f}', 
                           (all_x[idx], all_y[idx]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                           fontsize=9, ha='left')
    
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.title(f'Визуализация Embeddings (UMAP)\nЗапрос: "{query_text}"', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Сохраняем график с уникальным именем
    filename = f'vector_search_visualization_{visualization_type}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 График сохранен как '{filename}'")

def print_distance_info(distances, labels):
    """Вывод информации о расстояниях"""
    print("\n" + "=" * 60)
    print("РАССТОЯНИЯ МЕЖДУ ЗАПРОСОМ И ДОКУМЕНТАМИ")
    print("=" * 60)
    print(f"🎯 ЗАПРОС: Расстояния до найденных документов")
    print("-" * 50)
    
    # Сортируем документы по расстоянию (от ближайшего к дальнему)
    sorted_distances = sorted(enumerate(distances), key=lambda x: x[1])
    
    for i, (doc_idx, dist) in enumerate(sorted_distances):
        # Используем эмодзи для визуализации близости
        if dist < 0.3:
            proximity = "🔥 ОЧЕНЬ БЛИЗКО"
        elif dist < 0.5:
            proximity = "👍 БЛИЗКО"
        elif dist < 0.7:
            proximity = "👌 СРЕДНЕ"
        else:
            proximity = "❄️ ДАЛЕКО"
            
        print(f"📄 Документ {doc_idx}: Расстояние = {dist:.4f} ({proximity})")
    
    # Выводим статистику с иконками
    if distances:
        print(f"\n📈 СТАТИСТИКА РАССТОЯНИЙ:")
        print(f"  📊 Минимальное: {min(distances):.4f}")
        print(f"  📈 Максимальное: {max(distances):.4f}")
        print(f"  ⚖️  Среднее: {np.mean(distances):.4f}")
        print(f"  📐 Стандартное отклонение: {np.std(distances):.4f}")

def search_and_visualize(collection, query: str, n_results: int = 5):
    """Поиск и визуализация результатов"""
    print(f"🔍 Поиск по запросу: '{query}'")
    
    # Получаем embedding для запроса
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0]
    
    # Поиск ближайших документов
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "embeddings"]
    )
    
    # Получаем все embeddings из коллекции для визуализации
    all_data = get_all_embeddings(collection)
    
    # Подготавливаем данные для UMAP
    all_embeddings = all_data['embeddings']
    all_documents = all_data['documents']
    
    # Добавляем query embedding к данным
    embeddings_for_umap = [query_embedding] + all_embeddings
    labels = ["QUERY"] + ["DOC"] * len(all_embeddings)
    
    # Применяем UMAP для понижения размерности до 2D
    print("📐 Применение UMAP для визуализации...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings_for_umap)
    
    # Создаем matplotlib визуализацию для всех точек
    print("🎨 Создание красивой визуализации всех embeddings...")
    create_matplotlib_visualization(embeddings_2d, labels, query_text=query, visualization_type="all")
    
    # Вычисляем расстояния для найденных документов
    found_embeddings = results['embeddings'][0]
    distances = calculate_distances(query_embedding, found_embeddings)
    
    # Создаем визуализацию для ВСЕХ документов с выделением найденных
    # Используем все embeddings и помечаем найденные документы
    all_labels = ["QUERY"] + ["DOC"] * len(all_embeddings)  # Все документы как обычные
    
    # Находим индексы найденных документов в общем массиве
    found_indices = []
    found_docs_content = results['documents'][0]
    for found_doc in found_docs_content:
        try:
            doc_index = all_documents.index(found_doc)
            found_indices.append(doc_index)
        except ValueError:
            # Если документ не найден, пропускаем его
            continue
    
    # Создаем специальные метки для визуализации найденных документов
    found_labels = ["QUERY"] + ["DOC"] * len(all_embeddings)
    # Помечаем найденные документы как FOUND_DOC
    for idx in found_indices:
        if idx < len(found_labels) - 1:  # -1 потому что первый элемент - QUERY
            found_labels[idx + 1] = "FOUND_DOC"  # +1 потому что первый элемент - QUERY
    
    # Создаем matplotlib визуализацию для всех документов с выделением найденных
    print("🎯 Создание визуализации всех документов...")
    create_matplotlib_visualization(embeddings_2d, found_labels, distances=distances, query_text=query, visualization_type="found")
    
    # Выводим информацию о расстояниях (используем оригинальные found_labels для вывода)
    found_labels_for_print = ["QUERY"] + ["FOUND_DOC"] * len(found_embeddings)
    print_distance_info(distances, found_labels_for_print)
    
    return results

def main():
    """Основная функция скрипта"""
    print("📊 Демонстрация визуализации расстояний между embeddings")
    
    try:
        # Подключение к ChromaDB
        print("\n🗄️ Подключение к существующей ChromaDB...")
        client, collection = connect_to_chromadb()
        print(f"✅ Подключено к коллекции: {COLLECTION_NAME}")
        
        # Формирование запроса от пользователя
        query = "Какова чистая прибыль Сбербанка в 2024 году?"
        print(f"\n❓ Запрос пользователя: {query}")
        
        # Поиск и визуализация
        results = search_and_visualize(collection, query, n_results=5)
        
        # Вывод найденных документов
        print("\n" + "=" * 60)
        print("НАЙДЕННЫЕ ДОКУМЕНТЫ")
        print("=" * 60)
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
            print(f"\n{i}. Файл: {meta['source_file']}, Чанк: {meta['chunk_index']}")
            print(f"   Содержание: {doc}")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("Убедитесь, что ChromaDB существует и доступен")

if __name__ == "__main__":
    main()

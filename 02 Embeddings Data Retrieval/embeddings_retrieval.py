#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для демонстрации работы с embeddings и RAG (Retrieval-Augmented Generation)
"""

import os
import glob
from pypdf import PdfReader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import openai
from openai import OpenAI

# Конфигурационные параметры
import os
KB_DIRECTORY = os.path.join(os.path.dirname(__file__), "KB")
CHROMA_DB_PATH = "./chroma_db_embeddings"
COLLECTION_NAME = "knowledge_base_embeddings"
CHUNK_SIZE_TOKENS = 256

def load_text_files(directory: str) -> list:
    """Загрузка текстовых файлов из указанной директории"""
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

def split_documents_into_chunks(documents: list, chunk_size: int = CHUNK_SIZE_TOKENS) -> list:
    """Разбиение документов на чанки заданного размера"""
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

def initialize_chromadb():
    """Инициализация ChromaDB и создание коллекции"""
    # Создаем клиент ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Удаляем коллекцию, если она существует
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    
    # Создаем функцию эмбеддинга
    embedding_function = SentenceTransformerEmbeddingFunction()
    
    # Создаем коллекцию
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
    
    return client, collection

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

def search_documents(collection, query: str, n_results: int = 5):
    """Поиск документов в ChromaDB по запросу"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    
    return results

def print_search_results(results):
    """Вывод результатов поиска в консоль"""
    print("=" * 60)
    print("НАЙДЕННЫЕ ДОКУМЕНТЫ")
    print("=" * 60)
    
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        print(f"\n{i}. Файл: {meta['source_file']}, Чанк: {meta['chunk_index']}")
        print(f"   Содержание: {doc}")

def rag(query, retrieved_documents, model="google/gemma-3-12b"):
    """Функция RAG для генерации ответа на основе найденных документов"""
    information = "\n\n".join(retrieved_documents)

    # Создаем клиент OpenAI с локальной конфигурацией
    openai_client = OpenAI(
        base_url="http://127.0.0.1:1234/v1",
        api_key="dummy"
    )
    
    messages = [
        {
            "role": "system",
            "content": "Вы - полезный помощник-эксперт. Ваши пользователи задают вопросы об информации, содержащейся в технических документах."
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
        print(f"Ошибка при обращении к LLM: {e}")
        return "Не удалось получить ответ от LLM"

def main():
    """Основная функция скрипта"""
    print("🚀 Начинаем демонстрацию работы с embeddings и RAG")
    
    # 1. Загрузка текстовых файлов
    print("\n📚 Загрузка текстовых файлов...")
    documents = load_text_files(KB_DIRECTORY)
    print(f"Загружено {len(documents)} документов")
    
    # 2. Разбиение на чанки
    print("\n✂️ Разбиение документов на чанки...")
    chunks = split_documents_into_chunks(documents, CHUNK_SIZE_TOKENS)
    print(f"Получено {len(chunks)} чанков")
    
    # 3. Инициализация ChromaDB
    print("\n🗄️ Инициализация ChromaDB...")
    client, collection = initialize_chromadb()
    
    # 4. Добавление чанков в ChromaDB
    print("\n➕ Добавление чанков в базу данных...")
    add_chunks_to_chromadb(collection, chunks)
    
    # 5. Формирование запроса от пользователя
    query = "Какова чистая прибыль Сбербанка в 2024 году?"
    print(f"\n❓ Запрос пользователя: {query}")
    
    # 6. Поиск документов по запросу
    print("\n🔍 Поиск релевантных документов...")
    results = search_documents(collection, query, n_results=5)
    
    # 7. Вывод первых 5 найденных чанков
    print_search_results(results)
    
    # 8. Формирование сообщения для LLM и вывод ответа
    print("\n🤖 Формирование ответа с помощью LLM...")
    retrieved_documents = results['documents'][0]
    response = rag(query, retrieved_documents)
    
    print("\n" + "=" * 60)
    print("ОТВЕТ ОТ LLM")
    print("=" * 60)
    print(response)

if __name__ == "__main__":
    main()

import os
import glob
import chromadb
from chromadb.config import Settings
import openai
import re
from typing import List, Dict, Tuple
import json
from config import *

# Импорт всех конфигурационных переменных из config.py

class VectorSearchDemo:
    def __init__(self):
        """Инициализация системы векторного поиска"""
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = None
        self.openai_client = openai.OpenAI(
            base_url=EMBEDDING_MODEL_ENDPOINT,
            api_key=EMBEDDING_API_KEY
        )
        
    def get_text_files(self) -> List[str]:
        """Получение списка всех текстовых файлов в директории KB"""
        pattern = os.path.join(KB_DIRECTORY, "*.txt")
        return glob.glob(pattern)
    
    def read_text_file(self, file_path: str) -> str:
        """Чтение содержимого текстового файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            print(f"Ошибка чтения файла {file_path}: {e}")
            return ""
    
    def get_embedding(self, text: str) -> List[float]:
        """Получение эмбеддинга для текста через локальную модель"""
        try:
            response = self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL_ID,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Ошибка получения эмбеддинга: {e}")
            return []
    
    def create_collection(self):
        """Создание коллекции в ChromaDB"""
        try:
            # Удаляем существующую коллекцию если она есть
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except:
                pass
            
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "База знаний по автомобильной тематике"}
            )
            print(f"Коллекция '{COLLECTION_NAME}' создана успешно")
        except Exception as e:
            print(f"Ошибка создания коллекции: {e}")
    
    def index_documents(self):
        """Индексация всех текстовых файлов в векторную базу"""
        if not self.collection:
            self.create_collection()
        
        text_files = self.get_text_files()
        if not text_files:
            print(f"Текстовые файлы не найдены в директории {KB_DIRECTORY}")
            return
        
        print(f"Найдено {len(text_files)} текстовых файлов для индексации")
        
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        for file_path in text_files:
            content = self.read_text_file(file_path)
            if not content:
                continue
            
            # Получаем эмбеддинг
            embedding = self.get_embedding(content)
            if not embedding:
                continue
            
            # Извлекаем имя файла без расширения
            filename = os.path.splitext(os.path.basename(file_path))[0]
            
            documents.append(content)
            embeddings.append(embedding)
            metadatas.append({
                "filename": filename,
                "file_path": file_path,
                "content_length": len(content)
            })
            ids.append(f"doc_{len(ids)}")
            
            print(f"Индексирован файл: {filename}")
        
        if documents:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Успешно проиндексировано {len(documents)} документов")
        else:
            print("Нет документов для индексации")
    
    def semantic_search(self, query: str, n_results: int = DEFAULT_SEARCH_RESULTS) -> List[Dict]:
        """Семантический поиск по векторной базе"""
        if not self.collection:
            print("Коллекция не инициализирована")
            return []
        
        try:
            # Получаем эмбеддинг для запроса
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []
            
            # Выполняем поиск
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Формируем результат
            search_results = []
            for i in range(len(results['documents'][0])):
                search_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'filename': results['metadatas'][0][i]['filename']
                })
            
            return search_results
            
        except Exception as e:
            print(f"Ошибка семантического поиска: {e}")
            return []
    
    def keyword_search(self, query: str, n_results: int = DEFAULT_SEARCH_RESULTS) -> List[Dict]:
        """Поиск по ключевым словам (точное совпадение)"""
        if not self.collection:
            print("Коллекция не инициализирована")
            return []
        
        try:
            # Получаем все документы из коллекции
            all_docs = self.collection.get()
            
            # Ищем документы, содержащие ключевые слова
            query_words = query.lower().split()
            search_results = []
            
            for i, doc in enumerate(all_docs['documents']):
                doc_lower = doc.lower()
                matches = 0
                
                for word in query_words:
                    if word in doc_lower:
                        matches += 1
                
                if matches > 0:
                    # Вычисляем релевантность как процент совпадающих слов
                    relevance = matches / len(query_words)
                    
                    search_results.append({
                        'content': doc,
                        'metadata': all_docs['metadatas'][i],
                        'relevance': relevance,
                        'filename': all_docs['metadatas'][i]['filename'],
                        'matches': matches
                    })
            
            # Сортируем по релевантности и берем топ результаты
            search_results.sort(key=lambda x: x['relevance'], reverse=True)
            return search_results[:n_results]
            
        except Exception as e:
            print(f"Ошибка поиска по ключевым словам: {e}")
            return []
    
    def print_search_results(self, results: List[Dict], search_type: str):
        """Вывод результатов поиска в консоль"""
        print(f"\n{'='*60}")
        print(f"РЕЗУЛЬТАТЫ {search_type.upper()} ПОИСКА")
        print(f"{'='*60}")
        
        if not results:
            print("Результаты не найдены")
            return
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Файл: {result['filename']}")
            if 'distance' in result:
                print(f"   Расстояние: {result['distance']:.4f}")
            if 'relevance' in result:
                print(f"   Релевантность: {result['relevance']:.2f} ({result['matches']} совпадений)")
            
            # Показываем первые символы содержимого согласно настройке
            content_preview = result['content'][:MAX_CONTENT_PREVIEW_LENGTH] + "..." if len(result['content']) > MAX_CONTENT_PREVIEW_LENGTH else result['content']
            print(f"   Содержание: {content_preview}")
    
    def demo_search_comparison(self):
        """Демонстрация сравнения семантического и ключевого поиска"""
        test_queries = DEMO_QUERIES
        
        print("\n" + "="*80)
        print("ДЕМОНСТРАЦИЯ СРАВНЕНИЯ СЕМАНТИЧЕСКОГО И КЛЮЧЕВОГО ПОИСКА")
        print("="*80)
        
        for query in test_queries:
            print(f"\n\n🔍 ЗАПРОС: '{query}'")
            print("-" * 50)
            
            # Семантический поиск
            semantic_results = self.semantic_search(query, n_results=2)
            self.print_search_results(semantic_results, "СЕМАНТИЧЕСКОГО")
            
            # Ключевой поиск
            keyword_results = self.keyword_search(query, n_results=2)
            self.print_search_results(keyword_results, "КЛЮЧЕВОГО")
            
            print("\n" + "-" * 50)

def main():
    """Основная функция для запуска демонстрации"""
    print("🚀 Инициализация системы векторного поиска...")
    
    # Создаем экземпляр системы поиска
    search_system = VectorSearchDemo()
    
    # Индексируем документы
    print("\n📚 Индексация документов...")
    search_system.index_documents()
    
    # Запускаем демонстрацию
    print("\n🎯 Запуск демонстрации поиска...")
    search_system.demo_search_comparison()
    
    print("\n✅ Демонстрация завершена!")

if __name__ == "__main__":
    main() 
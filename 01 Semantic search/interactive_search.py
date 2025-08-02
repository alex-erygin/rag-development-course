#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from vector_search_demo import VectorSearchDemo
from config import *

def interactive_search():
    """Интерактивный режим поиска"""
    print("🔍 ИНТЕРАКТИВНЫЙ ПОИСК ПО БАЗЕ ЗНАНИЙ")
    print("=" * 50)
    print("Доступные команды:")
    print("  'semantic <запрос>' - семантический поиск")
    print("  'keyword <запрос>' - поиск по ключевым словам")
    print("  'compare <запрос>' - сравнение обоих методов")
    print("  'quit' или 'exit' - выход")
    print("=" * 50)
    
    # Инициализируем систему поиска
    search_system = VectorSearchDemo()
    
    # Проверяем, есть ли уже индексированные документы
    try:
        search_system.collection = search_system.client.get_collection(COLLECTION_NAME)
        print("✅ Найдена существующая база знаний")
    except:
        print("📚 Индексируем документы...")
        search_system.index_documents()
    
    while True:
        try:
            user_input = input("\n🎯 Введите команду: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 До свидания!")
                break
            
            if not user_input:
                continue
            
            # Парсим команду
            parts = user_input.split(' ', 1)
            if len(parts) < 2:
                print("❌ Неверный формат команды. Пример: 'semantic проблемы с двигателем'")
                continue
            
            command, query = parts
            
            if command.lower() == 'semantic':
                print(f"\n🔍 Семантический поиск: '{query}'")
                results = search_system.semantic_search(query, n_results=DEFAULT_SEARCH_RESULTS)
                search_system.print_search_results(results, "СЕМАНТИЧЕСКОГО")
                
            elif command.lower() == 'keyword':
                print(f"\n🔍 Поиск по ключевым словам: '{query}'")
                results = search_system.keyword_search(query, n_results=DEFAULT_SEARCH_RESULTS)
                search_system.print_search_results(results, "КЛЮЧЕВОГО")
                
            elif command.lower() == 'compare':
                print(f"\n🔍 Сравнение методов поиска: '{query}'")
                
                print("\n📊 СЕМАНТИЧЕСКИЙ ПОИСК:")
                semantic_results = search_system.semantic_search(query, n_results=2)
                search_system.print_search_results(semantic_results, "СЕМАНТИЧЕСКОГО")
                
                print("\n📊 ПОИСК ПО КЛЮЧЕВЫМ СЛОВАМ:")
                keyword_results = search_system.keyword_search(query, n_results=2)
                search_system.print_search_results(keyword_results, "КЛЮЧЕВОГО")
                
            else:
                print("❌ Неизвестная команда. Доступные команды: semantic, keyword, compare")
                
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")

def main():
    """Основная функция"""
    print("🚀 Запуск интерактивного поиска...")
    interactive_search()

if __name__ == "__main__":
    main() 
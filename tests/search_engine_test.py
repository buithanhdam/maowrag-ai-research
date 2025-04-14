from src.search_engine import SearchEngineManager, SearchEngineType
if __name__ == "__main__":
    try:
        # Thay thế API key thực tế
        manager = SearchEngineManager()
        tavily_class = manager.get_search_engine_implementation(SearchEngineType.TAVILY)
        tavily_client = tavily_class()
        
        # Ví dụ search
        search_results = tavily_client.search(query="who is leonel messi?", max_results=5)
        print(search_results["data"]["results"])
        
        # # Ví dụ qna_search
        # qna_results = tavily_client.qna_search("who is leonel messi?")
        # print(qna_results)
        
        # # Ví dụ extract
        # extract_results = tavily_client.extract("https://vi.wikipedia.org/wiki/Lionel_Messi")
        # print(extract_results["data"]["results"])
        
    except Exception as e:
        print(f"Error in example: {str(e)}")
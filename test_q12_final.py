#!/usr/bin/env python3

import sys
sys.path.append('./src')
from conversation import ConversationalAgenticRetrieval
import asyncio

async def test_bot():
    print('🤖 Testing Bot with Q1-2 Query')
    print('='*50)
    
    rag = ConversationalAgenticRetrieval('./data/', './chroma_db')
    
    # Test the exact query
    query = 'solve Q 1-2 from midterm exam'
    print(f'Query: {query}')
    print()
    
    response = await rag.ainvoke(query, 'test_session')
    print(f'Response:')
    print(response)
    print()
    print('='*50)
    
    # Test with different phrasing
    query2 = 'Q. 1-2 floating-point arithmetic 4-4+η'
    print(f'Query 2: {query2}')
    print()
    
    response2 = await rag.ainvoke(query2, 'test_session2')
    print(f'Response 2:')
    print(response2)

if __name__ == "__main__":
    asyncio.run(test_bot()) 
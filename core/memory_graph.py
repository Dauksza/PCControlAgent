"""Neo4j Memory Graph for context, memories, and files"""
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import os

class MemoryGraph:
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None
    
    def connect(self):
        if not self.driver:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self.driver
    
    def add_memory(self, content: str, context: Dict[str, Any]):
        with self.driver.session() as session:
            result = session.run(
                "CREATE (m:Memory {content: $content, timestamp: timestamp(), context: $context}) RETURN m",
                content=content, context=str(context)
            )
            return result.single()[0]
    
    def search_memories(self, query: str, limit: int = 10):
        with self.driver.session() as session:
            result = session.run(
                "MATCH (m:Memory) WHERE m.content CONTAINS $query RETURN m LIMIT $limit",
                query=query, limit=limit
            )
            return [record["m"] for record in result]
    
    def close(self):
        if self.driver:
            self.driver.close()

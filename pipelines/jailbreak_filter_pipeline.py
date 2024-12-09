"""
title: Jailbreak Detection Filter Pipeline
author: Your Name
date: 2024-03-21
version: 1.0
license: MIT
description: A pipeline for detecting jailbreak attempts using Arize AI's dataset embeddings guardrails
requirements: sentence-transformers, numpy, scikit-learn
"""

from typing import List, Optional
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0
        threshold: float = 0.85
        model_name: str = "all-MiniLM-L6-v2"

    def __init__(self):
        self.type = "filter"
        self.name = "Jailbreak Detection Filter"
        self.valves = self.Valves()
        self.model = None
        self.jailbreak_embeddings = None

    async def on_startup(self):
        print(f"on_startup: {__name__}")
        
        # Load the sentence transformer model
        self.model = SentenceTransformer(self.valves.model_name)
        
        # Load known jailbreak patterns from CSV
        csv_path = os.path.join(os.path.dirname(__file__), "jailbreak_filter_pipeline", "jailbreak_prompts_2023_05_07.csv")
        self.jailbreak_patterns = []

        with open(csv_path, 'r', encoding='utf-8') as file:
            import csv
            reader = csv.DictReader(file)
            for row in reader:
                if row and row.get('prompt'):
                    self.jailbreak_patterns.append(row['prompt'])
        
        # Generate embeddings for known jailbreak patterns
        self.jailbreak_embeddings = self.model.encode(self.jailbreak_patterns)

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")

    async def on_valves_updated(self):
        pass

    def check_similarity(self, text: str) -> float:
        # Generate embedding for input text
        text_embedding = self.model.encode([text])[0]
        
        # Calculate cosine similarity with known jailbreak patterns
        similarities = cosine_similarity([text_embedding], self.jailbreak_embeddings)[0]
        
        # Return highest similarity score
        return np.max(similarities)

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet: {__name__}")

        messages = body.get("messages", [])
        
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                similarity_score = self.check_similarity(content)
                
                if similarity_score > self.valves.threshold:
                    raise Exception(f"Potential jailbreak attempt detected")
                break

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        return body

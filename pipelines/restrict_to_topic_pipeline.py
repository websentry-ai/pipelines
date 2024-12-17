"""
title: Restrict to Topic Filter
author: unbound
date: 2024-12-13
version: 1.0
license: MIT
description: A pipeline for filtering out messages based on topic.
requirements: transformers
"""

from typing import List, Optional
from pydantic import BaseModel
from transformers import pipeline

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0
        threshold: float = 0.5

    def __init__(self):
        self.type = "filter"
        self.name = "Restrict to Topic Filter"
        
        self.valves = self.Valves()
        
        self.classifier = None

    async def on_startup(self):
        print(f"on_startup: {__name__}")
        
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        print("Topic classifier model loaded successfully.")

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")

    async def on_valves_updated(self):
        pass

    async def inlet(self, request: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet: {__name__}")

        body = request.body
        config = request.config

        valid_topics = config.get("valid_topics", [])
        invalid_topics = config.get("invalid_topics", [])

        if not valid_topics and not invalid_topics:
            return request

        message = body.get("text", "")

        if message and message.strip():
            matches_valid_topic = True
            matches_invalid_topic = False

            if valid_topics:
                result = self.classifier(
                    message,
                    candidate_labels=valid_topics,
                    multi_label=True
                )
                matches_valid_topic = any(score > self.valves.threshold for score in result['scores'])

            if invalid_topics:
                result = self.classifier(
                    message,
                    candidate_labels=invalid_topics,
                    multi_label=True
                )
                matches_invalid_topic = any(score > self.valves.threshold for score in result['scores'])

            if (valid_topics and not matches_valid_topic and not invalid_topics) or matches_invalid_topic:
                raise Exception("Message contains invalid topics or is not related to any valid topics.")

        return request

    async def outlet(self, request: dict, user: Optional[dict] = None) -> dict:
        return request


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

        topics = config.get("restrict_to_topic", [])

        if not topics:
            return request

        message = body.get("text", "")

        if message:
            if not message.strip():
                return request

            result = self.classifier(
                message,
                candidate_labels=topics,
                multi_label=True
            )

            matches_topic = any(score > self.valves.threshold for score in result['scores'])

            if not matches_topic:
                raise Exception(
                    "Message is not related to any of the configured topics."
                )

        return request

    async def outlet(self, request: dict, user: Optional[dict] = None) -> dict:
        return request


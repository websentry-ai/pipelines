"""
title: NSFW Filter Pipeline
author: Your Name
date: 2024-10-18
version: 1.0
license: MIT
description: A pipeline for filtering out NSFW messages using the 'michellejieli/NSFW_text_classifier' model.
requirements: transformers
"""

from typing import List, Optional
from pydantic import BaseModel
from transformers import pipeline as hf_pipeline
import re
import os

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]  # Connect to all pipelines by default
        priority: int = 0
        threshold: float = 0.9
        validation_method: str = "sentence"

    def __init__(self):
        self.type = "filter"
        self.name = "NSFW Filter Pipeline"
        self.valves = self.Valves()
        self.nsfw_model = None

    async def on_startup(self):
        print(f"on_startup: {__name__}")

        # Load the NSFW classifier model
        self.nsfw_model = hf_pipeline(
            "text-classification",
            model="michellejieli/NSFW_text_classifier",
            tokenizer="michellejieli/NSFW_text_classifier",
        )

        print("NSFW classifier model loaded successfully.")

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")

    async def on_valves_updated(self):
        pass

    async def inlet(self, request: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet: {__name__}")

        body = request.body
        messages = body.get("messages", [])

        # Manually extract the last user message
        user_message = None
        for message in reversed(messages):
            if message.get("role") == "user":
                user_message = message
                break

        if user_message:
            content = user_message.get("content", "")

            if not content.strip():
                raise Exception("Input message cannot be empty.")

            # Validate the content for NSFW text
            validation_result = self.validate(content)

            if not validation_result["is_safe"]:
                raise Exception("NSFW content detected in the input message.")

        return body

    async def outlet(self, request: dict, user: Optional[dict] = None) -> dict:
        return request

    def validate(self, value: str) -> dict:
        validation_method = self.valves.validation_method.lower()
        if validation_method not in ["sentence", "full"]:
            raise ValueError("validation_method must be 'sentence' or 'full'.")

        if validation_method == "sentence":
            return self.validate_each_sentence(value)
        else:
            return self.validate_full_text(value)

    def validate_each_sentence(self, value: str) -> dict:
        # Use regular expressions to split the text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', value)

        nsfw_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                if self.is_nsfw(sentence):
                    nsfw_sentences.append(sentence)

        if nsfw_sentences:
            return {
                "is_safe": False,
                "nsfw_sentences": nsfw_sentences,
            }

        return {"is_safe": True}

    def validate_full_text(self, value: str) -> dict:
        if self.is_nsfw(value):
            return {"is_safe": False}
        return {"is_safe": True}

    def is_nsfw(self, text: str) -> bool:
        threshold = self.valves.threshold

        results = self.nsfw_model(text)
        if not results:
            return False

        result = results[0]
        label = result.get("label")
        score = result.get("score")

        if label == "NSFW" and score > threshold:
            return True

        return False

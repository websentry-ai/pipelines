"""
title: Bias Check Pipeline
author: unbound
date: 2024-12-13
version: 1.0
license: MIT
description: A pipeline for checking if the content is biased.
requirements: transformers
"""

from typing import List, Optional
from pydantic import BaseModel
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
import os

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0
        threshold: float = 0.8

    def __init__(self):
        self.type = "filter"
        self.name = "Bias Check Filter"
        
        self.valves = self.Valves()
        self.bias_model = None

    async def on_startup(self):
        print(f"on_startup: {__name__}")
        
        model_name = "d4data/bias-detection-model"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
        
        self.bias_model = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer
        )
        
        print("Bias detection model loaded successfully.")

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")

    async def on_valves_updated(self):
        pass

    async def inlet(self, request: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet: {__name__}")

        body = request.body
        messages = body.get("messages", [])

        user_message = None
        for message in reversed(messages):
            if message.get("role") == "user":
                user_message = message
                break

        if user_message:
            content = user_message.get("content", "")
            
            if not content.strip():
                raise Exception("Input message cannot be empty.")

            result = self.bias_model(content)[0]
            bias_score = result["score"]
            is_biased = result["label"] == "Biased"

            if is_biased and bias_score > self.valves.threshold:
                raise Exception("Potentially biased content detected")

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        messages = body.get("messages", [])
        
        assistant_message = None
        for message in reversed(messages):
            if message.get("role") == "assistant":
                assistant_message = message
                break

        if assistant_message:
            content = assistant_message.get("content", "")
            result = self.bias_model(content)[0]
            bias_score = result["score"]
            is_biased = result["label"] == "LABEL_1"

            if is_biased and bias_score > self.valves.threshold:
                warning_prefix = "The response may contain biased content."
                assistant_message["content"] = warning_prefix + content
                body["messages"] = messages

        return body

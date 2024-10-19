"""
title: NSFW Filter Pipeline
author: Your Name
date: 2024-10-18
version: 1.0
license: MIT
description: A pipeline for filtering out NSFW messages using the 'michellejieli/NSFW_text_classifier' model.
requirements: transformers, nltk
"""

from typing import List, Optional
from pydantic import BaseModel
from utils.pipelines.main import get_last_user_message
from transformers import pipeline as hf_pipeline
import nltk
from sentence_splitter import SentenceSplitter
import os

class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        pipelines: List[str] = ["*"]  # Connect to all pipelines by default

        # Assign a priority level to the filter pipeline.
        priority: int = 0

        # Threshold for NSFW detection (between 0 and 1)
        threshold: float = 0.8

        # Validation method: 'sentence' or 'full'
        validation_method: str = "sentence"

    def __init__(self):
        self.type = "filter"
        self.name = "NSFW Filter Pipeline"

        # Initialize valves
        self.valves = self.Valves()

        # Placeholder for the NSFW classifier model
        self.nsfw_model = None

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup: {__name__}")

        # Initialize the sentence splitter
        self.splitter = SentenceSplitter(language='en')

        # Load the NSFW classifier model
        self.nsfw_model = hf_pipeline(
            "text-classification",
            model="michellejieli/NSFW_text_classifier",
            tokenizer="michellejieli/NSFW_text_classifier",
        )

        print("NSFW classifier model loaded successfully.")

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown: {__name__}")

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # This filter is applied to the user input before it is sent to the LLM.
        print(f"inlet: {__name__}")

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
                # Empty message, raise an exception
                raise Exception("Input message cannot be empty.")

            # Validate the content for NSFW text
            validation_result = self.validate(content)

            if not validation_result["is_safe"]:
                # NSFW content detected, raise an exception to block the message
                raise Exception("NSFW content detected in the input message.")

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # No changes needed in the outlet; pass the body as is.
        return body

    def validate(self, value: str) -> dict:
        """Validate the input text for NSFW content."""
        validation_method = self.valves.validation_method.lower()
        if validation_method not in ["sentence", "full"]:
            raise ValueError("validation_method must be 'sentence' or 'full'.")

        if validation_method == "sentence":
            return self.validate_each_sentence(value)
        else:
            return self.validate_full_text(value)

    def validate_each_sentence(self, value: str) -> dict:
        """Validate each sentence in the text."""
        # Split the text into sentences using sentence-splitter
        sentences = self.splitter.split(text=value)

        nsfw_sentences = []
        for sentence in sentences:
            if self.is_nsfw(sentence):
                nsfw_sentences.append(sentence)

        if nsfw_sentences:
            return {
                "is_safe": False,
                "nsfw_sentences": nsfw_sentences,
            }

        return {"is_safe": True}

    def validate_full_text(self, value: str) -> dict:
        """Validate the full text."""
        if self.is_nsfw(value):
            return {"is_safe": False}
        return {"is_safe": True}

    def is_nsfw(self, text: str) -> bool:
        """Determine if the text is NSFW."""
        threshold = self.valves.threshold

        results = self.nsfw_model(text)
        if not results:
            return False

        # The model returns a list of results
        result = results[0]
        label = result.get("label")
        score = result.get("score")

        if label == "NSFW" and score > threshold:
            return True

        return False

"""
title: Hallucination Detection Filter Pipeline
author: Your Name
date: 2024-10-18
version: 1.0
license: MIT
description: A pipeline for detecting hallucinations in assistant responses using the HHEM-2.1-Open model.
requirements: transformers
"""

from typing import List, Optional
from pydantic import BaseModel
from utils.pipelines.main import get_last_user_message, get_last_assistant_message
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        pipelines: List[str] = ["*"]  # Connect to all pipelines by default

        # Assign a priority level to the filter pipeline.
        priority: int = 0

    def __init__(self):
        self.type = "filter"
        self.name = "Hallucination Detection Filter"

        # Initialize valves
        self.valves = self.Valves()

        # Placeholders for the model and tokenizer
        self.hhem_model = None
        self.hhem_tokenizer = None

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup: {__name__}")

        # Load the HHEM-2.1-Open model and tokenizer
        self.hhem_model = AutoModelForSequenceClassification.from_pretrained(
            'vectara/hallucination_evaluation_model', trust_remote_code=True)
        self.hhem_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

        print("HHEM-2.1-Open model loaded successfully.")

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown: {__name__}")

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    async def inlet(self, request: dict, user: Optional[dict] = None) -> dict:
        # No changes needed in the inlet; pass the body as is.
        return request

    async def outlet(self, request: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet: {__name__}")

        body = request.body
        messages = body.get("messages", [])

        user_message = None
        assistant_message = None

        for message in reversed(messages):
            if not assistant_message and message.get("role") == "assistant":
                assistant_message = message
            elif not user_message and message.get("role") == "user":
                user_message = message
            if assistant_message and user_message:
                break

        if assistant_message and user_message:
            premise = user_message.get("content", "")
            hypothesis = assistant_message.get("content", "")

            # Prepare the input for the model
            prompt = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
            input_text = prompt.format(text1=premise, text2=hypothesis)
            inputs = self.hhem_tokenizer(input_text, return_tensors="pt")

            # Get the model's prediction
            outputs = self.hhem_model(**inputs)
            scores = outputs.logits.softmax(dim=1)
            # Assuming label 1 is 'consistent', label 0 is 'hallucinated'
            hallucination_score = scores[0][0].item()  # Score for 'hallucinated'

            # Debug output
            print(f"Hallucination score: {hallucination_score}")

            # Determine if the response is hallucinated
            threshold = 0.95  # Adjust this threshold as needed
            if hallucination_score > threshold:
                # The response is likely hallucinated
                # alert_message = "**Warning:** The following response may contain hallucinations.\n\n" + hypothesis
                # Inside your outlet method, where you construct the alert_message
                alert_message = "🚨🚨 **WARNING:** The following response may contain hallucinations. 🚨🚨\n\n" + hypothesis


                # Update the assistant's message
                for message in reversed(messages):
                    if message.get("role") == "assistant":
                        message["content"] = alert_message
                        break

                # Update the body with modified messages
                body = {**body, "messages": messages}

        return body
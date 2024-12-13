"""
title: Logging Pipeline
author: Your Name
date: 2023-11-12
version: 1.0
license: MIT
description: A filter pipeline that logs prompts and responses to a GraphQL API.
requirements: requests
"""

from typing import List, Optional
import os
import requests
import time
import uuid
from pydantic import BaseModel

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]  # Apply to all pipelines
        priority: int = 0
        LOG_URL: str = os.getenv("LOG_URL", "http://localhost:8000/graphql/")
        API_KEY: str = os.getenv("API_KEY", "YOUR_SECRET_API_KEY")
        AI_MODEL_NAME: str = os.getenv("AI_MODEL_NAME", "gpt-4o")
        FUNCTIONALITY: str = os.getenv("FUNCTIONALITY", "chat")
        APPLICATION_NAME: str = os.getenv("APPLICATION_NAME", "chat-ui")

    def __init__(self):
        self.type = "filter"
        self.name = "Logging Pipeline"
        self.valves = self.Valves()
        pass

    async def on_startup(self):
        print(f"Logging Pipeline started.")

    async def on_shutdown(self):
        print(f"Logging Pipeline stopped.")

    async def on_valves_updated(self):
        # Handle any updates to the valves
        print(f"Valves updated: {self.valves}")

    def get_last_user_message(self, messages):
        for message in reversed(messages):
            if message["role"] == "user":
                return message
        return None

    def get_last_assistant_message(self, messages):
        for message in reversed(messages):
            if message["role"] == "assistant":
                return message
        return None

    def get_system_prompt(self, messages):
        for message in messages:
            if message["role"] == "system":
                return message
        return None

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # Ensure chat_id is present
        if "chat_id" not in body:
            unique_id = f"SYSTEM MESSAGE {uuid.uuid4()}"
            body["chat_id"] = unique_id
            print(f"chat_id was missing, set to: {unique_id}")

        # Additional checks or modifications can be added here
        return body

    async def outlet(self, request: dict, user: Optional[dict] = None) -> dict:
        try:
            body = request.body
            messages = body.get("messages", [])
            model = body.get("model", self.valves.AI_MODEL_NAME)
            functionality = self.valves.FUNCTIONALITY

            # Extract the latest user, assistant, and system messages
            user_message = self.get_last_user_message(messages)
            assistant_message = self.get_last_assistant_message(messages)
            system_message = self.get_system_prompt(messages)

            user_prompt = user_message.get("content", "") if user_message else ""
            assistant_prompt = assistant_message.get("content", "") if assistant_message else ""
            system_prompt = system_message.get("content", "") if system_message else ""

            # Timestamps and duration
            request_timestamp = body.get("request_timestamp")
            if not request_timestamp:
                request_timestamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

            response_timestamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

            # Calculate total duration in milliseconds
            total_duration = (
                int(time.time() * 1000)
                - int(time.mktime(time.strptime(request_timestamp, '%Y-%m-%dT%H:%M:%SZ')) * 1000)
            )

            # Prepare variables for GraphQL mutation
            variables = {
                "applicationName": self.valves.APPLICATION_NAME,
                "aiModelName": model,
                "userPrompt": user_prompt,
                "functionality": functionality,
                "systemPrompt": system_prompt,
                "assistantPrompt": assistant_prompt
            }

            mutation_payload = {
                "query": """
                    mutation CreatePrompt(
                        $applicationName: String!,
                        $aiModelName: String!,
                        $userPrompt: String!,
                        $functionality: String,
                        $systemPrompt: String,
                        $assistantPrompt: String
                    ) {
                        createPrompt(
                            applicationName: $applicationName,
                            aiModelName: $aiModelName,
                            userPrompt: $userPrompt,
                            functionality: $functionality,
                            systemPrompt: $systemPrompt,
                            assistantPrompt: $assistantPrompt
                        ) {
                            prompt {
                                id
                                functionality
                            }
                        }
                    }
                """,
                "variables": variables
            }

            # Set up headers with API key
            headers = {
                'Content-Type': 'application/json',
                'X-API-KEY': self.valves.API_KEY
            }

            print(f"Logging the prompt to {self.valves.LOG_URL}")
            print(f"Payload: {mutation_payload}")
            print(f"Headers: {headers}")
            print(f"self.valves.LOG_URL: {self.valves.LOG_URL}")

            # Make the POST request to the GraphQL API
            response = requests.post(
                self.valves.LOG_URL,
                json=mutation_payload,
                headers=headers,
                timeout=3
            )
            response.raise_for_status()
            print(f"Logged the prompt to {self.valves.LOG_URL}")
        except Exception as e:
            print(f"[Logging Pipeline] Error: {e}")

        # Return the body unmodified
        return body
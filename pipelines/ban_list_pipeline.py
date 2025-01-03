"""
title: Ban List Pipeline
author: unbound
date: 2024-12-13
version: 1.0
license: MIT
description: A pipeline for filtering out banned words or similar variations.
requirements: fuzzysearch
"""

from typing import List, Optional
from pydantic import BaseModel
from fuzzysearch import find_near_matches

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]

        priority: int = 0
        
        max_l_dist: int = 1
        
        banned_words: List[str] = [
            "inappropriate",
            "offensive",
        ]

    def __init__(self):
        self.type = "filter"
        self.name = "Ban List Filter"
        self.valves = self.Valves()

    async def on_startup(self):
        print(f"on_startup: {__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")

    async def on_valves_updated(self):
        pass

    async def inlet(self, request: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet: {__name__}")

        body = request.body
        config = request.config

        banned_words = config.get("ban_list", [])
        if not banned_words:
            return body

        if 'text' in body:
            message = body.get("text", "")
        else:
            messages = body.get("messages", [])

            # Manually extract the last user message
            user_message = None
            for message in reversed(messages):
                if message.get("role") == "user":
                    user_message = message
                    break
            
            message = user_message.get("content", "")

        if message:
            if not message.strip():
                raise Exception("Input message cannot be empty.")

            matches = self.find_banned_words(message, banned_words)
            if matches:
                banned_words_found = [match["word"] for match in matches]
                raise Exception(f"Message contains banned words or similar variations: {', '.join(banned_words_found)}")

        return body

    def find_banned_words(self, text: str, banned_words: List[str]) -> List[dict]:
        """
        Find banned words in the text using fuzzy matching.
        
        Args:
            text (str): The input text to check
            banned_words (List[str]): List of banned words to check against
            
        Returns:
            List[dict]: List of matches with their details
        """
        matches = []
        text_lower = text.lower()
        
        for banned_word in banned_words:
            banned_word_lower = banned_word.lower()
            fuzzy_matches = find_near_matches(
                banned_word_lower,
                text_lower,
                max_l_dist=self.valves.max_l_dist
            )
            
            for match in fuzzy_matches:
                matches.append({
                    "word": banned_word,
                    "matched_text": text[match.start:match.end],
                    "start": match.start,
                    "end": match.end,
                    "distance": match.dist
                })
                
        return matches

    async def outlet(self, request: dict, user: Optional[dict] = None) -> dict:
        # No modifications needed for outgoing messages
        return request

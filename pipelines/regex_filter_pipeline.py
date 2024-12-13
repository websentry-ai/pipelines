"""
title: Regex Filter Pipeline
author: unbound
date: 2024-12-13
version: 1.0
license: MIT
description: A pipeline for filtering out messages based on regex patterns.
"""

from typing import List, Optional
from pydantic import BaseModel
import re

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0
        patterns: List[str] = []
        
        case_sensitive: bool = False

    def __init__(self):
        self.type = "filter"
        self.name = "Regex Filter Pipeline"
        self.valves = self.Valves()
        self.compiled_patterns = []

    async def on_startup(self):
        print(f"on_startup: {__name__}")
        self._compile_patterns()

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")

    async def on_valves_updated(self):
        self._compile_patterns()

    def _compile_patterns(self):
        self.compiled_patterns = []
        flags = 0 if self.valves.case_sensitive else re.IGNORECASE
        
        for pattern in self.valves.patterns:
            try:
                processed_pattern = pattern.encode().decode('unicode-escape')
                self.compiled_patterns.append(re.compile(processed_pattern, flags))
            except re.error as e:
                print(f"Invalid regex pattern '{pattern}': {e}")

    async def inlet(self, request: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet: {__name__}")
        body = request.body
        config = request.config

        regex_filters = config.get("regex_filters", [])
        if not regex_filters:
            return body

        self.valves.patterns = regex_filters
        self._compile_patterns()

        messages = body.get("messages", [])

        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                for pattern in self.compiled_patterns:
                    if pattern.search(content):
                        raise Exception(f"Message matches blocked pattern: {pattern.pattern}")

                break

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        return body

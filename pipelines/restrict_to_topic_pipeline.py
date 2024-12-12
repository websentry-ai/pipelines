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

    async def inlet(self, response: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet: {__name__}")

        topics = getattr(response, "restrict_to_topic", [])

        if not topics:
            return response

        user_message = response.body.text or None

        if user_message:
            if not user_message.strip():
                return response

            result = self.classifier(
                user_message,
                candidate_labels=topics,
                multi_label=True
            )

            matches_topic = any(score > self.valves.threshold for score in result['scores'])

            if not matches_topic:
                raise Exception(
                    "Message is not related to any of the configured topics."
                )

        return response

    async def outlet(self, response: dict, user: Optional[dict] = None) -> dict:
        return response

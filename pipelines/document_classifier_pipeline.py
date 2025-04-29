"""
title: Document Classifier Pipeline
author: unbound
date: 2024-12-13
version: 1.0
license: MIT
description: A pipeline for classifying documents using VertexAI.
requirements: requests,pyjwt,cryptography
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import requests
import json
import time
import jwt
import datetime
import os
from transformers import AutoTokenizer
class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0
        
        # VertexAI configuration
        project_id: str = "your-project-id"
        location: str = "us-central1"
        model_id: str = "text-bison"
        
        # Categories for classification
        categories: List[str] = [
            "Financial Statements",
            "Board Meeting Documents",
            "Customer Communication Documents"
        ]

        explanation_categories: Dict[str, str] = {
            "Financial Statements": "Internal reports containing detailed financial data (balance sheets, income/cash flow statements, forecasts) and management analysis not yet public; often feature extensive tables and numerical data.",
            "Board Meeting Documents": "Formal records of board discussions, decisions, and action items covering company strategy, governance, and high-level oversight; typically structured as minutes with specific agenda points.",
            "Customer Communication Documents": "Sensitive correspondence with clients/partners (e.g., proposals, contracts, issue reports, strategic discussions) containing specific business, operational, or commercial details."
        }

        default_config: Dict[str, Any] = {
            "document_categories": [
                "Financial Statements",
                "Board Meeting Documents",
                "Customer Communication Documents"
            ]
        }


    def __init__(self):
        self.type = "filter"
        self.name = "Document Classifier"
        self.valves = self.Valves()
        
        # System prompt template (kept in code, not in valves)
        self.system_prompt = """
        You are a helpful assistant that categorizes finance document prompts into specific categories. 
        You should only output the JSON result without any other text or characters such as thinking or reasoning criterias.
        Determine the document type of the following user prompt by selecting the most relevant category from this list:
        {categories}

        Below is the explanation of each category:
        {explanation_categories}

        Please follow these instructions carefully:
        - Analyze the content and intent of the document based on the prompt.
        - Identify which single category best represents the document type.
        - Do not make assumptions beyond the information explicitly provided.
        - Do not include any explanations or reasoning—output only the JSON.
        - Ensure the output is always in the exact JSON format as described below.
        - Return only the JSON block without any additional characters, symbols, or formatting.
        - Make sure the output can be loaded using json.loads function in python.
        - Make sure the output is a valid JSON object not any other text with a coding block.
        - Do not include any thinking or reasoning in the output other than the final JSON result.

        For any query, **do not include reasoning or explanations**. 
        Only provide the answer in the form requested below, without any additional context or steps. 
        If the answer is not immediately available, respond with "Unable to provide an answer" or a similar concise response.

        **YOU MUST NOT**:
        1. Engage in any reasoning process.
        2. Provide any justification for your responses.
        3. Use any form of thinking or intermediate calculations in your answer.

        Return your result strictly in the following JSON format:
        {{
        "category": "[SELECTED_CATEGORY]"
        }}

        If the document cannot be classified into one of the provided categories, output:
        {{
        "category": "Unknown Document Type"
        }}
        """

    async def on_startup(self):
        print(f"on_startup: {__name__}")
        # Check if required environment variables are set
        required_env_vars = [
            "GOOGLE_CLIENT_EMAIL", 
            "GOOGLE_PRIVATE_KEY", 
            "GOOGLE_PRIVATE_KEY_ID",
            "GOOGLE_PROJECT_ID",
            "GOOGLE_REGION",
            "GOOGLE_ENDPOINT_ID"
        ]
        missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
        if missing_vars:
            print(f"Warning: Missing required environment variables: {', '.join(missing_vars)}")

        model_name = "TheFinAI/Fino1-8B"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        categories_string = "\n".join([f"- {category}" for category in self.valves.categories])
        explanation_categories_string = "\n".join([f"- {category}: {self.valves.explanation_categories[category]}" for category in self.valves.categories]) 
        self.system_prompt = self.system_prompt.format(categories=categories_string, explanation_categories=explanation_categories_string)
        print("system prompt count tokens", self.count_tokens(self.system_prompt))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")

    async def on_valves_updated(self):
        pass

    def get_access_token(self) -> str:
        """Get access token for VertexAI API using service account credentials from environment variables"""
        try:
            # Get service account info from environment variables
            client_email = os.environ.get("GOOGLE_CLIENT_EMAIL")
            private_key = os.environ.get("GOOGLE_PRIVATE_KEY")
            private_key_id = os.environ.get("GOOGLE_PRIVATE_KEY_ID")
            
            if not client_email or not private_key or not private_key_id:
                raise Exception("Required environment variables GOOGLE_CLIENT_EMAIL, GOOGLE_PRIVATE_KEY, and GOOGLE_PRIVATE_KEY_ID must be set")
            
            # If private key is base64 encoded or escaped, handle it
            if "\\n" in private_key:
                private_key = private_key.replace("\\n", "\n")
                
            # Create JWT token
            scope = 'https://www.googleapis.com/auth/cloud-platform'
            iat = datetime.datetime.utcnow()
            exp = iat + datetime.timedelta(minutes=60)
            
            payload = {
                'iss': client_email,
                'sub': client_email,
                'aud': 'https://oauth2.googleapis.com/token',
                'iat': iat,
                'exp': exp,
                'scope': scope
            }
            
            # Sign the JWT with the private key
            signed_jwt = jwt.encode(
                payload,
                private_key,
                algorithm='RS256',
                headers={
                    'kid': private_key_id
                }
            )
            
            # Exchange JWT for access token
            token_url = 'https://oauth2.googleapis.com/token'
            token_data = {
                'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
                'assertion': signed_jwt
            }
            
            response = requests.post(
                token_url,
                data=token_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            response.raise_for_status()
            token_info = response.json()
            
            return token_info['access_token']
        except Exception as e:
            raise Exception(f"Failed to get access token: {str(e)}")

    def classify_document(self, text: str) -> Dict[str, Any]:
        """Classify document using VertexAI Prediction API"""
        try:
            access_token = self.get_access_token()
            
            # Format the system prompt with categories
            system_prompt = self.system_prompt

            region = os.environ.get("GOOGLE_REGION")
            project_id = os.environ.get("GOOGLE_PROJECT_ID")
            endpoint_id = os.environ.get("GOOGLE_ENDPOINT_ID")
            
            # Create the API endpoint URL
            url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/endpoints/{endpoint_id}/chat/completions"
            # Prepare the request
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }

            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.1,
                "max_tokens": 100,
                # "stream": True
            }
            
            # Make the API request
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                try:
                    # Extract the category from the response
                    prediction = result["choices"][0]["message"]["content"]
                    print("prediction", prediction)
                    if isinstance(prediction, str):
                        # Try to parse the JSON response
                        if '</think>' in prediction:
                            import re
                            prediction = prediction.split('</think>')[1].lstrip('\n')
                            prediction = re.sub(r'```json\s*|\s*```', '', prediction).strip()

                        if '## Final Response' in prediction and prediction.endswith('}'):
                            prediction = prediction.split('## Final Response')[1].lstrip("\n")
                            prediction = prediction.strip()
                        category_data = json.loads(prediction)
                        return category_data
                    
                except json.JSONDecodeError:
                    return {"category": "Failed to parse classification result"}
            
            return {"category": "Unknown Document Type"}
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"category": f"Error: {str(e)}"}

    async def inlet(self, request: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet: {__name__}")
        
        body = request.body
        config = request.config

        if not config:
            config = self.valves.default_config
        
        message = None
        if 'text' in body:
            message = body.get("text", "")
        elif 'messages' in body:
            messages = body.get("messages", [])

            # Extract the last user message
            user_message = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg
                    break
            
            if user_message:
                message = user_message.get("content", "")
        
        if message:
            if not message.strip():
                raise Exception("Input message cannot be empty.")
            
            # Classify the document
            classification_result = self.classify_document(message)
            
            print(f"Document classified as: {classification_result.get('category', 'Unknown')}")
            classification_result = classification_result.get('category', 'Unknown')
            if classification_result in config.get('document_categories',[]):
                raise Exception(f'The Document you uploaded or the prompt you chose falls under {classification_result} category. Please choose a different document or prompt.')
        
        return body

    async def outlet(self, request: dict, user: Optional[dict] = None) -> dict:
        # No modifications needed for outgoing messages
        return request
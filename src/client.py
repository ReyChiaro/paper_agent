import os
from openai import OpenAI, AsyncOpenAI


client = None
aclient = None


def get_client():
    global client
    if not client:
        client = OpenAI(
            api_key=os.environ.get("API_KEY"),
            base_url=os.environ.get("BASE_URL"),
        )
    return client


def get_aclient():
    global aclient
    if not aclient:
        aclient = AsyncOpenAI(
            api_key=os.environ.get("API_KEY"),
            base_url=os.environ.get("BASE_URL"),
        )
    return aclient
    

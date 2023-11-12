import os

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

class OpenAIAPI():
    """Object that will contain all API calls to OpenAI
    """
    def __init__(self):
        """Initialize the OpenAI Client
        """
        load_dotenv(find_dotenv())
        os.environ['OPENAI_API_KEY'] = os.environ.get('API_KEY')
        self.client = OpenAI()
        
    def request(
        self,
        messages
    ):
        """Make a request to OpenAI API

        Args:
            messages (list): List of messages and memory of the language model of this conversation

        Returns:
            response (ChatGeneration): Full OpenAI API Response
            response.choices[0].message.content (str): Generated answer
        """
        response = self.client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=messages,
            temperature=0.0, # deterministic outputs
            max_tokens=10    # 10 tokens as we only want yes or no
        )
        return response, response.choices[0].message.content
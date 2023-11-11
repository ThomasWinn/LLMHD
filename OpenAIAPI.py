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
        
        print(self.client.models.list())
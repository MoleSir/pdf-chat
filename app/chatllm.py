from openai import OpenAI
import tomli


class ChatLLM:
    def __init__(self, config_path: str = './config/config.toml'):
        with open(config_path, 'rb') as file: 
            config = tomli.load(file)

        self.config = config['llm']
        self.client = OpenAI(
            api_key=self.config['api_key'],
            base_url=self.config['base_url'],
        )
    
    def chat(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            messages=[
                {'role': 'user', 'content': prompt},
            ],
            model=self.config['chat_model'],
        )

        return response.choices[0].message.content
    

class ChatMemoryLLM:
    def __init__(self, config_path: str = './config/config.toml', system_prompt: str = "You are a helpful assistant."):
        with open(config_path, 'rb') as file: 
            config = tomli.load(file)

        self.config = config['llm']
        self.client = OpenAI(
            api_key=self.config['api_key'],
            base_url=self.config['base_url'],
        )

        self.system_prompt = system_prompt
        self.history = [{'role': 'system', 'content': self.system_prompt}]

    def chat(self, prompt: str) -> str:
        self.history.append({'role': 'user', 'content': prompt})
        response = self.client.chat.completions.create(
            messages=self.history,
            model=self.config['chat_model'],
        )
        content = response.choices[0].message.content
        self.history.append({'role': 'assistant', 'content': content})
        return content
    
    def reset(self):
        self.history = [{'role': 'system', 'content': self.system_prompt}]

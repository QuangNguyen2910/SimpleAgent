from typing import List
from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
import random
import requests

class LLM:
    """
    A wrapper class for language models, supporting both the OpenAI API via langchain
    """
    # Truyền tham số hàm invoke
    def __init__(self, api_key: str, base_url: List[str], model: str, top_p: float = None, temperature: float = None, max_tokens: int = 24768, add_stop_token: List[str] = None):
        """
        Initializes the LLM instance.

        Args:
            api_key (str): API key for the LLM service.
            base_url (List[str]): List of Base URLs for the LLM service.
            model (str): Model name or identifier.
            temperature (float): Sampling temperature for the model.
            max_tokens (int): Maximum number of tokens to generate.
            add_stop_token (List[str]): List of stop tokens to use in generation.
        """

        # Initialize openai_server client via Langchain's OpenAI-compatible wrapper
        self.add_stop_token = ["---\n", "STOP_HERE"]
        self.base_url = base_url
        self.model = model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        except:
            print(f"Cannot load tokenizer for model {model} since it is not a local path or a valid Huggingface model name. Please ensure the model is accessible.")
        if add_stop_token:
            self.add_stop_token.extend(add_stop_token)
        self.llms = []
        for url in base_url:
            # Create a dictionary of arguments that are always present
            kwargs = {
                "api_key": api_key,
                "base_url": url,
                "model": model,
                "max_tokens": max_tokens,
                "stop": self.add_stop_token
            }

            # Conditionally add arguments if they are not None
            if temperature is not None:
                kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p

            # Unpack the dictionary to create the ChatOpenAI instance
            self.llms.append(ChatOpenAI(**kwargs))

        # print(f"Openai server model '{model}' initialized.")

    def invoke(self, prompt: str) -> str:
        """
        Invokes the LLM with a given prompt and returns the text response.
        Handles both openai_server API calls and local model inference.
        """
        # For openai_server, the temperature is set at initialization
        if len(self.llms) == 1:
            picked_llm = self.llms[0]
        else:
            picked_llm = random.choice(self.llms)
        response = picked_llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def bind_tools(self, tools):
        """
        Bind tools to the LLM instance.
        """
        if len(self.llms) == 1:
            picked_llm = self.llms[0]
        else:
            picked_llm = random.choice(self.llms)
        return picked_llm.bind_tools(tools)
    
    def stream(self, prompt: str):
        """
        Stream the response from the LLM based on the provided prompt.
        This method is a placeholder for actual streaming logic.
        """
        if len(self.llms) == 1:
            picked_llm = self.llms[0]
        else:
            picked_llm = random.choice(self.llms)
        return picked_llm.stream(prompt)
    
    def with_structured_output(self, output_schema, **kwargs):
        """
        Configure the LLM to use structured output.
        Accepts and passes along any additional keyword arguments.
        """
        # Pass the schema and all other kwargs to the real method
        if len(self.llms) == 1:
            picked_llm = self.llms[0]
        else:
            picked_llm = random.choice(self.llms)
        return picked_llm.with_structured_output(output_schema, **kwargs)

    @property
    def llm(self):
        """Return a single ChatOpenAI instance for compatibility."""
        if len(self.llms) == 1:
            return self.llms[0]
        else:
            return random.choice(self.llms)

    def call_straight(self, messages: List[BaseMessage] = None, prompt: str = None, n: int = 1, temperature: float = None, top_p: float = None, top_k: int = None, max_tokens: int = 22768, reasoning_mode: bool = True):
        """
        Call the LLM directly with either messages or prompt string
        
        Args:
            messages: List of BaseMessage objects (HumanMessage, SystemMessage, AIMessage)
            prompt: Direct prompt string (used if messages is None)
            n: Number of responses to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate
        
        Returns:
            List of response strings
        """
        # Convert messages to the format expected by tokenizer
        if messages is not None:
            # Convert BaseMessage objects to dict format
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    formatted_messages.append({"role": "system", "content": msg.content})
                elif isinstance(msg, HumanMessage):
                    formatted_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    formatted_messages.append({"role": "assistant", "content": msg.content})
            
            # Apply chat template
            prompt_text = self.tokenizer.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=True, 
                enable_thinking=reasoning_mode,
            )
        else:
            # Use the provided prompt directly
            prompt_text = prompt
        headers = {
            'Content-Type': 'application/json',
        }

        json_data = {
            'model': self.model,
            'prompt': prompt_text,
            'max_tokens': max_tokens,
            'n': n,
        }

        # Conditionally add arguments if they are not None
        if temperature is not None:
            json_data["temperature"] = temperature
        if top_p is not None:
            json_data["top_p"] = top_p
        if top_k is not None:
            json_data["top_k"] = top_k

        answers = []
        thinkings = []
        # Pick a random base URL for the request
        if len(self.llms) == 1:
            host = self.base_url[0] + "/completions"
        else:
            host = random.choice(self.base_url) + "/completions"

        responses = requests.post(host, headers=headers, json=json_data)
        # print(responses.json())
        
        for response in responses.json()["choices"]:
            # print(response)
            # print("-" * 50)
            try:
                if "<think>" in response["text"]:
                    answers.append(response["text"].split("</think>")[1].strip())
                    thinkings.append(response["text"].split("</think>")[0].replace("<think>", "").strip())
                else:
                    answers.append(response["text"])
                    thinkings.append("")
            except:
                print("One response die because of model overthinking")
            
        return answers, thinkings
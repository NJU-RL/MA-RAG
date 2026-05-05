import os
import numpy as np
from copy import deepcopy
from logging import Logger
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from time import sleep
from typing import List, Dict, Optional


EXTRA_BODY_MAP = {
    'qwen3-8b': {'chat_template_kwargs': {'enable_thinking': False}},
}

DEFAULT_EXTRA_BODY = {}

class CustomLanguageModel:
    def __init__(self, model_name_or_path: str, logger: Logger, base_url: Optional[str] = None):
        self.llm_name = model_name_or_path.split('/')[-1]
        self.max_retry_times = 5
        self.logger = logger
        self.client = OpenAI(
            api_key=os.getenv('API_KEY'),
            base_url=os.getenv('BASE_URL') if base_url is None else base_url,
        )

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.6, max_new_tokens: int = 2048, top_logprobs: int = 0, enable_thinking: bool = False, continue_final_message: bool = False,  **kwargs) -> ChatCompletion:
        '''
        :param top_logprobs: 0 means not use logprobs, defaults to 0.
        '''
        extra_body = deepcopy(EXTRA_BODY_MAP.get(self.llm_name.lower(), DEFAULT_EXTRA_BODY))
        if enable_thinking:
            extra_body['chat_template_kwargs']['enable_thinking'] = True

        if top_logprobs:
            kwargs['logprobs'] = True
            kwargs['top_logprobs'] = top_logprobs
        if 'top_k' in kwargs:
            extra_body['top_k'] = kwargs['top_k']
            del kwargs['top_k']

        if continue_final_message:
            extra_body['continue_final_message'] = True
            extra_body['chat_template_kwargs']['add_generation_prompt'] = False

        for _ in range(self.max_retry_times):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_name,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_new_tokens,
                    extra_body=extra_body,
                    timeout=600,
                    **kwargs,
                )
                return response
            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.logger.warning(f"Timeout in generating response using {self.llm_name} model: {e}")
                sleep(np.random.uniform(2, 3))
        else:
            raise TimeoutError(f"Failed to generate response using {self.llm_name} model after {self.max_retry_times} retries.")

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        '''
        Only return the generated response, without any additional information. Same parameters as `generate` method. (messages, temperature, max_new_tokens, top_logprobs, enable_thinking).

        :return: Generated response.
        '''
        response = self.generate(messages=messages, **kwargs)
        answer = response.choices[0].message.content
        if 'stop_reason' in response.choices[0].model_extra and response.choices[0].model_extra['stop_reason'] and answer:
            answer += response.choices[0].model_extra['stop_reason']
        return answer.strip()


if __name__ == '__main__':
    llm_name = 'qwen3-8b'
    model = CustomLanguageModel(llm_name)
    response = model.generate([{'role': 'user', 'content': 'Hello, how are you?'}], use_logprob=False)
    print(response)
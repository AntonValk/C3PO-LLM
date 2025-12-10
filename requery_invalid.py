import openai
import httpx
import re
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import List, Self, Optional, Literal
from enum import StrEnum



class DatasetType(StrEnum):
    AQUA = "AQuA"
    CommonsenseQA = "CommonsenseQA"
    GSM8K = "GSM8K"
    SVAMP = "SVAMP"
    
    BIGBENCH_DATE = "Bigbench_Date"
    BIGBENCH_CAUSALJUDGEMENT = "Bigbench_Causal_Judgement"
    BIGBENCH_MOVIE_RECOMENDATION = "Bigbench_Movie_Recommendation"
    BIGBENCH_FORMAL_FALLACIES = "Bigbench_Formal_Fallacies"
    BIGBENCH_DisambiguationQA = "Bigbench_DisambiguationQA"
    BIGBENCH_SNARKS = "Bigbench_Snarks"
    BIGBENCH_SPORTS = "Bigbench_Sports"
    BIGBENCH_GEOMETRIC_SHAPES = "Bigbench_Geometric_Shapes"
    BIGBENCH_PENGUINS = "Bigbench_Penguins"
    BIGBENCH_RUIN_NAMES = "Bigbench_Ruin_Names"
    BIGBENCH_TEMPORAL_SEQUENCES = "Bigbench_Temporal_Sequences"
    
    MATH_500 = "Math_500"


class LLMType(StrEnum):
    DeepSeekV3 = "deepseek-ai/DeepSeek-V3"
    DeepSeekR1 = "deepseek-ai/DeepSeek-R1"
    LLaMA70B33 = "meta-llama/Llama-3.3-70B-Instruct"
    LLaMA70B31 = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    LLaMA8B31 = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    LLaMA405B31 = "meta-llama/Meta-Llama-3.1-405B-Instruct"
    LLaMA3B32 = "meta-llama/Llama-3.2-3B-Instruct"
    LLaMA1B32 = "meta-llama/Llama-3.2-1B-Instruct"
    Qwen1B = "Qwen/Qwen2.5-1.5B-Instruct"
    Qwen32B = "Qwen/Qwen2.5-32B-Instruct"
    Qwen72B = "Qwen/Qwen2.5-72B-Instruct"
    Mixtral8x7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    Mixtral8x22B = "mistralai/Mixtral-8x22B-Instruct-v0.1"

    #gpts
    GPT3 = "gpt-3.5-turbo"
    GPT4oMINI = "gpt-4o-mini"
    o3MINI = "o3-mini"
    
    @property
    def is_openai(self) -> bool:
        if self in [LLMType.GPT3, LLMType.GPT4oMINI, LLMType.o3MINI]:
            return True
        return False


class LLMConfig(BaseModel):
    name: LLMType
    temperature: float = Field(default=0.7, ge=0)
    max_tokens: int = Field(default=2048, gt=1)
    
class CoTMetadata(BaseModel):
    cost: Optional[float] = None
    finish_reason: Literal["stop", "max_tokens", "content_filter", "run_cancelled", "run_expired", "run_failed"]


class CoT(BaseModel):
    raw_text: str
    answer: Optional[str] = None
    metadata: CoTMetadata



class InvalidEntryMetadata(BaseModel):
    datatype: DatasetType
    llmtype: LLMType
    index: int
    
    
class InvalidEntry(BaseModel):
    metadata: InvalidEntryMetadata
    request_mesage: str
    system_prompt: Optional[str] = None
    llm_config: LLMConfig
    responses: Optional[List[CoT]] = None 
    
    
class InvalidEntryList(BaseModel):
    invalid_entries: List[InvalidEntry]
    
    def to_json(self, filename):
        report_str = self.model_dump_json(indent=2)
        with open(filename, "w") as file:
            file.write(report_str)

    @classmethod
    def from_json(cls, filename: str) -> Self:
        with open(filename, "r") as file:
            data = file.read()
        return InvalidEntryList.model_validate_json(data)


class BoxedAnswerParser(object):
    REMOVE_SPACES_REGEX = re.compile(r"\s|\\,|\\!|\\;")
    CLEAN_ANSWER_REGEX = re.compile(r"\\?\$|\\left|\\right|\^\{?\\circ\}?|\\%|(?<=[\di\}])\\(?:text|mbox)\{[\w\.]+\}(?:\^\d)?|[paPA]\.?[mM]\.?|\.$")
    REPLACE_COMMAS_REGEX = re.compile(r"(?<=\d)(?:,|\{,\})(?=\d{3})")
    
    def __call__(self, llm_responce: str) -> Optional[str]:
        stack = 1
        i = 0
        ans = llm_responce.split("boxed{")
        if len(ans) == 1:
            return None
        ans = ans[-1]
        for c in ans:
            if(c == '{'):
                stack += 1
            elif(c == '}'):
                stack -= 1
                if(stack == 0): 
                    break
            i += 1
        ans = BoxedAnswerParser.REMOVE_SPACES_REGEX.sub("", ans[:i])
        ans = BoxedAnswerParser.CLEAN_ANSWER_REGEX.sub("", ans)
        ans = BoxedAnswerParser.REPLACE_COMMAS_REGEX.sub("", ans)
        return ans


class OpenAILLMWrapper(object):
    def __init__(self, api_key):
        self.api_key = api_key
        self.answer_parser = BoxedAnswerParser()
              
    def __call__(self, message: str, config: LLMConfig, system_prompt: Optional[str]=None) -> List[CoT]:
        num_responses = self._get_num_responses(config.name)
        client = self._init_client()
        completion_kwargs_dict = self._update_completions_kwargs({
            "messages": self._make_messages(message, system_prompt),
            "model": config.name,
            "n": num_responses,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        })
        chat_completion = client.chat.completions.create(**completion_kwargs_dict)
        return [ self._parce_openai_choice(ccc) for ccc in chat_completion.choices ]
    
    
    def _parce_openai_choice(self, choice: openai.types.chat.chat_completion.Choice) -> CoT:
        finish_reason = choice.finish_reason
        if finish_reason == "length":
            finish_reason = "max_tokens"
        return CoT(raw_text=choice.message.content,
                    answer=self.answer_parser(choice.message.content),
                    metadata=CoTMetadata(finish_reason=finish_reason)
                )

    @staticmethod
    def _make_messages(message: str, system_prompt: Optional[str] = None):
        messages = []
        if system_prompt is not None:
            messages.append({
                    "role": "system",
                    "content": system_prompt,
                    })
        messages.append({
                    "role": "user",
                    "content": message,
                    })
        return messages
    
    def _init_client(self) -> openai.OpenAI:
        client = None
        if not SSL:
            client = httpx.Client(verify=False)
        return openai.OpenAI(
            api_key=self.api_key,
            base_url=self._base_url,
            http_client=client
        )
    
    def _get_num_responses(self, llm_type: LLMType) -> int:
        if llm_type is LLMType.o3MINI:
            return 3
        if llm_type is LLMType.DeepSeekR1:
            return 3
        return 40
    
    @property
    def _base_url(self):
        return None
    
    def _update_completions_kwargs(self, kwargs):
        if kwargs["model"] in [LLMType.o3MINI]:
            kwargs["max_completion_tokens"] = kwargs["max_tokens"]
            del kwargs["max_tokens"]
            del kwargs["temperature"]
        return kwargs


class NebiusLLMWrapper(OpenAILLMWrapper):
    @property
    def _base_url(self) -> str:
        return "https://api.studio.nebius.ai/v1"
    
    def _update_completions_kwargs(self, kwargs):
        return kwargs


def main(input_filename, output_filename, api_keys):
    invalid_inpus = InvalidEntryList.from_json(input_filename)
    outputs = []
    openAI_API = OpenAILLMWrapper(api_keys["openai"])
    nebius_API = NebiusLLMWrapper(api_keys["nebius"])
    
    for i, inp in enumerate(tqdm(invalid_inpus.invalid_entries)):
        if inp.responses is None:
            try:
                if inp.metadata.llmtype.is_openai:
                    cots = openAI_API(inp.request_mesage, inp.llm_config, inp.system_prompt)
                else:
                    cots = nebius_API(inp.request_mesage, inp.llm_config, inp.system_prompt)
                outputs.append(inp.model_copy(update={"responses": cots}))
            except KeyboardInterrupt:
                outputs.extend(invalid_inpus.invalid_entries[i:])
                InvalidEntryList(invalid_entries=outputs).to_json(output_filename)
                raise
            except Exception as e:
                print(e)
                outputs.append(inp)
        else:
            outputs.append(inp)
    
    InvalidEntryList(invalid_entries=outputs).to_json(output_filename)

if __name__ == "__main__":
    INPUT = "invalid_responses.json"
    OUTPUT = "invalid_responses.json"
    API_KEYS = {
        "nebius": "KEY_GOES_HERE",
        "openai": "KEY_GOES_HERE"
    }
    SSL=False
    main(INPUT, OUTPUT, API_KEYS)
    
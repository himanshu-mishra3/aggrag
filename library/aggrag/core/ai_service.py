from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


from library.aggrag.core.config import (
    settings,
    AzureOpenAIModelNames,
    AzureOpenAIModelEngines,
    TogetherLLMModelNames,
    ReplicateModelNames
)
from llama_index.llms.replicate import Replicate
from llama_index.llms.together import TogetherLLM
from llama_index.llms.openai import OpenAI
from library.aggrag.core.config import ai_services_config
from library.aggrag.core.config import settings, AzureOpenAIModelNames, AzureOpenAIModelEngines, OpenAIModelNames
from llama_index.llms.anthropic import Anthropic
import logging
logger = logging.getLogger(__name__)


rag_temperature = 0.1


class AzureAIService:
    def __init__(self, model=None, deployment_name=None, api_key=None, azure_endpoint=None, api_version=None, embed_model=None):
        try:
            model = ai_services_config["AzureOpenAI"]["chat_models"].get(model or AzureOpenAIModelNames.gpt_35_turbo.value).get("model_name", None)
            embed_model = ai_services_config["AzureOpenAI"]["embed_models"].get(embed_model or AzureOpenAIModelNames.text_embedding_ada_002.value).get("model_name", None)
            chat_model_deployment_name = ai_services_config["AzureOpenAI"]["chat_models"].get(model).get("deployment_name", None)
            embed_model_deployment_name = ai_services_config["AzureOpenAI"]["embed_models"].get(embed_model).get("deployment_name", None)
        except Exception as e:
            print(f"Error accessing model information from AZURE_SERVICE_CONFIG: {e}")
            raise

        self.llm = AzureOpenAI(
            model=model or AzureOpenAIModelNames.gpt_35_turbo.value,
            deployment_name=chat_model_deployment_name or AzureOpenAIModelEngines.gpt_35_turbo.value,
            api_key=api_key or settings.AZURE_OPENAI_KEY,
            azure_endpoint=azure_endpoint or settings.AZURE_API_BASE,
            api_version=api_version or settings.OPENAI_API_VERSION,
            temperature=rag_temperature
        )
    
        self.embed_model = AzureOpenAIEmbedding(
            model = embed_model or AzureOpenAIModelNames.text_embedding_ada_002.value,
            deployment_name = embed_model_deployment_name or AzureOpenAIModelEngines.text_embedding_ada_002.value,
            api_key=api_key or settings.AZURE_OPENAI_KEY,
            azure_endpoint = azure_endpoint or settings.AZURE_API_BASE,
            api_version = api_version or settings.OPENAI_API_VERSION
        )

        logger.info(f"llm_model :  {self.llm}")
        logger.info(f"embed_model : {self.embed_model}")

class ReplicateAIService:
    def __init__(self, model=None, embed_model=None):

        self.llm = Replicate(
            model=model or ReplicateModelNames.meta_llama_3_70b_instruct.value,
            temperature=0.1,
            # context_window=32,
        )
        logger.info(f"llm_model :  {self.llm}")



class TogetherAIService:
    def __init__(self, model=None, embed_model=None):
        self.llm = TogetherLLM(
            model=model or TogetherLLMModelNames.mixtral_8x7b_instruct.value,
            api_key=settings.TOGETHER_API_KEY,
        )

        logger.info(f"llm_model :  {self.llm}")


class OpenAIService:
    def __init__(self, model=None, deployment_name=None, api_key=None, azure_endpoint=None, api_version=None, embed_model=None):

        self.llm = OpenAI(
            model=model or OpenAIModelNames.gpt_4_turbo.value,
            api_key=settings.OPENAI_API_KEY,
        )
        
        self.embed_model = OpenAIEmbedding(
            model = embed_model or OpenAIModelNames.text_embedding_ada_002.value,
            api_key=api_key or settings.OPENAI_API_KEY)

        logger.info(f"llm_model :  {self.llm}")
        logger.info(f"embed_model : {self.embed_model}")


class AnthropicAIService:
    def __init__(self, model=None, embed_model=None):
        self.llm = Anthropic(
            model=model or "claude-3-opus-20240229",
            api_key=settings.ANTHROPIC_API_KEY
        )
        logger.info(f"llm_model :  {self.llm}")

class AIServiceFactory:
    @staticmethod
    def get_ai_service(ai_service, llm_model=None, embed_model=None):
        if ai_service not in ai_services_config.keys():
            raise ValueError(f"Unsupported AI service: {ai_service}")
        # Check if the model is valid for the selected service
        # model_names = [model['model_name'] for model in ai_services_config.get(ai_service, {}).values()]
        # if llm_model not in model_names:
        #     raise ValueError(f"Model '{llm_model}' is not available for service '{ai_service}'")


        
        # Return the appropriate AI service instance
        if ai_service == "AzureOpenAI":
            return AzureAIService(model=llm_model, embed_model=embed_model)
        elif ai_service == "Replicate":
            return ReplicateAIService(model=llm_model, embed_model=embed_model)
        elif ai_service == "Together":
            return TogetherAIService(model=llm_model, embed_model=embed_model)
        elif ai_service == "OpenAI":
            return OpenAIService(model=llm_model, embed_model=embed_model)
        elif ai_service == "Anthropic":
            return AnthropicAIService(model=llm_model, embed_model=embed_model)
        else:
            raise ValueError(f"Unsupported AI service: {ai_service}")
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


from library.aggrag.core.config import (
    settings,
    AzureOpenAIModelNames,
    TogetherLLMModelNames,
    ReplicateModelNames
)
from llama_index.llms.replicate import Replicate
from llama_index.llms.together import TogetherLLM
from llama_index.llms.openai import OpenAI
from library.aggrag.core.config import ai_services_config
from library.aggrag.core.config import settings, AzureOpenAIModelNames, OpenAIModelNames, AnthropicModelNames
from llama_index.llms.anthropic import Anthropic



def get_model_info(service=None, model_type=None, model_name=None, ai_services_config=None):
    service_config = ai_services_config.get(service)
    if not service_config:
        raise ValueError(f"No configuration found for service '{service}'")

    model_type_config = service_config.get(model_type)
    if model_type_config is None:
        raise ValueError(f"No configuration found for {service} {model_type}")

    model_config = model_type_config.get(model_name)
    if not model_config:
        raise ValueError(f"No mapping found for {service} {model_type} '{model_name}' in the configuration.")

    model_name = model_config.get("model_name")
    if model_name is None:
        raise ValueError(f"No 'model_name' found in the configuration for {service} {model_type} '{model_name}'")

    deployment_name = None
    if service == "AzureOpenAI":
        deployment_name = model_config.get("deployment_name")
        if deployment_name is None:
            raise ValueError(f"No 'deployment_name' found in the configuration for {service} {model_type} '{model_name}'")

    return model_name, deployment_name

class AzureAIService:
    def __init__(self, **kwargs):
        try:
            # Extract specific parameters from kwargs
            llm_model = kwargs.get('llm_model')
            embed_model = kwargs.get('embed_model')
            temperature = kwargs.get('temperature')  # Use default if not provided

            # Get LLM model info
            model, chat_model_deployment_name = get_model_info(
                "AzureOpenAI",
                "chat_models",
                llm_model or AzureOpenAIModelNames.gpt_35_turbo.value,
                ai_services_config
            )

            # Get embedding model info
            embed_model, embed_model_deployment_name = get_model_info(
                "AzureOpenAI",
                "embed_models",
                embed_model or AzureOpenAIModelNames.text_embedding_ada_002.value,
                ai_services_config
            )

            # Initialize the LLM
            self.llm = AzureOpenAI(
                model=model,
                deployment_name=chat_model_deployment_name,
                api_key=kwargs.get('api_key') or settings.AZURE_OPENAI_KEY,
                azure_endpoint=kwargs.get('azure_endpoint') or settings.AZURE_API_BASE,
                api_version=kwargs.get('api_version') or settings.OPENAI_API_VERSION,
                temperature=temperature
            )

            # Initialize the embedding model
            self.embed_model = AzureOpenAIEmbedding(
                model=embed_model,
                deployment_name=embed_model_deployment_name,
                api_key=kwargs.get('api_key') or settings.AZURE_OPENAI_KEY,
                azure_endpoint=kwargs.get('azure_endpoint') or settings.AZURE_API_BASE,
                api_version=kwargs.get('api_version') or settings.OPENAI_API_VERSION,
            )

        except Exception as e:
            print(f"Error initializing AzureAIService: {e}")
            raise

class ReplicateAIService:
    def __init__(self, **kwargs):
        try:
            model, _ = get_model_info(
                "Replicate",
                "chat_models",
                kwargs.get('llm_model') or ReplicateModelNames.meta_llama_3_70b_instruct.value,
                ai_services_config
            )
            temperature = kwargs.get('temperature')

            self.llm = Replicate(
                model=model,
                temperature=temperature,
            )

            # Initialize the embedding model
            embed_model, embed_model_deployment_name = get_model_info(
                "AzureOpenAI",
                "embed_models",
                kwargs.get('embed_model') or AzureOpenAIModelNames.text_embedding_ada_002.value,
                ai_services_config
            )
            self.embed_model = AzureOpenAIEmbedding(
                model=embed_model,
                deployment_name=embed_model_deployment_name,
                api_key=kwargs.get('api_key') or settings.AZURE_OPENAI_KEY,
                azure_endpoint=kwargs.get('azure_endpoint') or settings.AZURE_API_BASE,
                api_version=kwargs.get('api_version') or settings.OPENAI_API_VERSION
            )

        except Exception as e:
            print(f"Error initializing ReplicateAIService: {e}")
            raise

class TogetherAIService:
    def __init__(self, **kwargs):
        try:
            model, _ = get_model_info(
                "Together",
                "chat_models",
                kwargs.get('llm_model') or TogetherLLMModelNames.mixtral_8x7b_instruct.value,
                ai_services_config
            )
            temperature = kwargs.get('temperature')

            self.llm = TogetherLLM(
                model=model,
                api_key=settings.TOGETHER_API_KEY,
                temperature=temperature
            )

            # Initialize the embedding model
            embed_model, embed_model_deployment_name = get_model_info(
                "AzureOpenAI",
                "embed_models",
                kwargs.get('embed_model') or AzureOpenAIModelNames.text_embedding_ada_002.value,
                ai_services_config
            )
            self.embed_model = AzureOpenAIEmbedding(
                model=embed_model,
                deployment_name=embed_model_deployment_name,
                api_key=kwargs.get('api_key') or settings.AZURE_OPENAI_KEY,
                azure_endpoint=kwargs.get('azure_endpoint') or settings.AZURE_API_BASE,
                api_version=kwargs.get('api_version') or settings.OPENAI_API_VERSION
            )

        except Exception as e:
            print(f"Error initializing TogetherAIService: {e}")
            raise

class OpenAIService:
    def __init__(self, **kwargs):
        try:
            model, _ = get_model_info(
                "OpenAI",
                "chat_models",
                kwargs.get('llm_model') or OpenAIModelNames.gpt_35_turbo.value,
                ai_services_config
            )
            embed_model, _ = get_model_info(
                "OpenAI",
                "embed_models",
                kwargs.get('embed_model') or OpenAIModelNames.text_embedding_ada_002.value,
                ai_services_config
            )
            temperature = kwargs.get('temperature')

            self.llm = OpenAI(
                model=model,
                api_key=settings.OPENAI_API_KEY,
                temperature=temperature
            )

            self.embed_model = OpenAIEmbedding(
                model=embed_model,
                api_key=kwargs.get('api_key') or settings.OPENAI_API_KEY
            )
        except Exception as e:
            print(f"Error initializing OpenAIService: {e}")
            raise

class AnthropicAIService:
    def __init__(self, **kwargs):
        try:
            model, _ = get_model_info(
                "Anthropic",
                "chat_models",
                kwargs.get('llm_model') or AnthropicModelNames.claude_3_opus_20240229.value,
                ai_services_config
            )
            temperature = kwargs.get('temperature')

            self.llm = Anthropic(
                model=model,
                api_key=settings.ANTHROPIC_API_KEY,
                temperature=temperature
            )
            # Initialize the default embedding model
            embed_model, embed_model_deployment_name = get_model_info(
                "AzureOpenAI",
                "embed_models",
                kwargs.get('embed_model') or AzureOpenAIModelNames.text_embedding_ada_002.value,
                ai_services_config
            )
            self.embed_model = AzureOpenAIEmbedding(
                model=embed_model,
                deployment_name=embed_model_deployment_name,
                api_key=kwargs.get('api_key') or settings.AZURE_OPENAI_KEY,
                azure_endpoint=kwargs.get('azure_endpoint') or settings.AZURE_API_BASE,
                api_version=kwargs.get('api_version') or settings.OPENAI_API_VERSION
            )

        except Exception as e:
            print(f"Error initializing AnthropicAIService: {e}")
            raise
class AIServiceFactory:
    @staticmethod
    def get_ai_service(service=None, **kwargs):
        
        if service not in ai_services_config.keys():
            raise ValueError(f"Unsupported AI service: {service}")
        
        # Return the appropriate AI service instance
        if service == "AzureOpenAI":
            return AzureAIService(**kwargs)
        elif service == "Replicate":
            return ReplicateAIService(**kwargs)
        elif service == "Together":
            return TogetherAIService(**kwargs)
        elif service == "OpenAI":
            return OpenAIService(**kwargs)
        elif service == "Anthropic":
            return AnthropicAIService(**kwargs)
        else:
            raise ValueError(f"Unsupported AI service: {service}")
import abc
from typing import Optional

class AggragBaseAbstract(abc.ABC):
    """
    Abstract base class for handling document loading, index creation and retrieval, and chat functionalities
    for a specific configuration of RAG.
    """

    @abc.abstractmethod
    def __init__(self, usecase_name: str, iteration: str, DATA_DIR: Optional[str] = None, upload_type: Optional[str] = None, base_rag_setting=None, llm=None, embed_model=None):
        """
        Initializes a base configuration for RAG with given parameters, setting up directories and logging essential information.
        """
        pass

    @abc.abstractmethod
    def documents_loader(self, DIR=None):
        """
        Placeholder for a RAG-specific document loader method.
        """
        pass

    @abc.abstractmethod
    async def create_index_async(self, documents=None):
        """
        Asynchronously creates an index from the provided documents using specified embedding models and parsers.
        """
        pass

    @abc.abstractmethod
    async def retrieve_index_async(self, documents=None, upload_index: bool = False):
        """
        Asynchronously retrieves an existing index either from local storage or Azure blob storage, based on configuration.
        """
        pass

    @abc.abstractmethod
    async def get_chat_engine(self):
        """
        Configures and retrieves a chat engine based on the index and provided settings.
        """
        pass

    @abc.abstractmethod
    async def achat(self, query, chat_history=None, is_evaluate=False):
        """
        Asynchronously handles a single chat query, performs evaluation if required, and logs the chat response.
        """
        pass

    # @abc.abstractmethod
    # async def astream_chat(self, query, chat_history=None, is_evaluate=False):
    #     """
    #     Asynchronously handles a streaming chat session, providing real-time responses to a continuous query stream.
    #     """
    #     pass

    # @abc.abstractmethod
    # async def run(self, query, chat_history=None, is_evaluate=False):
    #     """
    #     Asynchronously handles a streaming chat session, providing real-time responses to a continuous query stream.
    #     """
    #     pass
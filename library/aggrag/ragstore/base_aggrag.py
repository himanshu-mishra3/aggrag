import os,time
import logging
from typing import Optional
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.chat_engine.types import ChatMode
from library.aggrag.core.utils import get_time_taken
from library.aggrag.evals.llm_evaluator import Evaluator
from library.aggrag.core.config import settings
from library.aggrag.aggrag_base_abstract import AggragBaseAbstract
from datasets import Dataset
from ragas.metrics import answer_relevancy

metrics = [answer_relevancy]

logger = logging.getLogger(__name__)

class Base(AggragBaseAbstract):
    def __init__(self, usecase_name: str, iteration: str, DATA_DIR: Optional[str] = None, upload_type: Optional[str] = None, base_rag_setting=None, llm=None, embed_model=None):
        self.name = 'base v2'
        self.ai_service = base_rag_setting.ai_service
        self.chunk_size = base_rag_setting.chunk_size
        self.system_prompt = base_rag_setting.system_prompt
        self.context_prompt = base_rag_setting.context_prompt
        self.llm = llm
        self.embed_model = embed_model
        self.documents = None
        self.index_name = base_rag_setting.index_name or "base_index"
        self.index = None
        self.usecase_name = usecase_name
        self.iteration = iteration
        self.BASE_DIR = os.path.join("configurations", self.usecase_name, self.iteration)
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'raw_docs')
        self.PERSIST_DIR = os.path.join(self.BASE_DIR, 'index')
        self.upload_type = upload_type
        self.model_name = ''
        self.chat_engine = None

        # Set up data directory based on upload type
        if self.upload_type == 'url' and os.path.exists(os.path.join(DATA_DIR, 'raw_docs')):
            self.DATA_DIR = os.path.join(DATA_DIR, 'raw_docs')
            logger.info(f"Data directory: {self.DATA_DIR}")
        elif self.upload_type == 'url':
            self.DATA_DIR = os.path.join(DATA_DIR, 'html_files')
            logger.info(f"Data directory: {self.DATA_DIR}")
        elif self.upload_type == 'doc' or self.upload_type == 'pdf':
            self.DATA_DIR = os.path.join(DATA_DIR, 'raw_docs')
            logger.info(f"Data directory: {self.DATA_DIR}")

        logger.info(f"embed model: {self.embed_model}")
        logger.info(f"llm model: {self.llm}")

    def documents_loader(self, DIR=None):
        """
        Placeholder for a RAG-specific document loader method.
        """
        raise NotImplementedError("documents_loader method must be implemented in the concrete class.")

    async def create_index_async(self, documents=None):
        """
        Asynchronously creates an index from the provided documents using specified embedding models and parsers.
        """
        persistent_path = os.path.join(self.PERSIST_DIR, self.index_name)
        if not documents:
            documents = self.documents
        
        # Index creation logic remains unchanged


        if self.upload_type == 'url':

            html_parser = HTMLNodeParser(tags=["p","h1", "h2", "h3", "h4", "h5", "h6", "li", "b", "i", "u", "section"])  # Keeping the default tags which llama-index provides
            pdf_files = [file for file in os.listdir(self.DATA_DIR) if file.lower().endswith(".pdf")] 
            html_files = [file for file in os.listdir(self.DATA_DIR) if file.lower().endswith(".html")]

            all_nodes=[]

            if pdf_files:

                logger.info("Using  Simple Node  parser to parse parent pdf page")            
                pdf_parser = SimpleNodeParser.from_defaults()
                pdf_nodes=pdf_parser.get_nodes_from_documents(documents, show_progress=True)           
                all_nodes.extend(pdf_nodes)

            if html_files:

                logger.info("Using  HTML nodes parser to parse htmls and index")            
                html_nodes = html_parser.get_nodes_from_documents(documents, show_progress=True)
                all_nodes.extend(html_nodes)

            index = VectorStoreIndex(all_nodes, embed_model=self.embed_model, show_progress=True)
        else:
            index = VectorStoreIndex.from_documents(documents, 
                                                    embed_model=self.embed_model,
                                                    show_progress=True, #use_async=True
                                                    )

        os.makedirs(os.path.dirname(self.PERSIST_DIR), exist_ok=True)  
        index.storage_context.persist(persist_dir=persistent_path)

        return index

        # raise NotImplementedError("create_index_async method must be implemented in the concrete class.")

    async def retrieve_index_async(self, documents=None, upload_index: bool = False):
        """
        Asynchronously retrieves an existing index either from local storage or Azure blob storage, based on configuration.
        """
        persistent_path = os.path.join(self.PERSIST_DIR, self.index_name)

        if os.path.exists(persistent_path):
            logger.debug(f"Retrieving index from the persistent path: {persistent_path}")
            storage_context = StorageContext.from_defaults(persist_dir=persistent_path)
            self.index = load_index_from_storage(storage_context, embed_model=self.embed_model)

        return self.index

    async def get_chat_engine(self):
        """
        Configures and retrieves a chat engine based on the index and provided settings.
        """
        index = await self.retrieve_index_async()
        if not index:
            raise Exception("Index is not available or unable to fetch")

        self.chat_engine = self.index.as_chat_engine(
            chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
            system_prompt=self.system_prompt,
            llm=self.llm,
            context_prompt=self.context_prompt
        )

        return self.chat_engine

    async def achat(self, query, chat_history=None, is_evaluate=False):
        """
        Asynchronously handles a single chat query, performs evaluation if required, and logs the chat response.
        """
        await self.get_chat_engine()

        logger.debug(f"Chat engine: {self.chat_engine}")
        start_time = time.time()
        response = await self.chat_engine.achat(query, chat_history=chat_history)
        logger.debug(f"Base response: {response.response}")
        interim_time = time.time()
        try:
            page_labels = [i.metadata['page_label'] for i in response.source_nodes]
            page_labels.sort()
        except Exception as e:
            logger.info(f"Could not retrieve page labels in response source nodes {e}")
            page_labels = []

        evaluation_score = None
        if is_evaluate:
            contexts = []
            contexts.append([c.node.get_content() for c in response.source_nodes])

            dataset_dict = {
                "question": [query],
                "answer": [response.response],
                "contexts": contexts,
            }

            ds_chat_engine = Dataset.from_dict(dataset_dict)
            evaluator1 = Evaluator(self.documents,
                                   None,
                                   self.llm,
                                   self.embed_model,
                                   rag_name=f"{'aggrag'}_{self.name}",
                                   project_path=f"{os.getcwd()}",
                                   model_name=self.model_name)

            eval_result = await evaluator1.aevaluate_models(None, None, metrics, ds_chat_engine)

            evaluation_score = round(eval_result.answer_relevancy.values[0], 2)

        final_time = time.time()
        return {
            "response": response.response,
            "page_labels": page_labels,
            "evaluation_score": evaluation_score,
            "time_taken": get_time_taken(start_time, interim_time, final_time),
            "rag_name": f"{self.name}"
        }

    async def astream_chat(self, query, chat_history=None, is_evaluate=False):
        """
        Asynchronously handles a streaming chat session, providing real-time responses to a continuous query stream.
        """
        raise NotImplementedError("astream_chat method must be implemented in the concrete class.")
"""Main module."""

import getpass
import logging
import os
import sys
from typing import List, Optional

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .prompts import MAP_PROMPT, PREPROCESSED_PROMPT


def set_openai_api_key():
    """Set OpenAI API key from environment variable or prompt."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter Open AI API key: ")


class VocabularyStore:
    """Vocabulary class."""

    def __init__(self, index_path: str, local_path: Optional[str] = None):
        """Initialize the Vocabulary class."""
        self.logger = logging.getLogger("VocabularyStore")
        set_openai_api_key()
        self.embeddings = OpenAIEmbeddings(timeout=30)
        if os.path.exists(f"{index_path}/index.faiss"):
            self.db = FAISS.load_local(
                index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True
            )
        else:
            self.logger.debug("Index not found. Creating a new index.")
            if not local_path:
                self.logger.error("File path is required to create a new index.")
                sys.exit(1)
            else:
                if not os.path.exists(local_path):
                    self.logger.error(f"File not found: {local_path}")
                    sys.exit(1)
                self.data = VocabularyStore.index_vocabulary(local_path)
                self.logger.info("Embedding the vocabulary using OpenAI.")
                self.db = FAISS.from_documents(
                    self.data, OpenAIEmbeddings(show_progress_bar=True, timeout=30)
                )
                self.logger.info(f"Saving the index to {index_path}")
                self.db.save_local(index_path)

    @staticmethod
    def csv_validator(file_path: str) -> bool:
        """Validate the CSV file. Chech the header equals to ['Term','ID','Alias']."""
        with open(file_path, "r") as f:
            header = f.readline().strip().split(",")
            header = [h.replace('"', "") for h in header]
            if header != ["Term", "ID", "Alias"]:
                return False
        return True

    @staticmethod
    def index_vocabulary(file_path: str):
        """Index the vocabulary."""
        if not VocabularyStore.csv_validator(file_path):
            raise ValueError("Invalid CSV file. Header must be ['Term','ID','Alias']")
        csv_loader = CSVLoader(file_path)
        data = csv_loader.load()
        return data

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search the vocabulary."""
        results = self.db.similarity_search(query, k)
        return results

    def search_score(
        self, query: str, k: int = 5, score_threshold: Optional[float] = None
    ) -> List[dict]:
        """Search the vocabulary with score."""
        search_r = self.db.similarity_search_with_score(query, k)
        results = []
        for i, j in search_r:
            if score_threshold and j > score_threshold:
                continue
            results.append({"term": i.page_content, "score": j})
        return results


class TraitGPT:
    """TraitGPT class."""

    def __init__(
        self,
        index_path: str,
        file_path: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.3,
    ):
        """Initialize the TraitGPT class."""
        self.logger = logging.getLogger("TraitGPT")
        set_openai_api_key()
        self.vocabulary = VocabularyStore(index_path, file_path)
        self.chat = ChatOpenAI(model=model, temperature=temperature, timeout=30)

    def preprocess(self, trait: str) -> str:
        """Preprocess the trait."""
        return self.chat.invoke(PREPROCESSED_PROMPT.format(input=trait)).content  # type: ignore

    def map_trait(self, trait: str) -> dict[str, str]:
        """Map the trait."""
        preprocessed_trait = self.preprocess(trait)
        self.logger.debug(f"Preprocessed: {trait} -> {preprocessed_trait}")
        res_d = self.vocabulary.search(preprocessed_trait)
        res = [i.page_content for i in res_d]
        self.logger.debug(f"Mapping: {preprocessed_trait} -> {res}")
        parser = JsonOutputParser(pydantic_object=Term)

        prompt = PromptTemplate(
            template=MAP_PROMPT
            + "\n{format_instructions}\nterminologies: {query_res}\nquery trait: {query_trait}\n",
            input_variables=["query_res", "query_trait"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.chat | parser

        map_res = chain.invoke({"query_res": res, "query_trait": preprocessed_trait})
        return map_res


class Term(BaseModel):
    """Term model."""

    term_name: str = Field(description="Perferred term in the terminology")
    term_id: str = Field(description="Unique ID of the term in the terminology")

import os
import logging
from typing import Optional
import re

from dotenv import load_dotenv
import google.generativeai as genai

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores.types import ExactMatchFilter, MetadataFilters

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.gemini import GeminiEmbedding

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Import prompts from the new prompts.py file
from prompts import (
    QUERY_GEN_PROMPT_TEMPLATE,
    DIRECT_PROMPT_TEMPLATE,
    INTENT_PROMPT_GEMINI,
)

load_dotenv()

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,  # Change to INFO when ready to deploy
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("guru_errors.log"), logging.StreamHandler()],
)


class GuruChatbot:
    def __init__(
        self,
        openai_api_key: str,
        google_api_key: str,
        qdrant_url: str,
        qdrant_api_key: str,
    ):
        """Initializes the GuruChatbot with API keys and Qdrant configuration."""
        self.openai_api_key = openai_api_key
        self.google_api_key = google_api_key
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self._llm = None
        self._embed_model = None
        self._genai_model = None
        self._vector_store = None
        self._index = None
        self._chat_engine = None
        self._query_engine_tao = None
        self._query_engine_meditations = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = OpenAI(
                model="gpt-4o-mini", api_key=self.openai_api_key, temperature=0.1
            )
        return self._llm

    @property
    def embed_model(self):
        if self._embed_model is None:
            self._embed_model = GeminiEmbedding(
                model_name="models/text-embedding-004", api_key=self.google_api_key
            )
        return self._embed_model

    @property
    def genai_model(self):
        if self._genai_model is None:
            genai.configure(api_key=self.google_api_key)
            self._genai_model = genai.GenerativeModel("gemini-1.5-flash")
        return self._genai_model

    @property
    def vector_store(self):
        if self._vector_store is None:
            client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
            self._vector_store = QdrantVectorStore(
                client=client, collection_name="guru_3"
            )
        return self._vector_store

    @property
    def index(self):
        if self._index is None:
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store
            )
        return self._index

    @property
    def chat_engine(self):
        if self._chat_engine is None:
            self._chat_engine = self.index.as_chat_engine(
                chat_mode="context",
                system_prompt=DIRECT_PROMPT_TEMPLATE,
            )
        return self._chat_engine

    @property
    def query_engine_tao(self):
        if self._query_engine_tao is None:
            similarity_postprocessor_tao = SimilarityPostprocessor(
                similarity_cutoff=0.58
            )
            filter_tao = MetadataFilters(
                filters=[ExactMatchFilter(key="doc_name", value="Tao Te Ching")]
            )
            self._query_engine_tao = self.index.as_query_engine(
                chat_mode="context",
                similarity_top_k=5,
                node_postprocessors=[similarity_postprocessor_tao],
                filters=filter_tao,
            )
        return self._query_engine_tao

    @property
    def query_engine_meditations(self):
        if self._query_engine_meditations is None:
            similarity_postprocessor_meditations = SimilarityPostprocessor(
                similarity_cutoff=0.64
            )
            filter_meditations = MetadataFilters(
                filters=[ExactMatchFilter(key="doc_name", value="Meditations")]
            )
            self._query_engine_meditations = self.index.as_query_engine(
                chat_mode="context",
                similarity_top_k=5,
                node_postprocessors=[similarity_postprocessor_meditations],
                filters=filter_meditations,
            )
        return self._query_engine_meditations

    def convert_meditations_format(self, text):
        """
        Convert the first line of text if it matches the format
        "**2.1** Some text..." to "Book 2, Chapter 1 --- Some text..."
        """
        pattern = re.compile(r"^\*\*(\d+)\.(\d+)\*\*\s+(.*)$")

        lines = text.strip().split("\n", 1)  # Split only at the first newline

        if lines:  # Ensure there's at least one line
            first_line = lines[0]
            match = pattern.match(first_line)
            if match:
                book = match.group(1)
                chapter = match.group(2)
                rest_of_line = match.group(3).strip()
                title = f"Book {book}, Chapter {chapter} ---"
                converted_first_line = f"{title} {rest_of_line}"

                # If there are more lines, combine them back
                if len(lines) > 1:
                    return f"{converted_first_line}\n{lines[1]}"
                else:
                    return converted_first_line
            else:
                return text  # If no match on the first line, return original text
        else:
            return ""  # Handle empty input

    def convert_tao_format(self, text):
        """
        Adds "---" after the "Chapter XX" portion of the text if it exists at the beginning.

        Args:
            text: The input string.

        Returns:
            The modified string with "---" added, or the original string if "Chapter" is not found.
        """
        if text.startswith("Chapter "):
            try:
                # Find the index of the first space after "Chapter "
                first_space_index = text.find(" ", len("Chapter "))
                if first_space_index != -1:
                    chapter_part = text[:first_space_index]
                    rest_of_text = text[first_space_index:]
                    return f"{chapter_part.strip()} --- {rest_of_text.strip()}"
                else:
                    # Handle cases like "Chapter 10" with no space after the number
                    return text
            except ValueError:
                # Handle potential errors if the string manipulation goes wrong
                return text
        return text

    def is_philosophical_query(self, question: str, temperature: float = 0.1) -> bool:
        """
        Determines if the query is related to philosophy or the content of the source texts using Gemini.
        """
        try:
            response = self.genai_model.generate_content(
                INTENT_PROMPT_GEMINI.format(question=question),
                generation_config=genai.types.GenerationConfig(temperature=temperature),
            )
            return "yes" in response.text.lower()
        except Exception as e:
            logging.error(f"Error during intent classification with Gemini: {e}")
            return False

    def generate_queries(
        self, query: str, num_queries: int = 2, temperature: float = 0.1
    ) -> str:
        """Generates alternative queries using the Gemini LLM.
        Returns a string with (ideally) the generated queries, one per line, or "NONE".
        """
        try:
            prompt = QUERY_GEN_PROMPT_TEMPLATE.format(
                num_queries=num_queries, query=query
            )
            response = self.genai_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature),
            )
            return response.text
        except Exception as e:
            logging.error(f"Error during query generation with Gemini: {e}")
            return "NONE"

    def get_advice(
        self, question: str
    ) -> Optional[tuple[str, Optional[str], Optional[str]]]:
        """
        Retrieves advice by analyzing the provided question.

        Args:
            question (str): The user's question to analyze.

        Returns:
            Optional[tuple[str, Optional[str], Optional[str]]]:
            A tuple containing:
            - A string with the advice response.
            - An optional string with a quote from the Tao Te Ching.
            - An optional string with a quote from Meditations.
            Returns None if an error occurs during processing.
        """
        if not self.is_philosophical_query(question):
            return (
                "I'm sorry, but that question is outside the scope of my expertise. I can only answer questions related to Meditations by Marcus Aurelius and the Tao Te Ching.",
                None,
                None,
            )

        try:
            # Generate alternative queries based on the user question
            # Then query our vector database twice using these alternative queries
            #   - Once with the Tao Te Ching vector database
            #   - Once with the Meditations vector database
            #   - The results from both queries will be the relevant text snippets from the books
            #   - Then we'll combine both results into a single string for the LLM to use as context
            questions = self.generate_queries(question)
            tao_query_result = self.query_engine_tao.query(questions)
            meditations_query_result = self.query_engine_meditations.query(questions)

            tao_context = "\n".join(
                [node.text for node in tao_query_result.source_nodes]
            )
            meditations_context = "\n".join(
                [node.text for node in meditations_query_result.source_nodes]
            )

            combined_context = f"Tao Te Ching Context:\n{tao_context}\n\nMeditations Context:\n{meditations_context}"
        except Exception as e:
            logging.error(
                f"Error during quering vector database and generating context: {e}"
            )
            return None

        try:
            formatted_tao_quote = "NONE"
            tao_quote = "NONE"
            if (
                hasattr(tao_query_result, "source_nodes")
                and tao_query_result.source_nodes
            ):
                tao_quote = tao_query_result.source_nodes[0].text

            if tao_quote != "NONE":
                formatted_tao_quote = self.convert_tao_format(tao_quote)

        except Exception as e:
            logging.error(f"Error extracting Tao quote: {e}")
            return None

        try:
            formatted_meditations_quote = "NONE"
            meditations_quote = "NONE"
            if (
                hasattr(meditations_query_result, "source_nodes")
                and meditations_query_result.source_nodes
            ):
                meditations_quote = meditations_query_result.source_nodes[0].text

            if meditations_quote != "NONE":
                formatted_meditations_quote = self.convert_meditations_format(
                    meditations_quote
                )
        except Exception as e:
            logging.error(f"Error extracting Meditations quote: {e}")
            return None

        try:
            response = self.llm.predict(
                DIRECT_PROMPT_TEMPLATE, context=combined_context, question=questions
            )
            return response, formatted_tao_quote, formatted_meditations_quote

        except Exception as e:
            logging.error(f"Error during advice generation and summary: {e}")
            return None


if __name__ == "__main__":
    # This is just for testing the backend in isolation
    chatbot = GuruChatbot(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    )

    print("Backend testing mode. Ask a question or type 'exit':")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        advice = chatbot.get_advice(query)
        if advice:
            print("Response:", advice)
        else:
            print("Could not get advice.")

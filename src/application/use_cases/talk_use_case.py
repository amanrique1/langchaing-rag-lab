from typing import List
from pathlib import Path
from src.domain.models.chunk import Chunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

# It's better to define this path logic outside the class
# so it's configured once on module import.
try:
    data_path = Path("assets/query_template.txt")
    if not data_path.exists():
        raise FileNotFoundError("The file assets/query_template.txt does not exist")
    QUERY_TEMPLATE_CONTENT = data_path.read_text()
except FileNotFoundError as e:
    # Handle the error appropriately, perhaps with a default template or a clearer startup error
    print(f"FATAL: Could not load query template. {e}")
    raise


class TalkUseCase:
    def __init__(self):
        # The chain is the core of your use case. Define it once.
        # This is a sequence of operations: prompt -> model -> output_parser
        self.chain: Runnable = (
            ChatPromptTemplate.from_template(QUERY_TEMPLATE_CONTENT)
            | ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0) # Updated model name for best practice
            | StrOutputParser()
        )

    def execute(self, query: str, relevant_chunks: List[Chunk]) -> str:
        """
        Executes the question-answering chain using the provided query and context.
        """
        context = "\n\n".join([chunk.content for chunk in relevant_chunks])

        # A simple guard clause is cleaner
        if not context:
            return "No relevant information found to answer the query. Please try rephrasing your question."

        # The chain is invoked with a dictionary matching the variables in the template
        response = self.chain.invoke({
            "context": context,
            "question": query
        })

        return response
from typing import List
from src.domain.models.chunk import Chunk
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


class TalkUseCase:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI( model = "gemini-2.5-flash", temperature = 0.0 )

    async def execute(self, query: str, relevant_chunks: List[Chunk]) -> str:
        context = "\n\n".join([chunk.content for chunk in relevant_chunks])

        if not context:
            return "No relevant information found to answer the query. Please try rephrasing your question."

        messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Answer the user's question based on the provided context. If you cannot find the answer in the context, state that you don't have enough information."
            ),
            HumanMessage(
                content=f"Context: {context}\n\nQuestion: {query}"
            ),
        ]

        response = await self.llm.invoke(messages)
        return response.content

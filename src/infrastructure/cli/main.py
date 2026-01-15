import typer
from dotenv import load_dotenv

# Import individual command modules
from src.infrastructure.cli.commands import ingest, info, talk, search, chat

app = typer.Typer(
    help="Document Chunking and Retrieval CLI - Chunk documents and interact with them via natural language.",
    add_completion=False
)

# Register commands from separate modules
app.command(name="save")(ingest.save)
app.command(name="clean")(ingest.clean_command)
app.command(name="talk")(talk.talk)
app.command(name="search")(search.search)
app.command(name="chat")(chat.chat)
app.command(name="info")(info.info)

@app.callback()
def main_callback():
    """
    Document Chunking and Retrieval CLI
    """
    load_dotenv()

if __name__ == "__main__":
    app()
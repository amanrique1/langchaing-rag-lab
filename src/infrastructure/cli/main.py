import typer
from dotenv import load_dotenv

# Import the command functions from the routers
from src.infrastructure.cli.commands import ingest, query, info

app = typer.Typer(
    help="Document Chunking and Retrieval CLI - Chunk documents and interact with them via natural language.",
    add_completion=False
)

# Register commands
# We register them manually to keep the CLI flat (e.g. "cli save" instead of "cli ingest save")
app.command(name="save")(ingest.save)
app.command(name="clean")(ingest.clean_command)
app.command(name="talk")(query.talk)
app.command(name="search")(query.search)
app.command(name="info")(info.info)

@app.callback()
def main_callback():
    """
    Document Chunking and Retrieval CLI
    """
    load_dotenv()

if __name__ == "__main__":
    app()
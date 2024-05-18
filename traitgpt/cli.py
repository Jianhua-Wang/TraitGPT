"""Console script for traitgpt."""

import logging

import typer
from rich.console import Console

from . import __version__

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(context_settings=CONTEXT_SETTINGS, add_completion=False)

console = Console()


@app.callback(invoke_without_command=True, no_args_is_help=True)
def main(
    version: bool = typer.Option(False, "--version", "-V", help="Show version."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose info."),
):
    """TraitGPT: A GPT-based tool for mapping traits to ontology terms."""
    console.rule("[bold blue]TraitGPT[/bold blue]")
    console.print(f"Version: {__version__}", justify="center")
    console.print("Author: Jianhua Wang", justify="center")
    console.print("Email: jianhua.mert@gmail.com", justify="center")
    if version:
        typer.echo(f"Version: {__version__}")
        raise typer.Exit()
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Verbose mode is on.")
    else:
        logging.getLogger().setLevel(logging.INFO)


@app.command()
def index(
    local_path: str = typer.Option(
        None,
        "--local-path",
        "-l",
        help="Local path to the vocabulary file, csv format.",
    ),
    index_path: str = typer.Option(
        None, "--index-path", "-i", help="Index path to store the vocabulary."
    ),
):
    """Index the vocabulary."""
    from traitgpt import VocabularyStore

    VocabularyStore(index_path=index_path, local_path=local_path)


@app.command()
def search(
    infile: str = typer.Argument(..., help="Input file, one trait per line."),
    outfile: str = typer.Argument(..., help="Output file. tab-separated."),
    index_path: str = typer.Option(
        None, "--index-path", "-i", help="Index path to store the vocabulary."
    ),
):
    """Search the vocabulary."""
    from traitgpt import TraitGPT

    tg = TraitGPT(index_path=index_path)
    out_write = open(outfile, "w")
    with open(infile, "r") as f:
        for line in f:
            res = tg.map_trait(line.strip())
            term_name, term_id = res.get("term_name"), res.get("term_id")
            logging.info(f"{line.strip()} -> {term_name} | {term_id}")
            out_write.write(f"{line.strip()}\t{term_name}\t{term_id}\n")
    out_write.close()


if __name__ == "__main__":
    app()

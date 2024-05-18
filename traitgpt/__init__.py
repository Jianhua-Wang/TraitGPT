"""Top-level package for TraitGPT."""
import getpass
import logging
import os

from rich.logging import RichHandler

from .traitgpt import TraitGPT, VocabularyStore

__author__ = """Jianhua Wang"""
__email__ = "jianhua.mert@gmail.com"
__version__ = "0.0.3"

logging.basicConfig(
    level=logging.WARNING,
    format="%(name)s - %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)

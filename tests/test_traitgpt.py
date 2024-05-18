"""Test cases for the traitgpt module."""

import pytest
import os
import tempfile
from unittest import mock
from unittest.mock import MagicMock

from traitgpt.traitgpt import VocabularyStore, set_openai_api_key


@pytest.fixture
def mock_openai_key_env():
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        yield


def test_set_openai_api_key_env(mock_openai_key_env):
    set_openai_api_key()
    assert os.getenv("OPENAI_API_KEY") == "test_key"


def test_set_openai_api_key_prompt(monkeypatch):
    monkeypatch.setattr("getpass.getpass", lambda _: "test_key_prompt")
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    set_openai_api_key()
    assert os.getenv("OPENAI_API_KEY") == "test_key_prompt"


def test_csv_validator():
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".csv"
    ) as temp_file:
        temp_file.write("Term,ID,Alias\nvalue1,value2,value3\n")
    assert VocabularyStore.csv_validator(temp_file.name) is True
    os.remove(temp_file.name)

    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".csv"
    ) as temp_file:
        temp_file.write("Invalid,CSV,Header\nvalue1,value2,value3\n")
    assert VocabularyStore.csv_validator(temp_file.name) is False
    os.remove(temp_file.name)


@pytest.fixture
def mock_csv_data():
    data = [
        {"Term": "term1", "ID": "1", "Alias": "alias1"},
        {"Term": "term2", "ID": "2", "Alias": "alias2"},
    ]
    return data


@pytest.fixture
def mock_faiss():
    FAISS = MagicMock()
    FAISS.load_local.return_value = FAISS
    FAISS.from_documents.return_value = FAISS
    FAISS.similarity_search.return_value = []
    return FAISS


def test_index_vocabulary_valid_csv():
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".csv"
    ) as temp_file:
        temp_file.write('"Term","ID","Alias"\n"term1","id1","alias1"\n')
        temp_file_path = temp_file.name

    assert VocabularyStore.csv_validator(temp_file_path)

    result = VocabularyStore.index_vocabulary(temp_file_path)
    assert result is not None

    os.remove(temp_file_path)

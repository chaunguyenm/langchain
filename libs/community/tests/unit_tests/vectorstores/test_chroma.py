import pytest

from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


@pytest.mark.requires("chromadb")

def test_from_documents() -> None:
    """Test end to end construction and search."""
    documents = [
        Document(page_content="foo", metadata={"a": 1}),
        Document(page_content="bar", metadata={"b": 1}),
        Document(page_content="baz", metadata={"c": 1}),
    ]
    vectorstore = Chroma.from_documents(
        collection_name="test_collection",
        documents=documents,
        embedding=FakeEmbeddings(),
    )

    assert vectorstore is not None


def test_from_texts() -> None:
    """Test end to end construction and search."""
    texts = [
        "foo",
        "bar",
        "baz",
    ]
    vectorstore = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
    )

    assert vectorstore is not None


def test_similarity_search() -> None:
    """Test end to end construction and search."""
    texts = [
        "foo",
        "bar",
        "baz",
    ]
    ids = ["a", "b", "c"]
    vectorstore = Chroma.from_texts(
        collection_name="test_collection2",
        texts=texts,
        ids=ids,
        embedding=FakeEmbeddings(),
    )

    results = vectorstore.similarity_search("foo")
    assert results[0] == Document(page_content="foo", id="a")

def test_similarity_search_with_doc() -> None:
    """Test end to end construction and search."""
    documents = [
        Document(page_content="foo", metadata={"a": 1}),
        Document(page_content="bar", metadata={"b": 1}),
        Document(page_content="baz", metadata={"c": 1}),
    ]
    ids = ["a", "b", "c"]
    vectorstore = Chroma.from_documents(
        collection_name="test_collection2",
        documents=documents,
        ids=ids,
        embedding=FakeEmbeddings(),
    )
    result = vectorstore.similarity_search(Document(page_content="foo"))
    assert result[0] == Document(page_content="foo", id="a", metadata={"a": 1})
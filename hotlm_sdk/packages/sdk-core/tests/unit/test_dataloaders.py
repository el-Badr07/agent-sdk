import csv
import json
import os

import pytest
from hotlm_core.dataloaders.csv_loader import CSVLoader
from hotlm_core.dataloaders.json_loader import JSONLoader
from hotlm_core.schema.documents import Document


def test_csv_loader(tmp_path):
    # Create a temporary CSV file
    file_path = tmp_path / "test.csv"
    rows = [{"text": "doc1", "meta": "a"}, {"text": "doc2", "meta": "b"}]
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "meta"])
        writer.writeheader()
        writer.writerows(rows)
    loader = CSVLoader(str(file_path), text_field="text")
    docs = loader.load()
    assert isinstance(docs, list)
    assert [d.page_content for d in docs] == ["doc1", "doc2"]
    # Test lazy load
    lazy_docs = list(loader.lazy_load())
    assert [d.page_content for d in lazy_docs] == ["doc1", "doc2"]


def test_json_loader(tmp_path):
    # Create temporary JSON files
    dir_path = tmp_path / "jsons"
    dir_path.mkdir()
    entries = [{"text": "j1", "foo": 1}, {"text": "j2", "foo": 2}]
    for i, data in enumerate(entries):
        file_path = dir_path / f"{i}.json"
        file_path.write_text(json.dumps(data), encoding="utf-8")
    loader = JSONLoader(str(dir_path), text_field="text")
    docs = loader.load()
    contents = {d.page_content for d in docs}
    assert contents == {"j1", "j2"}
    lazy_contents = {d.page_content for d in loader.lazy_load()}
    assert lazy_contents == {"j1", "j2"}

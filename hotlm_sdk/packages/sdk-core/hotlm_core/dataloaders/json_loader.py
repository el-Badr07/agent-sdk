import json
import os
from typing import Iterator, List

from hotlm_core.dataloaders.base import BaseDocumentLoader
from hotlm_core.schema.documents import Document


class JSONLoader(BaseDocumentLoader):
    """Loader that reads all JSON files in a directory into Document objects."""
    def __init__(self, dir_path: str, text_field: str = "text") -> None:
        """Initialize JSONLoader.

        Args:
            dir_path: Directory containing JSON files.
            text_field: Key in JSON containing document text.
        """
        self.dir_path = dir_path
        self.text_field = text_field

    def load(self) -> List[Document]:
        """Load all JSON files in the directory and return a list of Documents."""
        docs: List[Document] = []
        for fname in os.listdir(self.dir_path):
            if fname.lower().endswith('.json'):
                path = os.path.join(self.dir_path, fname)
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content = data.get(self.text_field, '')
                    metadata = {k: v for k, v in data.items() if k != self.text_field}
                    docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def lazy_load(self) -> Iterator[Document]:
        """Yield Document objects one by one from JSON files."""
        for fname in os.listdir(self.dir_path):
            if fname.lower().endswith('.json'):
                path = os.path.join(self.dir_path, fname)
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content = data.get(self.text_field, '')
                    metadata = {k: v for k, v in data.items() if k != self.text_field}
                    yield Document(page_content=content, metadata=metadata)
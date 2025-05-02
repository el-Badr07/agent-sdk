import csv
from typing import List

from hotlm_core.dataloaders.base import BaseDocumentLoader
from hotlm_core.schema.documents import Document


class CSVLoader(BaseDocumentLoader):
    """Loader that reads a CSV file into Document objects."""
    def __init__(self, file_path: str, text_field: str = "text") -> None:
        """Initialize CSVLoader.

        Args:
            file_path: Path to the CSV file.
            text_field: Column name containing document text.
        """
        self.file_path = file_path
        self.text_field = text_field

    def load(self) -> List[Document]:
        """Read the entire CSV and return list of Documents."""
        docs: List[Document] = []
        with open(self.file_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                content = row.get(self.text_field, "")
                metadata = {k: v for k, v in row.items() if k != self.text_field}
                docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def lazy_load(self) -> 'Iterator[Document]':
        """Yield Documents one by one from the CSV."""
        with open(self.file_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                content = row.get(self.text_field, "")
                metadata = {k: v for k, v in row.items() if k != self.text_field}
                yield Document(page_content=content, metadata=metadata)
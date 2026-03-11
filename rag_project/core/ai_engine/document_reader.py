import os
from django.conf import settings
from pypdf import PdfReader
import docx


class DocumentReader:

    def __init__(self):
        self.folder = os.path.join(settings.BASE_DIR, "AI_Data")

    def load_documents(self):

        documents = []

        for file in os.listdir(self.folder):

            file_path = os.path.join(self.folder, file)

            if file.endswith(".txt"):
                documents.append(self.read_txt(file_path))

            elif file.endswith(".pdf"):
                documents.append(self.read_pdf(file_path))

            elif file.endswith(".docx"):
                documents.append(self.read_docx(file_path))

        return documents


    def read_txt(self, path):

        with open(path, "r", encoding="utf-8") as f:
            return f.read()


    def read_pdf(self, path):

        reader = PdfReader(path)

        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        return text


    def read_docx(self, path):

        doc = docx.Document(path)

        text = ""

        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"

        return text

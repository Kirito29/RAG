import os
from django.conf import settings
import docx
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class DocumentReader:

    def __init__(self):
        self.folder = os.path.join(settings.BASE_DIR, "AI_Data")

    def load_documents(self):

        documents = []

        for file in os.listdir(self.folder):

            # ignore temp and hidden files
            if file.startswith("~") or file.startswith("."):
                continue

            file_path = os.path.join(self.folder, file)

            if file.endswith(".txt"):
                documents.append({
                    "text": self.read_txt(file_path),
                    "source": file
                })

            elif file.endswith(".pdf"):
                documents.append({
                    "text": self.read_pdf(file_path),
                    "source": file
                })

            elif file.endswith(".docx"):
                documents.append({
                    "text": self.read_docx(file_path),
                    "source": file
                })

            elif file.endswith((".png", ".jpg", ".jpeg")):
                documents.append({
                    "text": self.read_image(file_path),
                    "source": file
            })

        return documents


    def read_txt(self, path):

        with open(path, "r", encoding="utf-8") as f:
            return f.read()


    def read_pdf(self, path):

        text = ""
        doc = fitz.open(path)
    
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text += page_text + "\n"
            else:
                # OCR fallback for scanned pages
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    text += pytesseract.image_to_string(image) + "\n"
    
        return text



    def read_docx(self, path):

        doc = docx.Document(path)

        text = ""

        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"

        return text


    def read_image(self, path):

        image = Image.open(path)

        return f"[Image file detected: {os.path.basename(path)}]"

import os, base64, glob
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

PROMPT = """Convert this PDF page into clean Markdown.
- Reproduce all text faithfully (keep Tamil text as Tamil).
- Render every table as a Markdown table with all rows and columns. Repeat column headers if the table looks continued from a previous page.
- For each chart/graph, write a short description and list the data values, labels and units you can read.
- Do not add anything that isn't on the page."""

def page_to_markdown(page):
    pix = page.get_pixmap(dpi=150)
    b64 = base64.b64encode(pix.tobytes("png")).decode()
    msg = HumanMessage(content=[
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ])
    return llm.invoke([msg]).content

def pdf_to_documents(pdf_path):
    docs = []
    pdf = fitz.open(pdf_path)
    name = os.path.basename(pdf_path)
    for i, page in enumerate(pdf):
        text = page.get_text().strip()
        if len(text) > 200 and not page.get_images() and not page.find_tables().tables:
            content = text
        else:
            content = page_to_markdown(page)
        if content.strip():
            docs.append(Document(page_content=content,
                                 metadata={"source": name, "page": i + 1}))
        print(f"{name}: page {i + 1}/{len(pdf)} done")
    return docs

if __name__ == "__main__":
    all_docs = []
    for path in sorted(glob.glob("data/*.pdf")):
        all_docs.extend(pdf_to_documents(path))

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    print(f"{len(all_docs)} pages -> {len(chunks)} chunks")

    vs = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
    vs.save_local("faiss_index")
    print("Saved to faiss_index/")
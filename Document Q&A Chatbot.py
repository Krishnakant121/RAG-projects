import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, TextLoader
from langchain_core.prompts import PromptTemplate
import time
from pathlib import Path


load_dotenv()

# loading
def load_document_by_extension(file_path: str):
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    elif ext in [".html", ".htm"]:
        return UnstructuredHTMLLoader(str(path)).load()
    elif ext in [".md", ".txt"]:
        return TextLoader(str(path), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# Load documents using our function
file_path = r"C:\Users\HP\Downloads\Krishnakant_Narware_Python.pdf"
documents = load_document_by_extension(file_path)


# split into small chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)




chunks = splitter.split_documents(documents)

# embedding
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

#4 stored in vector store
vector_store = FAISS.from_documents(chunks,embedding)


# user side things
# retriving results
retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 4}
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

template = PromptTemplate(
    template='''
    you are a helpful assistant.
    answer only from the provided transcript context.
    if the context is insufficient , just say you don't know.
    
    {context}
    question : {question}
    ''',
    input_variables = ['context','question']

)

question = input("ask your question : ")

start_time = time.time()

retriever_docs = retriever.invoke(question)
context_text  = "\n\n".join(doc.page_content for doc in retriever_docs)

final_prompt = template.invoke({'context':context_text, 'question': question})

answer = llm.invoke(final_prompt)

end_time = time.time()

# Calculate latency in milliseconds
latency_ms = round((end_time - start_time) * 1000, 2)
#print('latency_ms :',latency_ms)

# token usage
response_metadata = getattr(answer, "response_metadata", {})
token_usage = response_metadata.get("token_usage", {})

print(len(retriever_docs))
print(answer.content)
print('latency_ms :',latency_ms)
print('token usage : ',token_usage)

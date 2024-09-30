import json
import re
from io import BytesIO
from typing import List
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
load_dotenv()
import csv
openai_apikey = os.environ.get("APIKEY")

#! --------------------------- preprocessing data from pdf file ------------------------------
def parse_pdf(file: BytesIO) -> List[str]:
    '''
    preprocessing file pdf.
    input: pdf file path
    
    return: list of string
    '''
    pdf = PdfReader(file) #! read content from pdf
    output = []
    #print(pdf.pages) # pdf.pages will result a list of pages type
    for page in pdf.pages:
        text = page.extract_text() #! get text in each page
        # Merge word which contant dash in the middle. Ex: a-b
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output

def text_to_docs(text: str) -> List[Document]:
    """
    Converts a string or list of strings to a list of Documents
    with metadata.
    """
    
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i})
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks

#! 
def test_embed(page):
    embedding = OpenAIEmbeddings(openai_api_key=openai_apikey)
    # Indexing
    #! safe page to vector database
    index = FAISS.from_documents(page,embedding)
    return index

def generate_question(context: str) -> str:
    """
    Generates a question based on the provided context.
    """
    llm = ChatOpenAI(openai_api_key=openai_apikey)

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["context"],
        template="Based on the following context, generate a relevant question for a primary school student:\n\n{context}\n\n"
    )

    # Create the LLMChain
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )
    
    # Run the chain with the context
    question = chain.run({"context": context})
    
    return question
# Function to load input JSON data from a file
def load_input_json(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to save generated questions to a JSON file
def save_to_json(data: list, output_file_path: str):
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)


# Function to save generated questions to a CSV file
def save_to_csv(data: list, output_file_path: str):
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["question", "answer", "contexts"])
        
        # Write each question, answer, and contexts
        for item in data:
            # Join contexts into a single string
            contexts_joined = " | ".join(item["contexts"])  # You can choose a different delimiter if needed
            writer.writerow([item["question"], item["answer"], contexts_joined])


def ChatBot(pathToPDF: str, numberOfQuestion: int):
    uploaded_file = pathToPDF
    
    # Parse the PDF and convert text to documents
    doc = parse_pdf(uploaded_file)
    pages = text_to_docs(doc)
    
    # Test the embeddings and save the index in a vector database
    index = test_embed(pages)
    
    # Set up the question-answering system
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_apikey),
        chain_type="map_reduce",
        retriever=index.as_retriever(),
        return_source_documents=True
    )
    
    # Set up the conversational agent
    tools = [
        Tool(
            name="Personalized QA Chat System",
            func=qa.run,
            description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
        )
    ]
    
    prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available. 
                You have access to a single tool:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""
    
    generated_questions = []
    for i in range(numberOfQuestion):
        # Retrieve a random context to generate a question from
        random_page = pages[i % len(pages)]
        context = random_page.page_content
        
        # Generate the question based on the context
        question = generate_question(context)
        
        # Use the chatbot to answer the generated question
        result = qa({"query": question})
        answer = result['result']
        source_documents = result['source_documents']
        
        contexts = [doc.page_content for doc in source_documents]
        
        generated_questions.append({
            "question": question,
            "answer": answer,
            "contexts": contexts
        })
    option = input("Enter type of data you want to save (csv/json) or if you dont want to save data just press enter: ")
    ch = 1
    while(ch):
        if (option == 'json'):
            save_to_json(generated_questions,"generateQuestion.json")
            ch = 0
        if (option == 'csv'):
            save_to_csv(generated_questions, "csvGenerateQuestion.csv")
            ch = 0
        

    #! return a list of dictionary!
    return generated_questions
   

    
print(ChatBot("../../pdfData/data.pdf", 20))
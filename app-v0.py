from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import os
from typing import Optional, List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

app = FastAPI(title="Chat Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQLAlchemy setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./chat_history.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Model
class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    session_id: str

class MessageModel(BaseModel):
    role: str
    content: str

class SessionHistoryResponse(BaseModel):
    messages: List[MessageModel]

    class Config:
        from_attributes = True

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize LLM components
model_local = ChatOllama(model="llama3")
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)

# Document processing functions
def load_and_split_documents(directory_path="docs"):
    loader = DirectoryLoader(directory_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=7500, chunk_overlap=100
    )
    return text_splitter.split_documents(documents)

def get_vectorstore(doc_splits):
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_name="rag-chroma"
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=embedding_model,
            collection_name="rag-chroma",
            persist_directory=persist_directory
        )
        vectorstore.persist()
    return vectorstore

# Initialize documents and vectorstore
doc_splits = load_and_split_documents()
vectorstore = get_vectorstore(doc_splits)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Chat template
template = """You are an AI assistant with expertise in the following documents. Your goal is to provide accurate and helpful answers based strictly on the provided context. Do not include information that isn't present in the context. If you don't know the answer, politely say so.

Context:
{context}

Conversation History:
{chat_history}

Question: {question}

Instructions:
- Provide clear and concise answers in English.
- Cite the source document when relevant.
- Use a friendly and professional tone.
- Do not use external information not included in the context.

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_chat_history(messages):
    return "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in messages])

def combine_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Database operations
def save_message(db: Session, session_id: str, role: str, content: str):
    db_message = Message(session_id=session_id, role=role, content=content)
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message

def get_session_messages(db: Session, session_id: str):
    return db.query(Message).filter(Message.session_id == session_id).order_by(Message.id.asc()).all()

# API endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        # Get conversation history
        messages = get_session_messages(db, request.session_id)
        
        # Create chain
        chain = {
            "context": retriever | combine_documents,
            "chat_history": lambda x: format_chat_history(messages),
            "question": RunnablePassthrough()
        } | prompt | model_local | StrOutputParser()
        
        # Generate response
        response = chain.invoke(request.message)
        
        # Save messages to database
        save_message(db, request.session_id, "human", request.message)
        save_message(db, request.session_id, "ai", response)
        
        return ChatResponse(response=response, session_id=request.session_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{session_id}", response_model=SessionHistoryResponse)
async def get_history(session_id: str, db: Session = Depends(get_db)):
    try:
        messages = get_session_messages(db, session_id)
        return SessionHistoryResponse(messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
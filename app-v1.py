from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
from typing import Optional, List, Dict
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.text_splitter import CharacterTextSplitter

# JWT Settings
SECRET_KEY = "your-secret-key-keep-it-secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database Setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./chat_app.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Database Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    messages = relationship("Message", back_populates="user")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    role = Column(String)
    content = Column(Text)
    user = relationship("User", back_populates="messages")

# Create tables
Base.metadata.create_all(bind=engine)

# Initialize LLM components
model_local = ChatOllama(model="deepseek-coder-v2")
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)

# Pydantic Models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    id: int
    
    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Helper Functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

# Chat Functions
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

def get_chat_messages(db: Session, user_id: int):
    return db.query(Message).filter(Message.user_id == user_id).order_by(Message.id.asc()).all()

# FastAPI App
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/signup", response_model=UserInDB)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Get user's chat history
        messages = get_chat_messages(db, current_user.id)
        
        # Create chain
        chain = {
            "context": retriever | combine_documents,
            "chat_history": lambda x: format_chat_history(messages),
            "question": RunnablePassthrough()
        } | prompt | model_local | StrOutputParser()
        
        # Generate response
        response = chain.invoke(request.message)
        
        # Save messages to database
        db_message = Message(
            session_id=f"user_{current_user.id}",
            user_id=current_user.id,
            role="user",
            content=request.message
        )
        db.add(db_message)
        
        db_response = Message(
            session_id=f"user_{current_user.id}",
            user_id=current_user.id,
            role="assistant",
            content=response
        )
        db.add(db_response)
        db.commit()
        
        return ChatResponse(response=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history", response_model=List[Dict[str, str]])
async def get_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    messages = get_chat_messages(db, current_user.id)
    return [{"role": msg.role, "content": msg.content} for msg in messages]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
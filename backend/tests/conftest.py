"""
Pytest configuration and fixtures for RAG system tests
"""
import pytest
import tempfile
import os
import sys
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Add backend directory to Python path for imports
backend_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_dir)

from config import Config
from vector_store import VectorStore, SearchResults
from ai_generator import AIGenerator
from search_tools import CourseSearchTool, ToolManager
from models import Course, Lesson, CourseChunk
from rag_system import RAGSystem


@pytest.fixture
def temp_chroma_path():
    """Create a temporary directory for ChromaDB during tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def test_config(temp_chroma_path):
    """Create a test configuration"""
    config = Config()
    config.CHROMA_PATH = temp_chroma_path
    config.MAX_RESULTS = 5  # Fix the critical bug for tests
    config.ANTHROPIC_API_KEY = "test-key"
    return config


@pytest.fixture
def test_config_with_chroma():
    """Create a test configuration with temporary ChromaDB"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = Config()
        config.CHROMA_PATH = temp_dir
        config.MAX_RESULTS = 5  # Fix the critical bug
        config.ANTHROPIC_API_KEY = "test-key-12345"
        config.CHUNK_SIZE = 200  # Smaller for tests
        config.CHUNK_OVERLAP = 50
        config.MAX_HISTORY = 2
        yield config


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for isolated testing"""
    mock_store = Mock(spec=VectorStore)
    return mock_store


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Introduction to Machine Learning",
        instructor="Dr. Jane Smith",
        course_link="https://example.com/ml-course",
        lessons=[
            Lesson(lesson_number=1, title="What is Machine Learning?", lesson_link="https://example.com/ml-course/lesson1"),
            Lesson(lesson_number=2, title="Data Preprocessing", lesson_link="https://example.com/ml-course/lesson2"),
            Lesson(lesson_number=3, title="Linear Regression", lesson_link="https://example.com/ml-course/lesson3")
        ]
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Machine learning is a method of data analysis that automates analytical model building.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Data preprocessing involves cleaning and preparing raw data for machine learning algorithms.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Linear regression is a linear approach to modeling the relationship between variables.",
            course_title=sample_course.title,
            lesson_number=3,
            chunk_index=2
        )
    ]


@pytest.fixture
def sample_search_results():
    """Create sample search results for testing"""
    return SearchResults(
        documents=[
            "Machine learning is a method of data analysis that automates analytical model building.",
            "Data preprocessing involves cleaning and preparing raw data for machine learning algorithms."
        ],
        metadata=[
            {"course_title": "Introduction to Machine Learning", "lesson_number": 1},
            {"course_title": "Introduction to Machine Learning", "lesson_number": 2}
        ],
        distances=[0.1, 0.2]
    )


@pytest.fixture
def empty_search_results():
    """Create empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )


@pytest.fixture
def error_search_results():
    """Create error search results for testing"""
    return SearchResults.empty("Test error message")


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for AI generator testing"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "Test response"
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_tool_manager():
    """Create a mock tool manager for testing"""
    mock_manager = Mock(spec=ToolManager)
    mock_manager.get_tool_definitions.return_value = []
    mock_manager.execute_tool.return_value = "Tool executed successfully"
    mock_manager.get_last_sources.return_value = []
    return mock_manager


@pytest.fixture
def sample_document_content():
    """Create sample course document content"""
    return """Course Title: Introduction to Machine Learning
Course Link: https://example.com/ml-course
Course Instructor: Dr. Jane Smith

Lesson 1: What is Machine Learning?
Lesson Link: https://example.com/ml-course/lesson1

Machine learning is a method of data analysis that automates analytical model building. 
It is a branch of artificial intelligence based on the idea that systems can learn from data, 
identify patterns and make decisions with minimal human intervention.

Lesson 2: Data Preprocessing  
Lesson Link: https://example.com/ml-course/lesson2

Data preprocessing involves cleaning and preparing raw data for machine learning algorithms. 
This includes handling missing values, removing outliers, and normalizing data formats.

Lesson 3: Linear Regression
Lesson Link: https://example.com/ml-course/lesson3

Linear regression is a linear approach to modeling the relationship between a scalar response
and one or more explanatory variables. It is one of the most fundamental algorithms in machine learning.
"""


@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for API testing"""
    mock_rag = Mock(spec=RAGSystem)
    mock_rag.query.return_value = (
        "This is a test response about machine learning.",
        [{"text": "Sample source text", "link": "https://example.com/source"}]
    )
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Introduction to Machine Learning", "Advanced Python"]
    }
    mock_rag.session_manager = Mock()
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    mock_rag.session_manager.clear_session.return_value = None
    return mock_rag


@pytest.fixture
def test_app():
    """Create a test FastAPI app without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Union

    # Import the Pydantic models from app.py
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class Source(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Union[str, Source]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    class NewChatRequest(BaseModel):
        current_session_id: Optional[str] = None

    class NewChatResponse(BaseModel):
        session_id: str

    app = FastAPI(title="Course Materials RAG System - Test", root_path="")

    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    return app, QueryRequest, QueryResponse, CourseStats, NewChatRequest, NewChatResponse, Source


@pytest.fixture
def client_with_mock_rag(test_app, mock_rag_system):
    """Create a test client with mocked RAG system"""
    app, QueryRequest, QueryResponse, CourseStats, NewChatRequest, NewChatResponse, Source = test_app

    # Add API endpoints with mocked RAG system
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            formatted_sources = []
            for source in sources:
                if isinstance(source, dict) and 'text' in source:
                    formatted_sources.append(Source(text=source['text'], link=source.get('link')))
                else:
                    formatted_sources.append(str(source))

            return QueryResponse(
                answer=answer,
                sources=formatted_sources,
                session_id=session_id
            )
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/new-chat", response_model=NewChatResponse)
    async def start_new_chat(request: NewChatRequest):
        try:
            if request.current_session_id:
                mock_rag_system.session_manager.clear_session(request.current_session_id)

            new_session_id = mock_rag_system.session_manager.create_session()
            return NewChatResponse(session_id=new_session_id)
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))

    # Add a simple root endpoint for testing
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System - Test"}

    return TestClient(app)


@pytest.fixture
def sample_query_request():
    """Sample query request for API testing"""
    return {
        "query": "What is machine learning?",
        "session_id": "test-session-123"
    }


@pytest.fixture
def sample_new_chat_request():
    """Sample new chat request for API testing"""
    return {
        "current_session_id": "old-session-456"
    }
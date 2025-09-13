"""
Pytest configuration and fixtures for RAG system tests
"""
import pytest
import tempfile
import os
import sys
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

# Add backend directory to Python path for imports
backend_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_dir)

from config import Config
from vector_store import VectorStore, SearchResults
from ai_generator import AIGenerator
from search_tools import CourseSearchTool, ToolManager
from models import Course, Lesson, CourseChunk


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
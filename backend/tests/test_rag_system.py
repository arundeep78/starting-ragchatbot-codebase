"""
Integration tests for RAGSystem - end-to-end functionality
"""

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
from config import Config
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem


class TestRAGSystemIntegration:
    """Integration tests for the complete RAG system"""

    @pytest.fixture
    def test_config_with_chroma(self):
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
    def sample_document_content(self):
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

    def test_rag_system_initialization(self, test_config_with_chroma):
        """Test RAG system initializes all components correctly"""
        rag_system = RAGSystem(test_config_with_chroma)

        # Check all components are initialized
        assert rag_system.document_processor is not None
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.session_manager is not None
        assert rag_system.tool_manager is not None
        assert rag_system.search_tool is not None
        assert rag_system.outline_tool is not None

        # Check tool registration
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_add_course_document_success(
        self, test_config_with_chroma, sample_document_content
    ):
        """Test adding a single course document successfully"""
        rag_system = RAGSystem(test_config_with_chroma)

        # Create a temporary file with course content
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as temp_file:
            temp_file.write(sample_document_content)
            temp_file_path = temp_file.name

        try:
            # Add the document
            course, chunk_count = rag_system.add_course_document(temp_file_path)

            # Verify results
            assert course is not None
            assert course.title == "Introduction to Machine Learning"
            assert course.instructor == "Dr. Jane Smith"
            assert course.course_link == "https://example.com/ml-course"
            assert len(course.lessons) == 3
            assert chunk_count > 0

            # Check lessons
            assert course.lessons[0].lesson_number == 1
            assert course.lessons[0].title == "What is Machine Learning?"
            assert (
                course.lessons[0].lesson_link == "https://example.com/ml-course/lesson1"
            )

        finally:
            os.unlink(temp_file_path)

    def test_add_course_folder_success(
        self, test_config_with_chroma, sample_document_content
    ):
        """Test adding course documents from a folder"""
        rag_system = RAGSystem(test_config_with_chroma)

        # Create temporary directory with course files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create two course files
            course1_path = os.path.join(temp_dir, "course1.txt")
            course2_content = sample_document_content.replace(
                "Introduction to Machine Learning", "Advanced Machine Learning"
            ).replace("Dr. Jane Smith", "Dr. John Doe")

            course2_path = os.path.join(temp_dir, "course2.txt")

            with open(course1_path, "w") as f:
                f.write(sample_document_content)

            with open(course2_path, "w") as f:
                f.write(course2_content)

            # Add folder
            total_courses, total_chunks = rag_system.add_course_folder(temp_dir)

            # Verify results
            assert total_courses == 2
            assert total_chunks > 0

            # Check analytics
            analytics = rag_system.get_course_analytics()
            assert analytics["total_courses"] == 2
            assert "Introduction to Machine Learning" in analytics["course_titles"]
            assert "Advanced Machine Learning" in analytics["course_titles"]

    def test_add_course_folder_skip_existing(
        self, test_config_with_chroma, sample_document_content
    ):
        """Test that adding the same folder twice skips existing courses"""
        rag_system = RAGSystem(test_config_with_chroma)

        with tempfile.TemporaryDirectory() as temp_dir:
            course_path = os.path.join(temp_dir, "course.txt")
            with open(course_path, "w") as f:
                f.write(sample_document_content)

            # Add folder first time
            total_courses1, total_chunks1 = rag_system.add_course_folder(temp_dir)
            assert total_courses1 == 1

            # Add folder second time - should skip existing
            total_courses2, total_chunks2 = rag_system.add_course_folder(temp_dir)
            assert total_courses2 == 0  # No new courses added
            assert total_chunks2 == 0  # No new chunks added

    @patch("ai_generator.anthropic")
    def test_query_without_tools(self, mock_anthropic, test_config_with_chroma):
        """Test query processing when Claude doesn't use tools"""
        # Setup mock Claude response (no tool use)
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = (
            "This is a general response about machine learning concepts."
        )
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        rag_system = RAGSystem(test_config_with_chroma)

        # Test query
        response, sources = rag_system.query("What is AI in general?")

        assert response == "This is a general response about machine learning concepts."
        assert sources == []  # No sources since no tools were used

    @patch("ai_generator.anthropic")
    def test_query_with_tool_use(
        self, mock_anthropic, test_config_with_chroma, sample_document_content
    ):
        """Test end-to-end query processing with tool use"""
        # Setup Claude to use tools
        mock_client = Mock()

        # Mock tool use response
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "machine learning definition"}
        mock_tool_content.id = "tool_123"

        tool_use_response = Mock()
        tool_use_response.content = [mock_tool_content]
        tool_use_response.stop_reason = "tool_use"

        # Mock final response
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = (
            "Based on the course materials, machine learning is a method of data analysis that automates model building."
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]
        mock_anthropic.Anthropic.return_value = mock_client

        rag_system = RAGSystem(test_config_with_chroma)

        # Add course content first
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as temp_file:
            temp_file.write(sample_document_content)
            temp_file_path = temp_file.name

        try:
            rag_system.add_course_document(temp_file_path)

            # Test query
            response, sources = rag_system.query("What is machine learning?")

            assert "Based on the course materials" in response
            assert len(sources) > 0  # Should have sources from search

        finally:
            os.unlink(temp_file_path)

    @patch("ai_generator.anthropic")
    def test_query_with_session_history(self, mock_anthropic, test_config_with_chroma):
        """Test query processing with conversation history"""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Follow-up response with context"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        rag_system = RAGSystem(test_config_with_chroma)

        # First query to establish session
        session_id = rag_system.session_manager.create_session()
        rag_system.query("What is machine learning?", session_id)

        # Second query with history
        response, sources = rag_system.query("Tell me more about that", session_id)

        # Check that history was passed to AI generator
        call_args = mock_client.messages.create.call_args[1]
        system_content = call_args["system"]
        assert "Previous conversation:" in system_content

        assert response == "Follow-up response with context"

    def test_query_processing_with_no_api_key(self, test_config_with_chroma):
        """Test that system fails gracefully without API key"""
        test_config_with_chroma.ANTHROPIC_API_KEY = ""

        rag_system = RAGSystem(test_config_with_chroma)

        # This should not crash, but may fail when trying to call API
        # We're testing the system doesn't crash on initialization
        assert rag_system.ai_generator is not None

    def test_empty_folder_processing(self, test_config_with_chroma):
        """Test processing empty or non-existent folders"""
        rag_system = RAGSystem(test_config_with_chroma)

        # Test non-existent folder
        courses, chunks = rag_system.add_course_folder("/non/existent/path")
        assert courses == 0
        assert chunks == 0

        # Test empty folder
        with tempfile.TemporaryDirectory() as temp_dir:
            courses, chunks = rag_system.add_course_folder(temp_dir)
            assert courses == 0
            assert chunks == 0


class TestRAGSystemWithRealVectorStore:
    """Tests with real vector store to catch actual database issues"""

    @pytest.fixture
    def rag_with_real_db(self, test_config_with_chroma):
        """Create RAG system with real ChromaDB for testing database operations"""
        return RAGSystem(test_config_with_chroma)

    def test_search_empty_vector_store(self, rag_with_real_db):
        """Test search behavior with empty vector store"""
        # Search tool should handle empty store gracefully
        search_tool = rag_with_real_db.search_tool
        result = search_tool.execute(query="machine learning")

        # Should return "no content found" message, not crash
        assert "No relevant content found" in result

    def test_course_outline_empty_store(self, rag_with_real_db):
        """Test course outline tool with empty store"""
        outline_tool = rag_with_real_db.outline_tool
        result = outline_tool.execute(course_name="Nonexistent Course")

        assert "No course found matching" in result

    def test_vector_store_search_with_real_data(
        self, rag_with_real_db, sample_document_content
    ):
        """Test vector store search with actual data"""
        # Add real document
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as temp_file:
            temp_file.write(sample_document_content)
            temp_file_path = temp_file.name

        try:
            rag_with_real_db.add_course_document(temp_file_path)

            # Test direct vector store search
            results = rag_with_real_db.vector_store.search("machine learning")

            # Should find content if MAX_RESULTS is not 0
            if rag_with_real_db.config.MAX_RESULTS > 0:
                assert not results.is_empty()
                assert len(results.documents) > 0
                assert "machine learning" in results.documents[0].lower()
            else:
                # This would be the bug case
                assert results.is_empty()

        finally:
            os.unlink(temp_file_path)


class TestRAGSystemErrorScenarios:
    """Test error handling in various scenarios"""

    def test_malformed_document_processing(self, test_config_with_chroma):
        """Test processing malformed documents"""
        rag_system = RAGSystem(test_config_with_chroma)

        # Create malformed document (missing required headers)
        malformed_content = "Just some random text without proper course structure."

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as temp_file:
            temp_file.write(malformed_content)
            temp_file_path = temp_file.name

        try:
            course, chunk_count = rag_system.add_course_document(temp_file_path)

            # Should handle gracefully (return None or minimal course)
            if course is None:
                assert chunk_count == 0
            else:
                # If it processes successfully, verify it doesn't crash
                assert isinstance(chunk_count, int)

        finally:
            os.unlink(temp_file_path)

    @patch("ai_generator.anthropic")
    def test_api_error_handling(self, mock_anthropic, test_config_with_chroma):
        """Test handling of API errors during query processing"""
        # Setup mock to raise exception
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API connection failed")
        mock_anthropic.Anthropic.return_value = mock_client

        rag_system = RAGSystem(test_config_with_chroma)

        # With sequential tool calling, API errors are handled gracefully
        response, sources = rag_system.query("Test query")

        assert (
            "An error occurred during tool execution: API connection failed" in response
        )
        assert sources == []

    def test_invalid_file_processing(self, test_config_with_chroma):
        """Test processing invalid or corrupted files"""
        rag_system = RAGSystem(test_config_with_chroma)

        # Test processing non-existent file
        course, chunk_count = rag_system.add_course_document("/non/existent/file.txt")
        assert course is None
        assert chunk_count == 0

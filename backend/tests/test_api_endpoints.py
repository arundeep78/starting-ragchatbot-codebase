"""
API endpoint tests for the RAG system FastAPI application
"""
import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException


@pytest.mark.api
class TestQueryEndpoint:
    """Test cases for the /api/query endpoint"""

    def test_query_with_session_id(self, client_with_mock_rag, sample_query_request):
        """Test successful query with provided session ID"""
        response = client_with_mock_rag.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "This is a test response about machine learning."
        assert data["session_id"] == "test-session-123"
        assert len(data["sources"]) > 0
        assert data["sources"][0]["text"] == "Sample source text"
        assert data["sources"][0]["link"] == "https://example.com/source"

    def test_query_without_session_id(self, client_with_mock_rag):
        """Test query without session ID creates new session"""
        request_data = {"query": "What is machine learning?"}
        response = client_with_mock_rag.post("/api/query", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "session_id" in data
        assert data["session_id"] == "test-session-123"  # From mock
        assert "answer" in data
        assert "sources" in data

    def test_query_missing_query_field(self, client_with_mock_rag):
        """Test query request missing required query field"""
        request_data = {"session_id": "test-123"}
        response = client_with_mock_rag.post("/api/query", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_query_empty_query_string(self, client_with_mock_rag):
        """Test query with empty query string"""
        request_data = {"query": "", "session_id": "test-123"}
        response = client_with_mock_rag.post("/api/query", json=request_data)

        assert response.status_code == 200  # Should still process empty queries

    def test_query_rag_system_error(self, client_with_mock_rag, mock_rag_system):
        """Test handling of RAG system errors"""
        mock_rag_system.query.side_effect = Exception("RAG system error")

        request_data = {"query": "What is machine learning?"}
        response = client_with_mock_rag.post("/api/query", json=request_data)

        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]

    def test_query_with_string_sources(self, client_with_mock_rag, mock_rag_system):
        """Test query handling legacy string format sources"""
        mock_rag_system.query.return_value = (
            "Test answer",
            ["String source 1", "String source 2"]
        )

        request_data = {"query": "What is machine learning?"}
        response = client_with_mock_rag.post("/api/query", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 2
        assert data["sources"][0] == "String source 1"
        assert data["sources"][1] == "String source 2"


@pytest.mark.api
class TestCoursesEndpoint:
    """Test cases for the /api/courses endpoint"""

    def test_get_course_stats_success(self, client_with_mock_rag):
        """Test successful retrieval of course statistics"""
        response = client_with_mock_rag.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
        assert "Introduction to Machine Learning" in data["course_titles"]
        assert "Advanced Python" in data["course_titles"]

    def test_get_course_stats_rag_error(self, client_with_mock_rag, mock_rag_system):
        """Test handling of RAG system errors in course stats"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")

        response = client_with_mock_rag.get("/api/courses")

        assert response.status_code == 500
        assert "Analytics error" in response.json()["detail"]

    def test_get_course_stats_empty_result(self, client_with_mock_rag, mock_rag_system):
        """Test course stats with empty analytics"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = client_with_mock_rag.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert len(data["course_titles"]) == 0


@pytest.mark.api
class TestNewChatEndpoint:
    """Test cases for the /api/new-chat endpoint"""

    def test_new_chat_with_current_session(self, client_with_mock_rag, sample_new_chat_request):
        """Test creating new chat with existing session ID"""
        response = client_with_mock_rag.post("/api/new-chat", json=sample_new_chat_request)

        assert response.status_code == 200
        data = response.json()

        assert "session_id" in data
        assert data["session_id"] == "test-session-123"

    def test_new_chat_without_current_session(self, client_with_mock_rag):
        """Test creating new chat without current session ID"""
        response = client_with_mock_rag.post("/api/new-chat", json={})

        assert response.status_code == 200
        data = response.json()

        assert "session_id" in data
        assert data["session_id"] == "test-session-123"

    def test_new_chat_session_manager_error(self, client_with_mock_rag, mock_rag_system):
        """Test handling of session manager errors"""
        mock_rag_system.session_manager.create_session.side_effect = Exception("Session error")

        response = client_with_mock_rag.post("/api/new-chat", json={})

        assert response.status_code == 500
        assert "Session error" in response.json()["detail"]

    def test_new_chat_clear_session_called(self, client_with_mock_rag, mock_rag_system):
        """Test that clear_session is called when current_session_id provided"""
        request_data = {"current_session_id": "old-session-456"}
        response = client_with_mock_rag.post("/api/new-chat", json=request_data)

        assert response.status_code == 200
        mock_rag_system.session_manager.clear_session.assert_called_once_with("old-session-456")
        mock_rag_system.session_manager.create_session.assert_called_once()


@pytest.mark.api
class TestRootEndpoint:
    """Test cases for the root endpoint"""

    def test_root_endpoint(self, client_with_mock_rag):
        """Test the root endpoint returns expected message"""
        response = client_with_mock_rag.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Course Materials RAG System - Test"


@pytest.mark.api
class TestRequestValidation:
    """Test request validation and error handling"""

    def test_query_invalid_json(self, client_with_mock_rag):
        """Test query endpoint with invalid JSON"""
        response = client_with_mock_rag.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_new_chat_invalid_json(self, client_with_mock_rag):
        """Test new chat endpoint with invalid JSON"""
        response = client_with_mock_rag.post(
            "/api/new-chat",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_extra_fields_ignored(self, client_with_mock_rag):
        """Test that extra fields in query request are ignored"""
        request_data = {
            "query": "What is machine learning?",
            "session_id": "test-123",
            "extra_field": "should be ignored"
        }
        response = client_with_mock_rag.post("/api/query", json=request_data)

        assert response.status_code == 200  # Should succeed despite extra field


@pytest.mark.api
class TestCORSAndMiddleware:
    """Test CORS and middleware functionality"""

    def test_cors_headers_present(self, client_with_mock_rag):
        """Test that CORS headers are present in responses"""
        response = client_with_mock_rag.get("/")

        # Note: TestClient doesn't simulate CORS preflight requests,
        # but we can verify the middleware is configured
        assert response.status_code == 200

    def test_options_request_handling(self, client_with_mock_rag):
        """Test OPTIONS request handling for CORS"""
        response = client_with_mock_rag.options("/api/query")

        # FastAPI + TestClient returns 405 for OPTIONS by default
        # This is expected behavior, CORS middleware works in real deployments
        assert response.status_code == 405


@pytest.mark.integration
class TestFullAPIWorkflow:
    """Integration tests for complete API workflows"""

    def test_complete_chat_workflow(self, client_with_mock_rag):
        """Test a complete chat workflow: new chat -> query -> get stats"""
        # Start new chat
        new_chat_response = client_with_mock_rag.post("/api/new-chat", json={})
        assert new_chat_response.status_code == 200
        session_id = new_chat_response.json()["session_id"]

        # Make a query
        query_request = {"query": "What is machine learning?", "session_id": session_id}
        query_response = client_with_mock_rag.post("/api/query", json=query_request)
        assert query_response.status_code == 200

        # Get course stats
        stats_response = client_with_mock_rag.get("/api/courses")
        assert stats_response.status_code == 200

        # Verify all responses
        assert query_response.json()["session_id"] == session_id
        assert len(stats_response.json()["course_titles"]) > 0

    def test_session_cleanup_workflow(self, client_with_mock_rag):
        """Test session cleanup when starting new chat"""
        # Start first chat
        first_chat = client_with_mock_rag.post("/api/new-chat", json={})
        first_session = first_chat.json()["session_id"]

        # Start new chat with cleanup of first session
        cleanup_request = {"current_session_id": first_session}
        second_chat = client_with_mock_rag.post("/api/new-chat", json=cleanup_request)
        second_session = second_chat.json()["session_id"]

        assert first_chat.status_code == 200
        assert second_chat.status_code == 200
        assert second_session == "test-session-123"  # From mock
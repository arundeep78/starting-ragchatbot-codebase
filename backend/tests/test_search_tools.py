"""
Unit tests for CourseSearchTool and ToolManager classes
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from search_tools import CourseOutlineTool, CourseSearchTool, Tool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool"""

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is properly formatted"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]

    def test_execute_successful_search(self, mock_vector_store, sample_search_results):
        """Test successful search execution with results"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="machine learning", course_name="ML Course")

        # Verify search was called with correct parameters
        mock_vector_store.search.assert_called_once_with(
            query="machine learning", course_name="ML Course", lesson_number=None
        )

        # Check result format
        assert "[Introduction to Machine Learning - Lesson 1]" in result
        assert "Machine learning is a method of data analysis" in result
        assert len(tool.last_sources) == 2
        assert (
            tool.last_sources[0]["text"]
            == "Introduction to Machine Learning - Lesson 1"
        )
        assert tool.last_sources[0]["link"] == "https://example.com/lesson1"

    def test_execute_with_lesson_filter(self, mock_vector_store, sample_search_results):
        """Test search execution with lesson number filter"""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="preprocessing", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="preprocessing", course_name=None, lesson_number=2
        )
        assert "Data preprocessing involves cleaning" in result

    def test_execute_empty_results(self, mock_vector_store, empty_search_results):
        """Test execution when no results are found"""
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="nonexistent topic", course_name="ML Course")

        assert "No relevant content found in course 'ML Course'" in result
        assert len(tool.last_sources) == 0

    def test_execute_empty_results_with_lesson_filter(
        self, mock_vector_store, empty_search_results
    ):
        """Test execution when no results found with lesson filter"""
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="topic", course_name="ML Course", lesson_number=5)

        assert "No relevant content found in course 'ML Course' in lesson 5" in result

    def test_execute_search_error(self, mock_vector_store, error_search_results):
        """Test execution when search returns an error"""
        mock_vector_store.search.return_value = error_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test query")

        assert result == "Test error message"
        assert len(tool.last_sources) == 0

    def test_execute_query_only(self, mock_vector_store, sample_search_results):
        """Test execution with only query parameter"""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="machine learning")

        mock_vector_store.search.assert_called_once_with(
            query="machine learning", course_name=None, lesson_number=None
        )
        assert "Machine learning is a method" in result

    def test_format_results_with_lesson_links(
        self, mock_vector_store, sample_search_results
    ):
        """Test result formatting includes lesson links when available"""
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/lesson1",
            "https://example.com/lesson2",
        ]

        tool = CourseSearchTool(mock_vector_store)
        tool._format_results(sample_search_results)

        # Check that lesson links were requested
        assert mock_vector_store.get_lesson_link.call_count == 2
        mock_vector_store.get_lesson_link.assert_any_call(
            "Introduction to Machine Learning", 1
        )
        mock_vector_store.get_lesson_link.assert_any_call(
            "Introduction to Machine Learning", 2
        )

        # Check sources include links
        assert tool.last_sources[0]["link"] == "https://example.com/lesson1"
        assert tool.last_sources[1]["link"] == "https://example.com/lesson2"

    def test_format_results_without_lesson_numbers(self, mock_vector_store):
        """Test result formatting when metadata has no lesson numbers"""
        results_no_lessons = SearchResults(
            documents=["Some general course content"],
            metadata=[{"course_title": "General Course"}],
            distances=[0.1],
        )

        tool = CourseSearchTool(mock_vector_store)
        formatted = tool._format_results(results_no_lessons)

        assert "[General Course]" in formatted
        assert "Some general course content" in formatted
        assert tool.last_sources[0]["text"] == "General Course"
        assert tool.last_sources[0]["link"] is None


class TestCourseOutlineTool:
    """Test cases for CourseOutlineTool"""

    def test_get_tool_definition(self, mock_vector_store):
        """Test that outline tool definition is properly formatted"""
        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert definition["input_schema"]["required"] == ["course_name"]
        assert "course_name" in definition["input_schema"]["properties"]

    def test_execute_successful_outline(self, mock_vector_store, sample_course):
        """Test successful course outline retrieval"""
        course_metadata = {
            "title": sample_course.title,
            "instructor": sample_course.instructor,
            "course_link": sample_course.course_link,
            "lessons": [
                {
                    "lesson_number": 1,
                    "lesson_title": "What is Machine Learning?",
                    "lesson_link": "https://example.com/lesson1",
                },
                {
                    "lesson_number": 2,
                    "lesson_title": "Data Preprocessing",
                    "lesson_link": "https://example.com/lesson2",
                },
            ],
        }

        mock_vector_store.get_course_metadata_by_name.return_value = course_metadata

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_name="ML")

        mock_vector_store.get_course_metadata_by_name.assert_called_once_with("ML")

        assert "Course: Introduction to Machine Learning" in result
        assert "Instructor: Dr. Jane Smith" in result
        assert "Course Link: https://example.com/ml-course" in result
        assert "Course Outline (2 lessons):" in result
        assert "Lesson 1: What is Machine Learning?" in result
        assert "Lesson 2: Data Preprocessing" in result

        # Check source tracking
        assert len(tool.last_sources) == 1
        assert (
            tool.last_sources[0]["text"]
            == "Introduction to Machine Learning - Course Outline"
        )
        assert tool.last_sources[0]["link"] == sample_course.course_link

    def test_execute_course_not_found(self, mock_vector_store):
        """Test execution when course is not found"""
        mock_vector_store.get_course_metadata_by_name.return_value = None

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_name="Nonexistent Course")

        assert "No course found matching 'Nonexistent Course'" in result

    def test_format_course_outline_minimal(self, mock_vector_store):
        """Test outline formatting with minimal metadata"""
        minimal_metadata = {"title": "Basic Course", "lessons": []}

        tool = CourseOutlineTool(mock_vector_store)
        result = tool._format_course_outline(minimal_metadata)

        assert "Course: Basic Course" in result
        assert "Instructor: Unknown Instructor" in result
        assert "Course Outline (0 lessons):" in result


class TestToolManager:
    """Test cases for ToolManager"""

    def test_register_tool(self):
        """Test tool registration"""
        manager = ToolManager()
        mock_tool = Mock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {"name": "test_tool"}

        manager.register_tool(mock_tool)

        assert "test_tool" in manager.tools
        assert manager.tools["test_tool"] == mock_tool

    def test_register_tool_without_name(self):
        """Test tool registration fails without name"""
        manager = ToolManager()
        mock_tool = Mock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {}

        with pytest.raises(
            ValueError, match="Tool must have a 'name' in its definition"
        ):
            manager.register_tool(mock_tool)

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        tool_names = [defn["name"] for defn in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_execute_tool(self, mock_vector_store, sample_search_results):
        """Test tool execution by name"""
        mock_vector_store.search.return_value = sample_search_results

        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        result = manager.execute_tool("search_course_content", query="test query")

        assert "Machine learning is a method" in result

    def test_execute_nonexistent_tool(self):
        """Test execution of non-existent tool"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test retrieving sources from last tool execution"""
        mock_vector_store.search.return_value = sample_search_results

        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        # Execute search to populate sources
        manager.execute_tool("search_course_content", query="test")

        sources = manager.get_last_sources()
        assert len(sources) > 0
        assert sources[0]["text"] == "Introduction to Machine Learning - Lesson 1"

    def test_reset_sources(self, mock_vector_store, sample_search_results):
        """Test resetting sources from all tools"""
        mock_vector_store.search.return_value = sample_search_results

        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        # Execute search to populate sources
        manager.execute_tool("search_course_content", query="test")
        assert len(manager.get_last_sources()) > 0

        # Reset sources
        manager.reset_sources()
        assert len(manager.get_last_sources()) == 0

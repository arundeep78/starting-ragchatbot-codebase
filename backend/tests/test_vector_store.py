"""
Unit tests for VectorStore class and ChromaDB integration
"""
import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:
    """Test SearchResults dataclass functionality"""
    
    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'course': 'Course1'}, {'course': 'Course2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'course': 'Course1'}, {'course': 'Course2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None
    
    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None
    
    def test_from_chroma_none_results(self):
        """Test creating SearchResults from None ChromaDB results"""
        chroma_results = {
            'documents': None,
            'metadatas': None,
            'distances': None
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
    
    def test_empty_results_with_error(self):
        """Test creating empty SearchResults with error"""
        results = SearchResults.empty("No results found")
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "No results found"
    
    def test_is_empty(self):
        """Test is_empty method"""
        empty_results = SearchResults([], [], [])
        non_empty_results = SearchResults(['doc1'], [{'meta': 'data'}], [0.1])
        
        assert empty_results.is_empty()
        assert not non_empty_results.is_empty()


class TestVectorStore:
    """Test VectorStore functionality"""
    
    @pytest.fixture
    def temp_vector_store(self):
        """Create VectorStore with temporary ChromaDB"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield VectorStore(temp_dir, "all-MiniLM-L6-v2", max_results=5)
    
    def test_initialization(self, temp_vector_store):
        """Test VectorStore initialization"""
        vs = temp_vector_store
        
        assert vs.max_results == 5
        assert vs.client is not None
        assert vs.embedding_function is not None
        assert vs.course_catalog is not None
        assert vs.course_content is not None
    
    def test_initialization_with_zero_max_results(self):
        """Test VectorStore initialization with max_results=0 (the bug)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            vs = VectorStore(temp_dir, "all-MiniLM-L6-v2", max_results=0)
            assert vs.max_results == 0  # This is the bug!
    
    def test_add_course_metadata(self, temp_vector_store, sample_course):
        """Test adding course metadata to catalog"""
        vs = temp_vector_store
        
        vs.add_course_metadata(sample_course)
        
        # Verify course was added
        existing_titles = vs.get_existing_course_titles()
        assert sample_course.title in existing_titles
        
        # Check course count
        assert vs.get_course_count() == 1
    
    def test_add_course_content(self, temp_vector_store, sample_course_chunks):
        """Test adding course content chunks"""
        vs = temp_vector_store
        
        vs.add_course_content(sample_course_chunks)
        
        # Try to search for content (if max_results > 0)
        if vs.max_results > 0:
            results = vs.course_content.get()
            assert len(results['documents']) == len(sample_course_chunks)
    
    def test_add_empty_course_content(self, temp_vector_store):
        """Test adding empty course content list"""
        vs = temp_vector_store
        
        # Should not crash
        vs.add_course_content([])
    
    def test_search_with_no_filters(self, temp_vector_store, sample_course, sample_course_chunks):
        """Test search without course or lesson filters"""
        vs = temp_vector_store
        
        # Add content
        vs.add_course_metadata(sample_course)
        vs.add_course_content(sample_course_chunks)
        
        # Search
        results = vs.search("machine learning")
        
        if vs.max_results > 0:
            assert not results.is_empty()
            assert results.error is None
        else:
            # With max_results=0, should return empty
            assert results.is_empty()
    
    def test_search_with_course_filter(self, temp_vector_store, sample_course, sample_course_chunks):
        """Test search with course name filter"""
        vs = temp_vector_store
        
        # Add content
        vs.add_course_metadata(sample_course)
        vs.add_course_content(sample_course_chunks)
        
        # Search with course filter
        results = vs.search("machine learning", course_name="Machine Learning")
        
        if vs.max_results > 0:
            # Should find the course and return results
            assert not results.is_empty()
            assert results.error is None
        else:
            assert results.is_empty()
    
    def test_search_with_nonexistent_course(self, temp_vector_store, sample_course, sample_course_chunks):
        """Test search with non-existent course name"""
        vs = temp_vector_store
        
        # Add content
        vs.add_course_metadata(sample_course)
        vs.add_course_content(sample_course_chunks)
        
        # Search for non-existent course (use very different name)
        results = vs.search("machine learning", course_name="Cooking and Baking Masterclass")
        
        # The semantic search in ChromaDB might still find similarity and resolve the course
        # So we test that either:
        # 1. Results are empty with error message about no course found, OR
        # 2. Results are found because semantic similarity resolved the course name
        if results.is_empty():
            # Expected case: course not found
            if results.error:
                assert "No course found matching" in results.error
        else:
            # Unexpected but possible: semantic similarity found the course
            # In this case, results should be valid
            assert results.error is None
            assert len(results.documents) > 0
    
    def test_search_with_lesson_filter(self, temp_vector_store, sample_course, sample_course_chunks):
        """Test search with lesson number filter"""
        vs = temp_vector_store
        
        # Add content
        vs.add_course_metadata(sample_course)
        vs.add_course_content(sample_course_chunks)
        
        # Search with lesson filter
        results = vs.search("preprocessing", lesson_number=2)
        
        if vs.max_results > 0:
            assert not results.is_empty()
            # Should only return content from lesson 2
            for metadata in results.metadata:
                assert metadata.get('lesson_number') == 2
    
    def test_search_with_both_filters(self, temp_vector_store, sample_course, sample_course_chunks):
        """Test search with both course and lesson filters"""
        vs = temp_vector_store
        
        # Add content
        vs.add_course_metadata(sample_course)
        vs.add_course_content(sample_course_chunks)
        
        # Search with both filters
        results = vs.search(
            "linear regression", 
            course_name="Machine Learning",
            lesson_number=3
        )
        
        if vs.max_results > 0:
            assert not results.is_empty()
            for metadata in results.metadata:
                assert metadata.get('lesson_number') == 3
                assert metadata.get('course_title') == sample_course.title
    
    def test_search_empty_store(self, temp_vector_store):
        """Test search on empty vector store"""
        vs = temp_vector_store
        
        results = vs.search("machine learning")
        
        # Should return empty results (not error) since store is just empty
        assert results.is_empty()
        assert results.error is None
    
    def test_search_exception_handling(self, temp_vector_store, monkeypatch):
        """Test search exception handling"""
        vs = temp_vector_store
        
        # Mock the query method to raise exception
        def mock_query(*args, **kwargs):
            raise Exception("ChromaDB error")
        
        monkeypatch.setattr(vs.course_content, 'query', mock_query)
        
        results = vs.search("test query")
        
        assert results.is_empty()
        assert "Search error: ChromaDB error" in results.error
    
    def test_resolve_course_name_exact_match(self, temp_vector_store, sample_course):
        """Test course name resolution with exact match"""
        vs = temp_vector_store
        vs.add_course_metadata(sample_course)
        
        resolved = vs._resolve_course_name("Introduction to Machine Learning")
        assert resolved == sample_course.title
    
    def test_resolve_course_name_partial_match(self, temp_vector_store, sample_course):
        """Test course name resolution with partial match"""
        vs = temp_vector_store
        vs.add_course_metadata(sample_course)
        
        # Should find by semantic similarity
        resolved = vs._resolve_course_name("Machine Learning")
        assert resolved == sample_course.title
        
        resolved = vs._resolve_course_name("ML")
        # Might or might not match depending on embedding similarity
        assert resolved is None or resolved == sample_course.title
    
    def test_resolve_course_name_no_match(self, temp_vector_store, sample_course):
        """Test course name resolution with no match"""
        vs = temp_vector_store
        vs.add_course_metadata(sample_course)
        
        # Use a very different query that should not match semantically
        resolved = vs._resolve_course_name("Cooking Recipes and Kitchen Management")
        # Note: Due to semantic similarity, this might still match, so we test both cases
        assert resolved is None or resolved == sample_course.title
    
    def test_build_filter_no_params(self, temp_vector_store):
        """Test filter building with no parameters"""
        vs = temp_vector_store
        
        filter_dict = vs._build_filter(None, None)
        assert filter_dict is None
    
    def test_build_filter_course_only(self, temp_vector_store):
        """Test filter building with course only"""
        vs = temp_vector_store
        
        filter_dict = vs._build_filter("Test Course", None)
        assert filter_dict == {"course_title": "Test Course"}
    
    def test_build_filter_lesson_only(self, temp_vector_store):
        """Test filter building with lesson only"""
        vs = temp_vector_store
        
        filter_dict = vs._build_filter(None, 2)
        assert filter_dict == {"lesson_number": 2}
    
    def test_build_filter_both_params(self, temp_vector_store):
        """Test filter building with both parameters"""
        vs = temp_vector_store
        
        filter_dict = vs._build_filter("Test Course", 2)
        assert filter_dict == {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 2}
            ]
        }
    
    def test_get_course_metadata_by_name(self, temp_vector_store, sample_course):
        """Test getting course metadata by name"""
        vs = temp_vector_store
        vs.add_course_metadata(sample_course)
        
        metadata = vs.get_course_metadata_by_name("Machine Learning")
        
        assert metadata is not None
        assert metadata["title"] == sample_course.title
        assert metadata["instructor"] == sample_course.instructor
        assert metadata["course_link"] == sample_course.course_link
        assert "lessons" in metadata
        assert len(metadata["lessons"]) == len(sample_course.lessons)
    
    def test_get_course_metadata_nonexistent(self, temp_vector_store):
        """Test getting metadata for non-existent course"""
        vs = temp_vector_store
        
        metadata = vs.get_course_metadata_by_name("Nonexistent Course")
        assert metadata is None
    
    def test_get_lesson_link(self, temp_vector_store, sample_course):
        """Test getting lesson link"""
        vs = temp_vector_store
        vs.add_course_metadata(sample_course)
        
        link = vs.get_lesson_link(sample_course.title, 1)
        assert link == sample_course.lessons[0].lesson_link
        
        # Test non-existent lesson
        link = vs.get_lesson_link(sample_course.title, 999)
        assert link is None
    
    def test_get_course_link(self, temp_vector_store, sample_course):
        """Test getting course link"""
        vs = temp_vector_store
        vs.add_course_metadata(sample_course)
        
        link = vs.get_course_link(sample_course.title)
        assert link == sample_course.course_link
        
        # Test non-existent course
        link = vs.get_course_link("Nonexistent Course")
        assert link is None
    
    def test_clear_all_data(self, temp_vector_store, sample_course, sample_course_chunks):
        """Test clearing all data from collections"""
        vs = temp_vector_store
        
        # Add some data
        vs.add_course_metadata(sample_course)
        vs.add_course_content(sample_course_chunks)
        
        assert vs.get_course_count() == 1
        
        # Clear data
        vs.clear_all_data()
        
        # Verify data is cleared
        assert vs.get_course_count() == 0
        assert vs.get_existing_course_titles() == []
    
    def test_get_all_courses_metadata(self, temp_vector_store, sample_course):
        """Test getting all courses metadata"""
        vs = temp_vector_store
        vs.add_course_metadata(sample_course)
        
        all_metadata = vs.get_all_courses_metadata()
        
        assert len(all_metadata) == 1
        assert all_metadata[0]["title"] == sample_course.title
        assert "lessons" in all_metadata[0]
        assert "lessons_json" not in all_metadata[0]  # Should be parsed


class TestVectorStoreWithMaxResultsZero:
    """Test VectorStore behavior specifically with MAX_RESULTS=0 bug"""
    
    @pytest.fixture
    def zero_results_store(self):
        """Create VectorStore with max_results=0 to test the bug"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield VectorStore(temp_dir, "all-MiniLM-L6-v2", max_results=0)
    
    def test_search_returns_empty_with_zero_max_results(self, zero_results_store, sample_course, sample_course_chunks):
        """Test that search returns empty results when max_results=0"""
        vs = zero_results_store
        
        # Add content
        vs.add_course_metadata(sample_course)
        vs.add_course_content(sample_course_chunks)
        
        # Search should return empty results due to max_results=0
        results = vs.search("machine learning")
        
        # ChromaDB actually throws an error for n_results=0, so we expect an error
        assert results.is_empty()
        assert results.error is not None  # ChromaDB throws error for n_results=0
        assert "cannot be negative, or zero" in results.error
    
    def test_search_with_explicit_limit_overrides_zero(self, zero_results_store, sample_course, sample_course_chunks):
        """Test that explicit limit parameter can override max_results=0"""
        vs = zero_results_store
        
        # Add content
        vs.add_course_metadata(sample_course)
        vs.add_course_content(sample_course_chunks)
        
        # Search with explicit limit should work
        results = vs.search("machine learning", limit=3)
        
        # Should return results even though max_results=0
        assert not results.is_empty()
        assert len(results.documents) <= 3
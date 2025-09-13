# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh && ./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Development Setup
```bash
# Install dependencies (runtime + development)
uv sync --group dev

# Set environment variables
# Create .env file with: ANTHROPIC_API_KEY=your_key_here
```

### Code Quality Workflow
```bash
# Format code with Black and isort
./scripts/format.sh

# Run all quality checks (format, lint, type check, tests)
./scripts/quality.sh

# Run specific checks
./scripts/lint.sh    # Flake8 and mypy only
./scripts/test.sh    # Tests only

# Set up pre-commit hooks (optional)
uv run pre-commit install
```

### Application Access
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for course materials built with FastAPI, ChromaDB, and Anthropic's Claude API.

### Core Architecture Pattern
The system uses **tool-based RAG** where Claude intelligently decides when to search course materials:

1. **Frontend** (`frontend/`) - Vanilla HTML/CSS/JS that communicates via REST API
2. **FastAPI Backend** (`backend/app.py`) - Main API server serving both static files and API endpoints
3. **RAG Orchestration** (`backend/rag_system.py`) - Core coordinator managing all components
4. **AI Generation** (`backend/ai_generator.py`) - Claude API integration with tool calling capability
5. **Search Tools** (`backend/search_tools.py`) - Pluggable tool system for semantic search
6. **Vector Storage** (`backend/vector_store.py`) - ChromaDB interface for similarity search
7. **Document Processing** (`backend/document_processor.py`) - Text chunking and course structure parsing

### Key Design Decisions

**Tool-Based Search**: Unlike traditional RAG that always retrieves, Claude decides whether to search based on query analysis. This prevents unnecessary retrievals for general questions.

**Dual Collections**: ChromaDB stores both course metadata (`course_catalog`) and content chunks (`course_content`) for flexible querying.

**Session Management**: Conversation context is maintained via `SessionManager` for multi-turn interactions.

**Course Structure Parsing**: Documents follow a specific format:
```
Course Title: [title]
Course Link: [url]  
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [url]
[lesson content...]
```

### Data Flow Architecture

1. **Query Processing**: User query → FastAPI → RAGSystem → AIGenerator
2. **Tool Decision**: Claude analyzes query and decides whether to use search_course_content tool
3. **Search Execution**: If needed: ToolManager → CourseSearchTool → VectorStore → ChromaDB
4. **Response Generation**: Tool results fed back to Claude for final synthesized answer
5. **Source Tracking**: Search sources tracked and returned to frontend for citation display

### Configuration System

All settings centralized in `backend/config.py`:
- `CHUNK_SIZE`: 800 characters (text chunk size for embeddings)
- `CHUNK_OVERLAP`: 100 characters (overlap between chunks)
- `MAX_RESULTS`: 5 (maximum search results)
- `MAX_HISTORY`: 2 (conversation turns to remember)
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2" (sentence transformer model)

### Document Processing Pipeline

Documents are processed through `DocumentProcessor`:
1. **Metadata Extraction**: Parse course title, link, and instructor from header
2. **Lesson Segmentation**: Split content by "Lesson N:" markers
3. **Text Chunking**: Sentence-aware chunking with configurable overlap
4. **Context Enhancement**: Add course/lesson context to each chunk for better retrieval

### Development Notes

- **Environment**: Requires `ANTHROPIC_API_KEY` in `.env` file
- **Dependencies**: Managed via `uv` (Python package manager)
- **Database**: ChromaDB persisted to `./chroma_db` directory
- **Document Loading**: Course files from `docs/` folder loaded on startup
- **Hot Reload**: FastAPI runs with `--reload` for development
- **Code Quality**: Use `./scripts/quality.sh` before committing changes

### Code Quality Standards
- **Formatting**: Black (88 char line length)
- **Import Sorting**: isort (Black-compatible profile)
- **Linting**: Flake8 (extends Black configuration)
- **Type Checking**: mypy (strict mode enabled)
- **Testing**: pytest (run via `./scripts/test.sh`)

When adding new features:
- Extend `Tool` interface in `search_tools.py` for new search capabilities
- Modify `DocumentProcessor` for different document formats
- Update `Config` class for new configuration options
- Add API endpoints in `app.py` following existing patterns
- **Always run `./scripts/quality.sh` before committing**
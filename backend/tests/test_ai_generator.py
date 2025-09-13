"""
Unit tests for AIGenerator class
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test cases for AIGenerator"""
    
    def test_init(self):
        """Test AIGenerator initialization"""
        generator = AIGenerator("test-api-key", "claude-3-sonnet")
        
        assert generator.model == "claude-3-sonnet"
        assert generator.base_params["model"] == "claude-3-sonnet"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
    
    @patch('ai_generator.anthropic')
    def test_generate_response_without_tools(self, mock_anthropic):
        """Test basic response generation without tools"""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response without tools"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-3-sonnet")
        result = generator.generate_response("What is machine learning?")
        
        assert result == "Test response without tools"
        mock_client.messages.create.assert_called_once()
        
        # Check API call parameters
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["model"] == "claude-3-sonnet"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "What is machine learning?"
        assert "tools" not in call_args
    
    @patch('ai_generator.anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Test response generation with conversation history"""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response with history"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-3-sonnet")
        history = "Previous conversation context"
        result = generator.generate_response("Follow up question", conversation_history=history)
        
        assert result == "Response with history"
        
        # Check that history was included in system prompt
        call_args = mock_client.messages.create.call_args[1]
        system_content = call_args["system"]
        assert "Previous conversation context" in system_content
    
    @patch('ai_generator.anthropic')
    def test_generate_response_with_tools(self, mock_anthropic):
        """Test response generation with tools but no tool use"""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response with tools available"
        mock_response.stop_reason = "end_turn"  # No tool use
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-3-sonnet")
        tools = [{"name": "search_tool", "description": "Search for content"}]
        mock_tool_manager = Mock()
        
        result = generator.generate_response(
            "General question", 
            tools=tools, 
            tool_manager=mock_tool_manager
        )
        
        assert result == "Response with tools available"
        
        # Check API call included tools
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}
        
        # No tool execution should have occurred
        mock_tool_manager.execute_tool.assert_not_called()
    
    @patch('ai_generator.anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic):
        """Test response generation when Claude uses tools"""
        # Setup mock for initial response with tool use
        mock_client = Mock()
        
        # First response: tool use
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_tool"
        mock_tool_content.input = {"query": "machine learning"}
        mock_tool_content.id = "tool_123"
        
        initial_response = Mock()
        initial_response.content = [mock_tool_content]
        initial_response.stop_reason = "tool_use"
        
        # Second response: final answer
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Final response after tool use"
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        mock_anthropic.Anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-3-sonnet")
        tools = [{"name": "search_tool", "description": "Search for content"}]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results here"
        
        result = generator.generate_response(
            "Search for machine learning", 
            tools=tools, 
            tool_manager=mock_tool_manager
        )
        
        assert result == "Final response after tool use"
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with("search_tool", query="machine learning")
        
        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2
    
    @patch('ai_generator.anthropic')
    def test_api_call_exception_handling(self, mock_anthropic):
        """Test exception handling in API calls"""
        # Setup mock to raise exception
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.Anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-3-sonnet")
        
        with pytest.raises(Exception, match="API Error"):
            generator.generate_response("Test query")
    
    @patch('ai_generator.anthropic')
    def test_handle_tool_execution(self, mock_anthropic):
        """Test the _handle_tool_execution method directly"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-3-sonnet")
        
        # Create mock initial response with tool use
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use" 
        mock_tool_content.name = "test_tool"
        mock_tool_content.input = {"param": "value"}
        mock_tool_content.id = "tool_456"
        
        initial_response = Mock()
        initial_response.content = [mock_tool_content]
        
        # Create mock final response
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Tool execution complete"
        mock_client.messages.create.return_value = final_response
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        # Base API parameters
        base_params = {
            "model": "claude-3-sonnet",
            "messages": [{"role": "user", "content": "Original query"}],
            "system": "System prompt"
        }
        
        result = generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)
        
        assert result == "Tool execution complete"
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with("test_tool", param="value")
        
        # Verify final API call structure
        call_args = mock_client.messages.create.call_args[1]
        messages = call_args["messages"]
        
        # Should have: original user message, assistant tool use, user tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        
        # Check tool result format
        tool_result = messages[2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "tool_456"
        assert tool_result["content"] == "Tool result"
    
    @patch('ai_generator.anthropic')
    def test_handle_multiple_tool_calls(self, mock_anthropic):
        """Test handling multiple tool calls in one response"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-3-sonnet")
        
        # Create mock initial response with multiple tools
        mock_tool1 = Mock()
        mock_tool1.type = "tool_use"
        mock_tool1.name = "search_tool"
        mock_tool1.input = {"query": "query1"}
        mock_tool1.id = "tool_1"
        
        mock_tool2 = Mock()
        mock_tool2.type = "tool_use"
        mock_tool2.name = "outline_tool"
        mock_tool2.input = {"course": "course1"}
        mock_tool2.id = "tool_2"
        
        initial_response = Mock()
        initial_response.content = [mock_tool1, mock_tool2]
        
        # Mock final response
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Multiple tools executed"
        mock_client.messages.create.return_value = final_response
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        base_params = {
            "model": "claude-3-sonnet",
            "messages": [{"role": "user", "content": "Multi-tool query"}],
            "system": "System prompt"
        }
        
        result = generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)
        
        assert result == "Multiple tools executed"
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_tool", query="query1")
        mock_tool_manager.execute_tool.assert_any_call("outline_tool", course="course1")
        
        # Check tool results in final API call
        call_args = mock_client.messages.create.call_args[1]
        tool_results = call_args["messages"][2]["content"]
        
        assert len(tool_results) == 2
        assert tool_results[0]["tool_use_id"] == "tool_1"
        assert tool_results[0]["content"] == "Result 1"
        assert tool_results[1]["tool_use_id"] == "tool_2"  
        assert tool_results[1]["content"] == "Result 2"
    
    def test_system_prompt_content(self):
        """Test that system prompt contains expected instructions"""
        generator = AIGenerator("test-key", "claude-3-sonnet")
        
        assert "search_course_content" in generator.SYSTEM_PROMPT
        assert "get_course_outline" in generator.SYSTEM_PROMPT
        assert "Sequential Tool Usage Guidelines" in generator.SYSTEM_PROMPT
        assert "UP TO 2 tool calls" in generator.SYSTEM_PROMPT
        assert "Round 1" in generator.SYSTEM_PROMPT
        assert "Round 2" in generator.SYSTEM_PROMPT
    
    @patch('ai_generator.anthropic')
    def test_generate_response_error_logging(self, mock_anthropic, capsys):
        """Test that API errors are logged properly"""
        # Setup mock to raise exception
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("Test API Error")
        mock_anthropic.Anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-3-sonnet")
        
        with pytest.raises(Exception):
            generator.generate_response("Test query")
        
        # Check that error was printed (logged)
        captured = capsys.readouterr()
        assert "API call failed with exception:" in captured.out
        assert "Test API Error" in captured.out


class TestSequentialToolCalling:
    """Test cases for sequential tool calling functionality"""
    
    @patch('ai_generator.anthropic')
    def test_sequential_tool_calling_two_rounds(self, mock_anthropic):
        """Test successful sequential tool calling with 2 rounds"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Round 1: Tool use response
        mock_tool_content_1 = Mock()
        mock_tool_content_1.type = "tool_use"
        mock_tool_content_1.name = "get_course_outline"
        mock_tool_content_1.input = {"course_name": "MCP Course"}
        mock_tool_content_1.id = "tool_1"
        
        round_1_response = Mock()
        round_1_response.content = [mock_tool_content_1]
        round_1_response.stop_reason = "tool_use"
        
        # Round 2: Another tool use response
        mock_tool_content_2 = Mock()
        mock_tool_content_2.type = "tool_use"
        mock_tool_content_2.name = "search_course_content"
        mock_tool_content_2.input = {"query": "authentication", "course_name": "FastAPI Course"}
        mock_tool_content_2.id = "tool_2"
        
        round_2_response = Mock()
        round_2_response.content = [mock_tool_content_2]
        round_2_response.stop_reason = "tool_use"
        
        # Final response: No tool use
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Based on both searches, here's the comparison..."
        final_response.stop_reason = "end_turn"
        
        # Configure mock to return responses in sequence
        mock_client.messages.create.side_effect = [round_1_response, round_2_response, final_response]
        
        # Setup tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Course outline result", "Search result"]
        mock_tool_manager.get_last_sources.side_effect = [
            [{"text": "MCP Course - Course Outline", "link": None}],
            [{"text": "FastAPI Course - Lesson 1", "link": "https://example.com"}]
        ]
        
        generator = AIGenerator("test-key", "claude-3-sonnet")
        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]
        
        result = generator.generate_response_with_sequential_tools(
            "Compare MCP course structure with FastAPI authentication content",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        assert result == "Based on both searches, here's the comparison..."
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="MCP Course")
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="authentication", course_name="FastAPI Course")
        
        # Verify 3 API calls were made (2 rounds + final synthesis)
        assert mock_client.messages.create.call_count == 3
    
    @patch('ai_generator.anthropic')
    def test_sequential_tool_calling_early_termination(self, mock_anthropic):
        """Test early termination when Claude provides final answer after 1 tool call"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Round 1: Tool use response
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "machine learning basics"}
        mock_tool_content.id = "tool_1"
        
        round_1_response = Mock()
        round_1_response.content = [mock_tool_content]
        round_1_response.stop_reason = "tool_use"
        
        # Round 2: Final response without tool use
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Here's what I found about machine learning basics..."
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [round_1_response, final_response]
        
        # Setup tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Machine learning is..."
        mock_tool_manager.get_last_sources.return_value = [{"text": "ML Course - Lesson 1", "link": None}]
        
        generator = AIGenerator("test-key", "claude-3-sonnet")
        tools = [{"name": "search_course_content"}]
        
        result = generator.generate_response_with_sequential_tools(
            "What are machine learning basics?",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        assert result == "Here's what I found about machine learning basics..."
        
        # Verify only 1 tool execution
        assert mock_tool_manager.execute_tool.call_count == 1
        
        # Verify 2 API calls (1 tool use + 1 final response)
        assert mock_client.messages.create.call_count == 2
    
    @patch('ai_generator.anthropic')
    def test_sequential_tool_calling_tool_error_handling(self, mock_anthropic):
        """Test error handling when tool execution fails"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Round 1: Tool use response
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test query"}
        mock_tool_content.id = "tool_1"
        
        round_1_response = Mock()
        round_1_response.content = [mock_tool_content]
        round_1_response.stop_reason = "tool_use"
        
        mock_client.messages.create.return_value = round_1_response
        
        # Setup tool manager to fail
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        generator = AIGenerator("test-key", "claude-3-sonnet")
        tools = [{"name": "search_course_content"}]
        
        result = generator.generate_response_with_sequential_tools(
            "Test query",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        assert "Tool execution failed in round 1" in result
        assert mock_tool_manager.execute_tool.call_count == 1
    
    @patch('ai_generator.anthropic')
    def test_sequential_tool_calling_max_rounds_limit(self, mock_anthropic):
        """Test that sequential calling stops after max rounds"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Both rounds return tool use (Claude wants to continue)
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_content.id = "tool_1"
        
        tool_use_response = Mock()
        tool_use_response.content = [mock_tool_content]
        tool_use_response.stop_reason = "tool_use"
        
        # Final synthesis response
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Final synthesis after 2 rounds"
        
        # Return tool use for rounds 1&2, then final synthesis
        mock_client.messages.create.side_effect = [tool_use_response, tool_use_response, final_response]
        
        # Setup tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        mock_tool_manager.get_last_sources.return_value = []
        
        generator = AIGenerator("test-key", "claude-3-sonnet")
        tools = [{"name": "search_course_content"}]
        
        result = generator.generate_response_with_sequential_tools(
            "Complex query requiring multiple searches",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        assert result == "Final synthesis after 2 rounds"
        
        # Should execute exactly 2 tools (max rounds limit)
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Should make 3 API calls (2 tool rounds + final synthesis)
        assert mock_client.messages.create.call_count == 3
    
    @patch('ai_generator.anthropic')
    def test_sequential_tool_calling_no_tools_provided(self, mock_anthropic):
        """Test fallback to regular generation when no tools provided"""
        # Setup mock client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response without tools"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-3-sonnet")
        
        result = generator.generate_response_with_sequential_tools(
            "Simple query",
            tools=None,
            tool_manager=None,
            max_rounds=2
        )
        
        assert result == "Response without tools"
    
    @patch('ai_generator.anthropic')
    def test_sequential_tool_calling_api_error(self, mock_anthropic):
        """Test error handling when API call fails during sequential execution"""
        # Setup mock client to fail
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API connection failed")
        mock_anthropic.Anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-3-sonnet")
        tools = [{"name": "search_course_content"}]
        mock_tool_manager = Mock()
        
        result = generator.generate_response_with_sequential_tools(
            "Test query",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        assert "An error occurred during tool execution: API connection failed" in result
import anthropic
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ToolRoundState:
    """State management for sequential tool calling rounds"""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    round_count: int = 0
    max_rounds: int = 2
    all_sources: List = field(default_factory=list)
    completed_tool_calls: List[Dict[str, Any]] = field(default_factory=list)

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **search_course_content**: For questions about specific course content or detailed educational materials
2. **get_course_outline**: For questions about course structure, outlines, lesson lists, or course overview

Sequential Tool Usage Guidelines:
- You can make UP TO 2 tool calls across separate rounds to answer complex queries
- **Round 1**: Use for initial information gathering or primary search
- **Round 2** (if needed): Use for follow-up searches, comparisons, or additional context

Tool Strategy by Query Type:
- **Simple queries**: Use 1 tool call for direct answers
- **Comparison queries**: Round 1 for first item, Round 2 for second item  
- **Multi-part questions**: Round 1 for primary info, Round 2 for additional details
- **Cross-course analysis**: Round 1 for one course, Round 2 for another

Specific Tool Usage:
- Use **get_course_outline** for:
  - Course structure requests ("What's in the course?", "Course outline", "List of lessons")
  - Course overview questions
  - When user asks about course organization or what lessons are included
- Use **search_course_content** for:
  - Specific content questions about lessons or topics
  - Detailed educational material searches

Complex Query Examples:
- "Compare lesson 1 of Course A with lesson 1 of Course B" → search each course separately
- "What courses cover authentication?" → search broadly, then search specific courses if needed
- "Find a course similar to lesson 3 of MCP course" → get outline first, then search for similar topics

Response Requirements:
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Course Outline Responses:
- When using get_course_outline, provide the complete course information:
  - Course title and instructor
  - Course link (if available)
  - Complete lesson list with numbers and titles
  - Lesson links (if available)

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the tool results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        
        try:
            response = self.client.messages.create(**api_params)
        except Exception as e:
            print("API call failed with exception:", e)
            print("API parameters:", api_params)
            raise

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def generate_response_with_sequential_tools(self, query: str,
                                              conversation_history: Optional[str] = None,
                                              tools: Optional[List] = None,
                                              tool_manager=None,
                                              max_rounds: int = 2) -> str:
        """
        Generate AI response with support for up to max_rounds sequential tool calls.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default 2)
            
        Returns:
            Generated response as string
        """
        if not tools or not tool_manager:
            # Fall back to basic response generation without tools
            return self.generate_response(query, conversation_history)
        
        # Initialize state for sequential tool calling
        round_state = ToolRoundState(max_rounds=max_rounds)
        round_state.messages.append({"role": "user", "content": query})
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Execute sequential tool calling rounds
        for round_num in range(1, max_rounds + 1):
            try:
                response = self._execute_round(round_state, tools, system_content, round_num)
                
                if response.stop_reason != "tool_use":
                    # No more tools needed - return final answer
                    return self._extract_final_response(response, round_state)
                
                # Execute tools for this round
                tool_success = self._process_tool_execution_for_round(response, round_state, tool_manager, round_num)
                
                if not tool_success:
                    # Tool execution failed - return error response
                    return f"Tool execution failed in round {round_num}. Unable to complete the request."
                
                # Continue to next round if we haven't reached max rounds
                
            except Exception as e:
                print(f"Error in tool calling round {round_num}:", e)
                return f"An error occurred during tool execution: {str(e)}"
        
        # Max rounds reached - generate final response
        return self._generate_final_response_after_tools(round_state, system_content)
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
    
    def _execute_round(self, round_state: ToolRoundState, tools: List, system_content: str, round_num: int):
        """Execute a single round of tool calling"""
        # Prepare API call parameters with tools available
        api_params = {
            **self.base_params,
            "messages": round_state.messages.copy(),
            "system": system_content,
            "tools": tools,
            "tool_choice": {"type": "auto"}
        }
        
        # Get response from Claude
        try:
            response = self.client.messages.create(**api_params)
            return response
        except Exception as e:
            print(f"API call failed in round {round_num} with exception:", e)
            raise
    
    def _process_tool_execution_for_round(self, response, round_state: ToolRoundState, tool_manager, round_num: int) -> bool:
        """Execute tools for current round and update state"""
        # Add assistant's tool use response to message history
        round_state.messages.append({"role": "assistant", "content": response.content})
        
        # Execute all tool calls in this response
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                    
                    # Track completed tool call for debugging/analytics
                    round_state.completed_tool_calls.append({
                        "round": round_num,
                        "tool": content_block.name,
                        "params": content_block.input,
                        "result": tool_result
                    })
                    
                except Exception as e:
                    print(f"Tool execution failed in round {round_num}:", e)
                    return False
        
        # Add tool results to message history for next round
        if tool_results:
            round_state.messages.append({"role": "user", "content": tool_results})
            
            # Collect sources from this round but don't reset yet
            # Sources will be accumulated across rounds and retrieved by RAG system
            round_sources = tool_manager.get_last_sources()
            round_state.all_sources.extend(round_sources)
            # Don't reset sources here - let RAG system handle it after all rounds
        
        round_state.round_count += 1
        return True
    
    def _extract_final_response(self, response, round_state: ToolRoundState) -> str:
        """Extract final response when no more tools are needed"""
        return response.content[0].text
    
    def _generate_final_response_after_tools(self, round_state: ToolRoundState, system_content: str) -> str:
        """Generate final response after max rounds reached"""
        # Prepare final API call without tools to get synthesis
        final_params = {
            **self.base_params,
            "messages": round_state.messages,
            "system": system_content
        }
        
        try:
            final_response = self.client.messages.create(**final_params)
            return final_response.content[0].text
        except Exception as e:
            print("Final response generation failed:", e)
            return "Unable to generate final response after tool execution."
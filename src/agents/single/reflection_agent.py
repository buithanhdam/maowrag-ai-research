import asyncio
from typing import List, Any, Generator, Optional, AsyncGenerator, Dict, Tuple
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool

from src.llm import BaseLLM
from ..base import BaseAgent
from ..design import (
    AgentOptions,
    retry_on_error,
)


class ReflectionAgent(BaseAgent):
    """
    Agent that uses a reflection process to iteratively improve content generation.
    This implementation aligns better with PlanningAgent's structure while maintaining
    the reflection-based approach.

    """

    def __init__(
        self,
        llm: BaseLLM,
        options: AgentOptions,
        system_prompt: str = "",
        tools: List[FunctionTool] = [],
    ):
        super().__init__(llm, options, system_prompt, tools)
        self._log_debug(
            f"Agent {self.name}-[Reflection] initialized successfully. With {len(tools)} tools."
        )

    def _get_default_generation_prompt(self) -> str:
        """Default prompt for the generation phase"""
        return """
        Your task is to generate the best content possible for the user's request.
        If the user provides critique, respond with a revised version of your previous attempt.
        You must always output ONLY the revised content based on the feedback provided. Do not include any introductory.
        Focus on making each iteration better than the last by addressing all feedback points.
        """

    def _get_default_reflection_prompt(self) -> str:
        """Default prompt for the reflection phase"""
        return """
        You are tasked with generating critique and recommendations to improve the content.
        Analyze the content carefully and provide specific, actionable feedback.
        If the content has issues or can be improved, output a numbered list of recommendations.
        If the content is satisfactory and there's nothing to change, output this: <OK>
        Be critical but constructive - focus on how to make the content better.
        Utilize available tools if necessary to improve or validate the content.
        """

    def _extract_tool_recommendations(
        self, critique: str, verbose: bool = False
    ) -> List[Tuple[str, str]]:
        """
        Extract tool recommendations from critique with improved pattern matching.
        Looks for patterns indicating tool usage recommendations.

        Args:
            critique: The critique text to analyze
            verbose: Whether to log detailed information

        Returns:
            List of tuples containing (tool_name, description)
        """
        tool_recommendations = []

        if verbose:
            self._log_debug(f"Extracting tool recommendations from critique")

        # Check each tool name in the critique
        for tool_name in self.tools_dict.keys():
            if tool_name.lower() in critique.lower():
                import re

                # Try different patterns for tool recommendation extraction
                patterns = [
                    # "Use [tool_name] to [description]"
                    rf"(?:use|utilize|apply|try)\s+{re.escape(tool_name)}\s+(?:to|for)\s+([^\.]+)",
                    # "[tool_name] could [description]"
                    rf"{re.escape(tool_name)}\s+(?:could|should|would|can|might)\s+([^\.]+)",
                    # "Try [tool_name] [description]"
                    rf"(?:try|consider|recommend)\s+{re.escape(tool_name)}\s+([^\.]+)",
                ]

                for pattern in patterns:
                    match = re.search(pattern, critique, re.IGNORECASE)
                    if match:
                        description = match.group(1).strip()
                        tool_recommendations.append((tool_name, description))
                        break
                else:
                    # Fallback if no pattern matched but tool name was mentioned
                    tool_recommendations.append(
                        (tool_name, f"Improve the content using {tool_name}")
                    )

        if verbose:
            self._log_info(
                f"Extracted {len(tool_recommendations)} tool recommendations"
            )
            for tool, desc in tool_recommendations:
                self._log_debug(f"Tool: {tool}, Description: {desc}")

        return tool_recommendations

    async def _generate_content(self, query: str, verbose: bool = False) -> str:
        """Generate content based on the current generation history"""
        if verbose:
            self._log_debug("Generating content...")

        try:
            if not query:
                self._log_warning("No user message found")
                return None

            query_gen = f"""
            Your task: {self._get_default_generation_prompt()}
            {f"Short memory: {self.chat_memory.get_short_memories()}" if self.chat_memory.get_short_memories() else ""}
            User query: {query}
            """

            # Use long_memories for context and the query_gen as prompt
            response = await self.llm.achat(
                query=query_gen, chat_history=self.chat_memory.get_long_memories()
            )

            if verbose:
                response_preview = (
                    f"{response[:100]}..." if len(response) > 100 else response
                )
                self._log_info(f"Generated content: {response_preview}")
            self.chat_memory.add_short_memory("user", query_gen)
            self.chat_memory.add_short_memory("assistant", response)
            return response

        except Exception as e:
            self._log_error(f"Error in content generation: {str(e)}")
            raise

    async def _reflect_on_content(
        self, available_tools: List[str] = None, verbose: bool = False
    ) -> str:
        """Generate critique and feedback on the provided content"""
        if verbose:
            self._log_debug("Reflecting on content")

        try:
            # Add the content to be reflected upon
            reflection_msg = self._get_default_reflection_prompt()
            # Add tools information if available
            if available_tools:
                tools_info = "\n\nAvailable tools for verification or improvement:\n"
                tools_info += self._format_tool_signatures()
                reflection_msg += tools_info

            critique = await self.llm.achat(
                query=reflection_msg, chat_history=self.chat_memory.get_short_memories()
            )

            if verbose:
                critique_preview = (
                    f"{critique[:100]}..." if len(critique) > 100 else critique
                )
                self._log_info(f"Generated critique: {critique_preview}")

            self.chat_memory.add_short_memory("user", reflection_msg)
            return critique

        except Exception as e:
            self._log_error(f"Error in reflection: {str(e)}")
            raise

    async def _apply_tool_improvements(
        self, query: str, critique: str, max_tool_steps: int = 2, verbose: bool = False
    ) -> Dict[str, Any]:
        """Apply tool-based improvements based on the critique"""
        tool_results = {}
        tool_recommendations = self._extract_tool_recommendations(critique, verbose)

        if not tool_recommendations:
            if verbose:
                self._log_warning("No tool recommendations identified in critique")
            return tool_results

        tool_count = 0
        for tool_name, description in tool_recommendations:
            if tool_count >= max_tool_steps:
                if verbose:
                    self._log_warning(f"Reached maximum tool steps ({max_tool_steps})")
                break

            if tool_name not in self.tools_dict:
                self._log_warning(
                    f"Recommended tool '{tool_name}' not found in available tools"
                )
                continue

            if verbose:
                self._log_debug(f"Applying tool: {tool_name} for: {description}")
            task_todo = f"""
            User query: {query}
            Step description: {description}
            """
            result = await self._execute_tool(
                task=task_todo,
                tool_name=tool_name,
                requires_tool=True,
                verbose=verbose,
            )

            tool_results[tool_name] = result
            tool_count += 1

        return tool_results

    def _format_critique_with_tool_results(
        self, critique: str, tool_results: Dict[str, Any]
    ) -> str:
        """Format the critique with tool results for better context in next generation"""
        if not tool_results:
            return critique

        formatted_critique = critique + "\n\nTool Results:\n"
        for tool_name, result in tool_results.items():
            result_str = str(result)
            # Truncate very long results for readability
            if len(result_str) > 500:
                result_str = result_str[:500] + "... [truncated]"

            formatted_critique += f"\n- {tool_name}: {result_str}"

        return formatted_critique

    @retry_on_error()
    async def run(
        self,
        query: str,
        n_iterations: int = 2,
        max_tool_steps: int = 2,
        verbose: bool = False,
        chat_history: List[ChatMessage] = [],
    ) -> str:
        """
        Run the reflection-based content generation process

        Args:
            query: The user's query/request
            n_iterations: Maximum number of reflection-generation cycles
            max_tool_steps: Maximum number of tool executions per iteration
            verbose: Whether to log detailed information
            chat_history: Previous conversation history

        Returns:
            The final generated content after reflection iterations
        """
        if verbose:
            self._log_debug(f"Starting reflection process for query: {query}")

        self.chat_memory.set_initial_long_memories(chat_history)
        self.chat_memory.reset_short_memories()

        # Initialize tracking variables
        current_iteration = 0
        gen_content = None

        # Main reflection loop
        while current_iteration < n_iterations:
            current_iteration += 1
            if verbose:
                self._log_debug(
                    f"Starting iteration {current_iteration}/{n_iterations}"
                )

            # Generate content
            gen_content = await self._generate_content(query, verbose)
            if not gen_content:
                self._log_error(
                    "Failed to generate content now, ending reflection process"
                )
                return None

            # Generate critique/reflection
            critique = await self._reflect_on_content(
                [tool.metadata.name for tool in self.tools], verbose
            )

            # Check if content is satisfactory
            if "<OK>" in critique:
                self.chat_memory.add_short_memory("assistant", critique)
                if verbose:
                    self._log_info(
                        "Content deemed satisfactory, ending reflection process"
                    )
                return gen_content

            # Apply tool improvements
            tool_results = await self._apply_tool_improvements(
                query, critique, max_tool_steps, verbose
            )

            # Format critique with tool results
            enhanced_critique = self._format_critique_with_tool_results(
                critique, tool_results
            )

            self.chat_memory.add_short_memory("assistant", enhanced_critique)

            if verbose:
                self._log_info(f"Completed iteration {current_iteration}")

        if verbose:
            self._log_warning(
                f"Reflection process complete with {n_iterations} iterations but no satisfactory content found"
            )
            self._log_info(f"Failing back to summary generation")

        return await self._generate_final_response(query, verbose)

    async def achat(
        self,
        query: str,
        verbose: bool = False,
        chat_history: List[ChatMessage] = [],
        n_iterations: int = 5,
        max_tool_steps: int = 5,
        *args,
        **kwargs,
    ) -> str:
        """Async chat interface for the reflection agent"""
        # Get additional parameters or use defaults

        if self.callbacks:
            self.callbacks.on_agent_start(self.name)

        try:
            # Run the reflection process
            result = await self.run(
                query=query,
                n_iterations=n_iterations,
                max_tool_steps=max_tool_steps,
                verbose=verbose,
                chat_history=chat_history,
            )

            return result

        finally:
            if self.callbacks:
                self.callbacks.on_agent_end(self.name)

    def chat(
        self,
        query: str,
        verbose: bool = False,
        chat_history: List[ChatMessage] = [],
        n_iterations: int = 5,
        max_tool_steps: int = 5,
        *args,
        **kwargs,
    ) -> str:
        """Synchronous chat interface for the reflection agent"""
        # Create an event loop if one doesn't exist
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the async chat method in the event loop
        return loop.run_until_complete(
            self.achat(
                query=query,
                verbose=verbose,
                chat_history=chat_history,
                n_iterations=n_iterations,
                max_tool_steps=max_tool_steps,
                *args,
                **kwargs,
            )
        )

    async def astream_chat(
        self,
        query: str,
        verbose: bool = False,
        chat_history: Optional[List[ChatMessage]] = None,
        n_iterations: int = 5,
        max_tool_steps: int = 5,
        *args,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Async streaming chat interface for the reflection agent"""
        # Get parameters
        chat_history = chat_history or []

        if self.callbacks:
            self.callbacks.on_agent_start(self.name)

        try:
            # First perform the reflection process to get the final content
            final_content = await self.run(
                query=query,
                n_iterations=n_iterations,
                max_tool_steps=max_tool_steps,
                verbose=verbose,
                chat_history=chat_history,
            )

            # Simulate streaming for better UX
            chunk_size = 5
            for i in range(0, len(final_content), chunk_size):
                chunk = final_content[i : i + chunk_size]
                yield chunk
                await asyncio.sleep(0.01)  # Small delay to simulate streaming

                # If there's a token callback, use it
                if self.callbacks and hasattr(self.callbacks, "on_llm_new_token"):
                    self.callbacks.on_llm_new_token(chunk)

        except Exception as e:
            self._log_error(f"Error in astream_chat: {str(e)}")
            yield f"Error: {str(e)}"
        finally:
            if self.callbacks:
                self.callbacks.on_agent_end(self.name)

    def stream_chat(
        self,
        query: str,
        verbose: bool = False,
        chat_history: Optional[List[ChatMessage]] = None,
        n_iterations: int = 5,
        max_tool_steps: int = 5,
        *args,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Synchronous streaming chat interface for the reflection agent"""
        # Create an event loop if one doesn't exist
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Get the async generator
        async_gen = self.astream_chat(
            query=query,
            verbose=verbose,
            chat_history=chat_history,
            n_iterations=n_iterations,
            max_tool_steps=n_iterations,
            *args,
            **kwargs,
        )

        # Helper function to convert async generator to sync generator
        def sync_generator():
            agen = async_gen.__aiter__()
            while True:
                try:
                    yield loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    break

        return sync_generator()

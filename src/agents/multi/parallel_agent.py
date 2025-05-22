from typing import Any, AsyncGenerator, Dict, Generator, List, Optional
import asyncio
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool

from src.llm import BaseLLM
from .base import BaseMultiAgent
from ..design import (
    AgentOptions,
    retry_on_error,
)


class ParallelAgent(BaseMultiAgent):
    """ParallelAgent that executes multiple agents in parallel and combines their results"""

    def __init__(
        self,
        llm: BaseLLM,
        options: AgentOptions,
        system_prompt: str = "",
        tools: List[FunctionTool] = [],
        validation_threshold: float = 0.7,
    ):
        super().__init__(llm, options, system_prompt, tools, validation_threshold)

    async def _execute_parallel(
        self, query: str, verbose: bool = False, *args, **kwargs
    ) -> bool:
        """Execute multiple agents in parallel and collect their results"""

        agents_to_execute = list(self.agent_registry.values())
        if verbose:
            self._log_debug(f"Executing {len(agents_to_execute)} agents in parallel")

        # Create tasks for each agent
        tasks = []
        for agent in agents_to_execute:
            if verbose:
                self._log_debug(f"Creating task for agent: {agent.name}")
            task = asyncio.create_task(
                agent.achat(
                    query=query,
                    verbose=verbose,
                    chat_history=self.chat_memory.get_long_memories(),
                    *args,
                    **kwargs,
                )
            )
            tasks.append((agent.name, task))

        for agent_name, task in tasks:
            try:
                result = await task
                self.chat_memory.add_short_memory(
                    "assistant",
                    f"Agent: {agent_name} process successfully with result: {result}",
                )
                if verbose:
                    self._log_info(f"Agent {agent_name} completed successfully")
            except Exception as e:
                self._log_error(f"Error executing agent {agent_name}: {str(e)}")
                self.chat_memory.add_short_memory(
                    "assistant",
                    f"Agent: {agent_name} process failed with error: {str(e)}",
                )
        return True

    @retry_on_error()
    async def run(
        self,
        query: str,
        chat_history: List[ChatMessage] = [],
        verbose: bool = False,
        *args,
        **kwargs,
    ) -> str:
        """Process user request by executing multiple agents in parallel"""
        self.chat_memory.set_initial_long_memories(chat_history)
        self.chat_memory.reset_short_memories()

        if verbose:
            self._log_debug(f"🔍 Starting Parallel agent for query: {query}")
        try:

            if not list(self.agent_registry.values()):
                if verbose:
                    self._log_warning(
                        "No agents available for execution, Falling back to default agent response"
                    )
                response = await self.llm.achat(
                    query, chat_history=self.chat_memory.get_long_memories()
                )
                await asyncio.sleep(0.1)
                return response

            # Execute agents in parallel
            await self._execute_parallel(query=query, verbose=verbose, *args, **kwargs)
            # Integrate results
            final_response = await self._generate_final_response(
                query=query, verbose=verbose
            )

            return final_response

        except Exception as e:
            self._log_error(f"Error in parallel execution: {str(e)}")
            return (
                "I encountered an error while processing your request in parallel mode. "
                f"Error: {str(e)}"
            )

    # Override the achat method to support parallel execution
    async def achat(
        self,
        query: str,
        verbose: bool = False,
        chat_history: List[ChatMessage] = [],
        *args,
        **kwargs,
    ) -> str:
        """Async chat implementation that supports both parallel and sequential execution"""
        if self.callbacks:
            self.callbacks.on_agent_start(self.name)

        try:
            result = await self.run(
                query=query, chat_history=chat_history, verbose=verbose, *args, **kwargs
            )

            return result

        finally:
            if self.callbacks:
                self.callbacks.on_agent_end(self.name)

    # Override the chat method to support parallel execution
    def chat(
        self,
        query: str,
        verbose: bool = False,
        chat_history: List[ChatMessage] = [],
        *args,
        **kwargs,
    ) -> str:
        """Sync chat implementation that supports both parallel and sequential execution"""
        # Create an event loop if one doesn't exist
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the async chat method in the event loop
        return loop.run_until_complete(
            self.achat(
                query=query, verbose=verbose, chat_history=chat_history, *args, **kwargs
            )
        )

    async def astream_chat(
        self,
        query: str,
        verbose: bool = False,
        chat_history: Optional[List[ChatMessage]] = None,
        *args,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Async streaming chat interface for the agent"""
        # Get additional parameters
        chat_history = chat_history or []
        if self.callbacks:
            self.callbacks.on_agent_start(self.name)

        try:
            result = await self.run(
                query=query,
                verbose=verbose,
                chat_history=chat_history,
                *args,
                **kwargs,
            )

            # Stream the final result in chunks to simulate streaming
            chunk_size = 5
            for i in range(0, len(result), chunk_size):
                yield result[i : i + chunk_size]
                await asyncio.sleep(0.01)

        finally:
            if self.callbacks:
                self.callbacks.on_agent_end(self.name)

    def stream_chat(
        self,
        query: str,
        verbose: bool = False,
        chat_history: Optional[List[ChatMessage]] = None,
        *args,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Synchronous streaming chat interface for the agent"""
        # Create an event loop if one doesn't exist
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Get the async generator
        async_gen = self.astream_chat(
            query=query, verbose=verbose, chat_history=chat_history, *args, **kwargs
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

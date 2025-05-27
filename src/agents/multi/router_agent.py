from typing import Any, Dict, List, Optional, Tuple, Generator, AsyncGenerator
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
import json
import asyncio

from ..utils import AgentOptions, clean_json_response, retry_on_error
from ..base import BaseAgent
from .base import BaseMultiAgent
from src.llm import BaseLLM


class RouterAgent(BaseMultiAgent):
    """RouterAgent that routes user requests to appropriate agents based on classification"""
    def __init__(
        self,
        llm: BaseLLM,
        options: AgentOptions,
        system_prompt: str = "",
        tools: List[FunctionTool] = [],
        validation_threshold=0.7,
    ):
        super().__init__(llm, options, system_prompt, tools, validation_threshold)

    async def _classify_request(
        self, query: str, verbose: bool = False
    ) -> Tuple[Optional[BaseAgent], float, str]:
        """Classify user request using LLM and return appropriate agent with confidence score and reasoning"""
        if verbose:
            self._log_debug("Starting classification process...")
        CLASSIFY_PROMPT = f"""\
        You are AgentMatcher, an intelligent assistant designed to analyze user queries and match them with 
        the most suitable agent or department. Your task is to understand the user request,
        identify key entities and intents, and determine which agent or department would be best equipped
        to handle the query.

        Important: The user input may be a follow-up response to a previous interaction.
        The conversation history, including the name of the previously selected agent, is provided.
        If the user's input appears to be a continuation of the previous conversation
        (e.g., 'yes', 'ok', 'I want to know more', '1'), select the same agent as before.

        Available agents and their capabilities: {self._get_agent_descriptions()}

        Based on the user input and chat history, determine the most appropriate agent and provide a confidence score (0-1).

        Respond in JSON format:
        {{
            "selected_agent": "agent_id",
            "confidence": 0.0,
            "reasoning": "brief explanation"
        }}
                
        User query: {query}
        \
        """
        try:

            if len(self.agent_registry) == 0:
                if verbose:
                    self._log_warning("No agents registered with manager")
                return None, 0.0, "No agents available"
            # Get classification from LLM
            response = await self.llm.achat(
                CLASSIFY_PROMPT, chat_history=self.chat_memory.get_long_memories()
            )
            response = clean_json_response(response)

            try:
                # Parse LLM response
                classification = json.loads(response)
                selected_agent_id = classification["selected_agent"]
                confidence = float(classification["confidence"])
                reasoning = classification["reasoning"]

                # Get selected agent
                selected_agent = self.agent_registry.get(selected_agent_id)

                if selected_agent:
                    if verbose:
                        self._log_info(
                            f"Request classified to {selected_agent.name} with (confidence: {confidence:.2f}). Reasoning: {reasoning}"
                        )
                    return selected_agent, confidence, reasoning
                else:
                    if verbose:
                        self._log_warning(
                            f"Selected agent {selected_agent_id} not found in registry"
                        )
                    default_agent = (
                        next(iter(self.agent_registry.values()))
                        if self.agent_registry
                        else None
                    )
                    return (
                        default_agent,
                        0.5,
                        f"Selected agent {selected_agent_id} not found, using default",
                    )

            except (json.JSONDecodeError, KeyError) as e:
                self._log_error(f"Error parsing LLM classification response: {str(e)}")
                default_agent = (
                    next(iter(self.agent_registry.values()))
                    if self.agent_registry
                    else None
                )
                return default_agent, 0.5, f"Error in classification: {str(e)}"

        except Exception as e:
            self._log_error(f"Error during request classification: {str(e)}")
            default_agent = (
                next(iter(self.agent_registry.values()))
                if self.agent_registry
                else None
            )
            return default_agent, 0.5, f"Exception in classification: {str(e)}"

    async def _validate_response(
        self, query: str, verbose: bool = False
    ) -> Dict[str, Any]:
        """Validate an agent's response to ensure it properly addresses the user query"""
        if verbose:
            self._log_debug("Validating agent response...")
        VALIDATION_PROMPT = f"""\
        You are a ValidatorAgent, responsible for evaluating the quality and relevance of agent responses to user queries.

        Your task is to assess whether the agent's response appropriately addresses the user's query, both in terms of content and context.

        User Query: {query}
        Selected Agent Response: {[f"{memory.content}" for memory in self.chat_memory.get_short_memories()]}

        Please evaluate and respond in JSON format:
        {{
            "is_valid": true/false,
            "score": 0.0,  // Score between 0-1, where 1 is perfect
            "reasoning": "your reasoning here",
            "needs_refinement": true/false,
            "refinement_suggestions": "specific suggestions if needed"
        }}
        """
        try:

            validation_response = await self.llm.achat(
                VALIDATION_PROMPT, chat_history=self.chat_memory.get_long_memories()
            )
            validation_response = clean_json_response(validation_response)

            try:
                validation_result = json.loads(validation_response)
                if verbose:
                    self._log_info(f"Validation result: {validation_result}")
                return validation_result
            except json.JSONDecodeError as e:
                if verbose:
                    self._log_error(f"Error parsing validation response: {str(e)}")
                return {
                    "is_valid": True,  # Default to accepting the response
                    "score": 0.75,
                    "reasoning": "Failed to parse validation result",
                    "needs_refinement": False,
                    "refinement_suggestions": "",
                }

        except Exception as e:
            if verbose:
                self._log_error(f"Error during response validation: {str(e)}")
            return {
                "is_valid": True,  # Default to accepting the response
                "score": 0.75,
                "reasoning": f"Validation error: {str(e)}",
                "needs_refinement": False,
                "refinement_suggestions": "",
            }

    @retry_on_error()
    async def run(
        self,
        query: str,
        chat_history: List[ChatMessage] = [],
        verbose: bool = False,
        *args,
        **kwargs,
    ) -> str:
        """Process user request by classifying and delegating to appropriate agent"""
        self.chat_memory.set_initial_long_memories(chat_history)
        self.chat_memory.reset_short_memories()

        if verbose:
            query_preview = (
                str(query)[:100] + "..."
                if len(str(query)) > 100
                else str(query)
            )
            self._log_debug(f"🔍 Starting Router agent for query: {query_preview}")
        try:

            # Classify the request
            selected_agent, confidence, _ = await self._classify_request(query, verbose)
            if not selected_agent:
                if verbose:
                    self._log_warning(
                        "No agents available for execution -> Falling back to default agent response"
                    )
                response = await self.llm.achat(
                    query, chat_history=self.chat_memory.get_long_memories()
                )
                return response

            # If confidence is too low, maybe ask for clarification or fall back to LLM
            if confidence < 0.6:
                if verbose:
                    self._log_warning(
                        f"Low confidence classification ({confidence:.2f}) -> Falling back to default agent response."
                    )
                response = await self.llm.achat(
                    query, chat_history=self.chat_memory.get_long_memories()
                )
                return response

            # Execute the request with the selected agent
            agent_response = await selected_agent.achat(
                query=query, verbose=verbose, chat_history=chat_history, *args, **kwargs
            )
            self.chat_memory.add_short_memory(
                "assistant",
                f"Agent: [{selected_agent.name}] process successfully with result: {agent_response}",
            )
            # Validate the response
            validation_result = await self._validate_response(
                query=query, verbose=verbose
            )
            if verbose:
                self._log_info(f"Validation score: {validation_result['score']:.2f}")

            # Check if refinement is needed
            if (
                validation_result.get("needs_refinement", False)
                and validation_result.get("score", 1.0) < self.validation_threshold
            ):

                if verbose:
                    self._log_debug(f"Refining response based on validation feedback")

                task_todo = f"""
                User Query: {query}
                Validation Feedback: {validation_result}
                """
                final_response = await self._generate_final_response(
                    query=task_todo,
                    verbose=verbose,
                )
            else:
                final_response = agent_response

            return final_response

        except Exception as e:
            self._log_error(f"Error processing request: {str(e)}")
            return (
                "I encountered an error while processing your request. "
                "Please try again or rephrase your question."
            )

    async def achat(
        self,
        query: str,
        verbose: bool = False,
        chat_history: List[ChatMessage] = [],
        *args,
        **kwargs,
    ) -> str:
        """Async chat implementation"""
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

import json
from typing import AsyncGenerator, Generator, List, Any, Optional, Dict, Union
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage
import asyncio

from src.llm import BaseLLM
from ..utils import (
    clean_json_response,
    AgentOptions,
    ExecutionPlan,
    PlanStep,
    retry_on_error,
)
from ..base import BaseAgent


class PlanningAgent(BaseAgent):
    """Agent that creates and executes plans using available tools"""

    def __init__(
        self,
        llm: BaseLLM,
        options: AgentOptions,
        system_prompt: str = "",
        tools: List[FunctionTool] = [],
    ):
        super().__init__(llm, options, system_prompt, tools)
        self.plan = ExecutionPlan()
        self._log_debug(
            f"Agent {self.name}-[Planning] initialized successfully. With {len(tools)} tools."
        )

    async def _evaluate_step_success(
        self, step_num: int, result: Any, verbose: bool
    ) -> bool:
        """Evaluate if the step was successful based on its result."""
        if result is None or (
            isinstance(result, str)
            and ("error" in result.lower() or not result.strip())
        ):
            self._log_warning(
                f"Step {step_num} failed: Result is None or indicates error."
            )
            return False
        if verbose:
            self._log_info(f"Step {step_num} evaluated as successful.")
        return True

    async def _gen_plan(self, query: str,max_steps:int, verbose: bool) -> bool:
        """Generate an optimized execution plan using available tools."""
        prompt = f"""
        Acting as a planning assistant, create a focused plan based on user query using ONLY the tools listed below.

        User query: {query}

        Maximum number of steps: {max_steps}
        
        Available tools:
        {self._format_tool_signatures()}

        Rules:
        1. Use only listed tools.
        2. Use general knowledge if no tool applies.
        3. Return a list of steps.
        4. Each step must specify if it needs a tool.

        Respond using this format:
        {{
        "steps": [
            {{
            "description": "...",
            "requires_tool": true,
            "tool_name": "tool_name"
            }}
        ]
        }}
        """
        if verbose:
            self._log_debug("Generating initial plan...")
        response = await self.llm.achat(
            query=prompt, chat_history=self.chat_memory.get_long_memories()
        )
        plan_data = json.loads(clean_json_response(response))

        def _is_valid_step(step: dict) -> bool:
            if (
                step.get("requires_tool")
                and step.get("tool_name") not in self.tools_dict
            ):
                return False
            return bool(step.get("description"))

        if not plan_data.get("steps") or not all(
            _is_valid_step(step) for step in plan_data["steps"]
        ):
            if verbose:
                self._log_warning(f"Invalid plan generated. tool names are not valid.")

            raise ValueError("Invalid plan generated. tool names are not valid.")
        for step_data in plan_data["steps"]:
            if (
                step_data["requires_tool"]
                and step_data.get("tool_name") not in self.tools_dict
            ):
                # Skip invalid tools
                if verbose:
                    self._log_warning(
                        f"Skipping invalid tool: {step_data.get('tool_name')}"
                    )
                continue
            self.plan.add_step(
                PlanStep(
                    description=step_data["description"],
                    tool_name=step_data.get("tool_name"),
                    requires_tool=step_data.get("requires_tool", True),
                )
            )
        if not self.plan.steps:
            # Fallback if no valid steps were generated
            if verbose:
                self._log_warning("No valid steps in plan, adding default step.")
            self.plan.add_step(
                PlanStep(description=f"Handle query: {query}", requires_tool=False)
            )
        if verbose:
            self._log_info(f"Plan generated with {self.plan.get_num_steps()} steps.")
        return True

    async def _replan(
        self,
        query: str,
        max_steps: int,
        failed_step: PlanStep,
        verbose: bool,
    ) -> bool:
        """Generate a new plan after a step failure."""
        prompt = f"""
        Previous plan failed at step: {failed_step.description}.
        Completed steps: {[f"{memory.role}:{memory.content}" for memory in self.chat_memory.get_short_memories()]}
        Replan for user query: {query}, starting from the failed step.
        """
        if verbose:
            self._log_debug(
                f"Replanning after failure in step: {failed_step.description}"
            )
        return await self._gen_plan(prompt,max_steps, verbose)

    async def _execute_plan(
        self, query: str, max_steps: int, verbose: bool, max_replan: int = 2
    ):
        """Execute the plan step by step with error handling and replanning capability"""
        replan_attempts = 0
        while (
            self.plan.current_step_idx < self.plan.get_num_steps()
            and self.plan.current_step_idx < max_steps
        ):
            step = self.plan.get_current_step()
            if not step:
                self._log_warning(
                    f"No step found at index {self.plan.current_step_idx}. Breaking loop."
                )
                break

            step_num = self.plan.current_step_idx + 1
            if verbose:
                self._log_debug(
                    f"[Step {step_num}/{self.plan.get_num_steps()}]: {step.description} - Starting..."
                )

            try:
                result = await self._execute_step(query, step, verbose)
                success = await self._evaluate_step_success(step_num, result, verbose)
                if not success:
                    if replan_attempts >= max_replan:
                        self._log_warning("Exceeded maximum number of replans.")
                        break
                    if verbose:
                        self._log_warning(f"Step {step_num} failed. Replanning...")
                    # Reset current step index as we have a new plan
                    self.plan.mark_current_step_fail(result)
                    await self._replan(query, step, verbose)
                    self.plan.current_step_idx = 0
                    replan_attempts += 1
                else:
                    # Only increment step index if step was successful
                    self.plan.mark_current_step_complete(result)
                    self.plan.current_step_idx += 1
                self.chat_memory.add_short_memory(
                    "user", str(f"Do the Task {step_num}: {step.description}")
                )
                self.chat_memory.add_short_memory(
                    "assistant",
                    str(
                        f"Task {step_num} {"Successfully!" if success else "Failed!"}, Here is the result: {result}"
                    ),
                )
            except Exception as e:
                if verbose:
                    self._log_error(f"Error executing step {step_num}: {str(e)}")
                raise Exception(
                    f"I apologize, but I encountered an error while processing your request: {str(e)}"
                )

    async def _execute_step(self, query: str, step: PlanStep, verbose: bool) -> Any:
        """Execute a single step of the plan"""
        try:
            if step.requires_tool:
                if verbose:
                    self._log_debug(f"Using tool: {step.tool_name}")
                task_todo = f"""
                User query: {query}
                Step description: {step.description}
                """
                result = await self._execute_tool(
                    task_todo, step.tool_name, step.requires_tool, verbose
                )
                if verbose:
                    result_preview = (
                        str(result)[:100] + "..."
                        if len(str(result)) > 100
                        else str(result)
                    )
                    self._log_info(
                        f"Tool {step.tool_name} executed successfully. Result snippet: {result_preview}"
                    )
            else:
                if verbose:
                    self._log_debug(
                        "No tool need for this step. Processing with general knowledge..."
                    )
                step_task = f"""
                User query: {query}
                Task todo: {step.description}
                """
                result = await self.llm.achat(
                    query=step_task, chat_history=self.chat_memory.get_short_memories()
                )
                if verbose:
                    result_preview = (
                        str(result)[:100] + "..."
                        if len(str(result)) > 100
                        else str(result)
                    )
                    self._log_info(
                        f"Non-tool step completed. Result snippet: {result_preview}"
                    )
            return result
        except Exception as e:
            if verbose:
                self._log_error(f"Error executing step: {str(e)}")
            raise Exception(f"Step execution failed: {str(e)}")

    @retry_on_error()
    async def run(
        self,
        query: str,
        max_steps: int = 5,
        verbose: bool = False,
        chat_history: List[ChatMessage] = [],
    ) -> str:
        """Run the planning agent process on the given query"""
        if verbose:
            query_preview = (
                str(query)[:100] + "..."
                if len(str(query)) > 100
                else str(query)
            )
            self._log_debug(f"🔍 Starting planning agent for query: {query_preview}")

        self.chat_memory.set_initial_long_memories(chat_history)
        self.chat_memory.reset_short_memories()
        await self._gen_plan(query,max_steps, verbose)

        try:
            await self._execute_plan(query, max_steps=max_steps, verbose=verbose)
            if self.plan.get_num_steps() <= 1:
                return self.plan.get_steps()[0].result
            final_response = await self._generate_final_response(query, verbose)
            return final_response

        except Exception as e:
            if verbose:
                self._log_error(f"Error in run: {str(e)}")
            raise Exception(
                f"I apologize, but I encountered an error while processing your request: {str(e)}"
            )

    async def achat(
        self,
        query: str,
        verbose: bool = False,
        chat_history: List[ChatMessage] = [],
        max_steps: int = 3,
        *args,
        **kwargs,
    ) -> str:
        """Async chat interface for the planning agent"""
        # Get additional parameters or use defaults

        if self.callbacks:
            self.callbacks.on_agent_start(self.name)

        try:
            # Run the planning and execution flow
            result = await self.run(
                query=query,
                max_steps=max_steps,
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
        max_steps: int = 3,
        *args,
        **kwargs,
    ) -> str:
        """Synchronous chat interface for the planning agent"""
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
                max_steps=max_steps,
                *args,
                **kwargs,
            )
        )

    async def astream_chat(
        self,
        query: str,
        verbose: bool = False,
        chat_history: Optional[List[ChatMessage]] = None,
        max_steps: int = 3,
        *args,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Async streaming chat interface for the planning agent"""
        # Get additional parameters
        chat_history = chat_history or []
        if self.callbacks:
            self.callbacks.on_agent_start(self.name)

        try:
            result = await self.run(
                query=query,
                max_steps=max_steps,
                verbose=verbose,
                chat_history=chat_history,
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
        max_steps: int = 3,
        *args,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Synchronous streaming chat interface for the planning agent"""
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
            max_steps=max_steps,
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

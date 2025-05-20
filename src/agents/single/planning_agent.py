import json
from typing import AsyncGenerator, Generator, List, Any, Optional, Dict, Union
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage
import asyncio

from src.llm import BaseLLM
from ..design import (
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

    async def _evaluate_step_success(
        self, step_num: int, step: PlanStep, result: Any, verbose: bool
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

    async def _gen_plan(
        self, task: str, verbose: bool
    ) -> ExecutionPlan:
        """Generate an optimized execution plan using available tools."""
        prompt = f"""
        Acting as a planning assistant, create a focused plan using ONLY the tools listed below.

        Task: {task}

        Available tools:
        {self._format_tool_signatures()}

        Rules:
        1. Use only listed tools
        2. Use general knowledge if no suitable tool exists
        3. Keep the plan simple
        4. Use a single step with requires_tool: false if no tools are needed

        Response format:
        {{
            "steps": [
                {{
                    "description": "step description",
                    "requires_tool": true/false,
                    "tool_name": "tool_name or null"
                }}
            ]
        }}
        """
        if verbose:
            self._log_info("Generating initial plan...")
        response = await self.llm.achat(query=prompt, chat_history=self.chat_memory.get_long_memories())
        plan_data = json.loads(clean_json_response(response))
        plan = ExecutionPlan()
        
        def _is_valid_step(step: dict) -> bool:
            if step.get("requires_tool") and step.get("tool_name") not in self.tools_dict:
                return False
            return bool(step.get("description"))
        if not plan_data.get("steps") or not all(_is_valid_step(step) for step in plan_data["steps"]):
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
                    self._log_warning(f"Skipping invalid tool: {step_data.get('tool_name')}")
                continue
            plan.add_step(
                PlanStep(
                    description=step_data["description"],
                    tool_name=step_data.get("tool_name"),
                    requires_tool=step_data.get("requires_tool", True),
                )
            )
        if not plan.steps:
            # Fallback if no valid steps were generated
            if verbose:
                self._log_warning("No valid steps in plan, adding default step.")
            plan.add_step(
                PlanStep(description=f"Handle task: {task}", requires_tool=False)
            )
        if verbose:
            self._log_info(f"Plan generated with {len(plan.steps)} steps.")
        return plan

    async def _replan(
        self,
        task: str,
        failed_step: PlanStep,
        verbose: bool,
    ) -> ExecutionPlan:
        """Generate a new plan after a step failure."""
        prompt = f"""
        Previous plan failed at step: {failed_step.description}.
        Completed steps: {self.chat_memory.get_short_memories()}
        Replan for task: {task}, starting from the failed step.
        """
        if verbose:
            self._log_info(f"Replanning after failure in step: {failed_step.description}")
        return await self._gen_plan(prompt, verbose)

    async def _generate_summary(
        self,
        task: str,
        verbose: bool,
    ) -> str:
        """Generate a coherent summary of the results"""
        SUMMARY_PROMPT = f"""\
        You are responsible for combining Task Results into a summary coherent response.
        Original task: {task}
        Task Results:
        {self.chat_memory.get_short_memories()}
        Please provide a comprehensive response that integrates all the information.
        Be concise and ensure all critical information is included.
        """
            
        if verbose:
            self._log_info("Generating summary...")

        try:
            # Make sure self._output_parser is defined in BaseAgent
            if not self.structured_output:
                result = await self.llm.achat(query=SUMMARY_PROMPT, chat_history=self.chat_memory.get_long_memories())
            else:
                result = await self._output_parser(
                    output=SUMMARY_PROMPT
                )
            if verbose:
                self._log_info(
                    f"Summary generated successfully with final result: {result[:100]}..."
                    if len(str(result)) > 100
                    else f"Summary generated successfully with final result: {result}."
                )
            return result
        except Exception as e:
            if verbose:
                self._log_error(f"Error generating summary: {str(e)}")
            raise e

    async def _execute_plan(
        self,
        plan: ExecutionPlan,
        task: str,
        max_steps: int,
        verbose: bool,
    ):
        """Execute the plan step by step with error handling and replanning capability"""
        while plan.current_step_idx < len(plan.steps) and plan.current_step_idx < max_steps:
            step = plan.get_current_step()
            if not step:
                self._log_warning(f"No step found at index {plan.current_step_idx}. Breaking loop.")
                break
            
            step_num = plan.current_step_idx + 1
            if verbose:
                self._log_info(
                    f"\nStep {step_num}/{len(plan.steps)}: {step.description}"
                )

            try:
                result = await self._execute_step(task, step, verbose)
                success = await self._evaluate_step_success(
                    step_num, step, result, verbose
                )
                if not success:
                    if verbose:
                        self._log_warning(f"Step {step_num} failed. Replanning...")
                    # Reset current step index as we have a new plan
                    plan.mark_current_step_fail(result)
                    plan = await self._replan(task, step, verbose)
                    plan.current_step_idx = 0
                else:
                    # Only increment step index if step was successful
                    plan.mark_current_step_complete(result)
                    plan.current_step_idx += 1
                self.chat_memory.add_short_memory("assistant", str(f"Step {plan.current_step_idx+1} ({step.description}): {step.result}"))
            except Exception as e:
                if verbose:
                    self._log_error(f"Error executing step {step_num}: {str(e)}")
                raise Exception(f"I apologize, but I encountered an error while processing your request: {str(e)}")


    async def _execute_step(
        self, task: str, step: PlanStep, verbose: bool
    ) -> Any:
        """Execute a single step of the plan"""
        try:
            if step.requires_tool:
                if verbose:
                    self._log_info(f"Using tool: {step.tool_name}")
                result = await self._execute_tool(
                    task,
                    step.tool_name,
                    step.description,
                    step.requires_tool
                )
                if verbose:
                    result_preview = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                    self._log_info(f"Tool {step.tool_name} executed successfully. Result snippet: {result_preview}")
            else:
                if verbose:
                    self._log_info("Processing with general knowledge...")
                result = await self.llm.achat(
                    query=step.description, chat_history=self.chat_memory.get_short_memories()
                )
                if verbose:
                    result_preview = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                    self._log_info(f"Non-tool step completed. Result snippet: {result_preview}")
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
            self._log_info(f"🔍 Starting planning agent for query: {query}")

        self.chat_memory.set_initial_long_memories(chat_history)
        plan = await self._gen_plan(query, verbose)

        try:
            await self._execute_plan(
                plan, query, max_steps=max_steps, verbose=verbose
            )
            final_summary = await self._generate_summary(
                query, verbose
            )
            return final_summary

        except Exception as e:
            if verbose:
                self._log_error(f"Error in run: {str(e)}")
            raise Exception(f"I apologize, but I encountered an error while processing your request: {str(e)}")

    async def achat(
        self,
        query: str,
        verbose: bool = False,
        chat_history: List[ChatMessage] = [],
        *args,
        **kwargs,
    ) -> str:
        """Async chat interface for the planning agent"""
        # Get additional parameters or use defaults
        max_steps = kwargs.get("max_steps", 3)

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
        """Async streaming chat interface for the planning agent"""
        # Get additional parameters
        max_steps = kwargs.get("max_steps", 3)
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
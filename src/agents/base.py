from abc import ABC, abstractmethod
import json
from typing import Any, Generator, List, Optional
from pydantic import BaseModel
from llama_index.core.llms import ChatMessage
from typing import AsyncGenerator
from llama_index.core.tools import FunctionTool
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import LLMTextCompletionProgram

from src.logger import get_formatted_logger
from .design import (
    clean_json_response,
    AgentCallbacks,
    AgentOptions,
    AgentResponse,
    ChatMemory,
    retry_on_json_parse_error,
)
from src.llm import BaseLLM


class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(
        self,
        llm: BaseLLM,
        options: AgentOptions,
        system_prompt: str = "",
        tools: List[FunctionTool] = [],
    ):
        self.llm = llm
        if system_prompt:
            self.llm._set_system_prompt(system_prompt)
        self.system_prompt = self.llm._get_system_prompt()
        self.name = options.name
        self.description = options.description
        self.id = options.id or self._generate_id_from_name(self.name)
        self.region = options.region
        self.save_chat = options.save_chat
        self.callbacks = options.callbacks or AgentCallbacks()
        self.structured_output = options.structured_output
        self.tools = tools
        self.tools_dict = {tool.metadata.name: tool for tool in tools}
        self.logger = get_formatted_logger(__file__)
        self.chat_memory = ChatMemory()

    @staticmethod
    def _generate_id_from_name(name: str) -> str:
        import re

        # Remove special characters and replace spaces with hyphens
        key = re.sub(r"[^a-zA-Z\s-]", "", name)
        key = re.sub(r"\s+", "-", key)
        return key.lower()

    def _get_config(self) -> dict[str, Any]:
        """Get detailed config of the agent"""
        return {
            "name": self.name,
            "description": self.description,
            "id": self.id,
            "llm": self.llm._get_model_config(),
            "structured_output": self.structured_output,
        }

    def _log_info(self, message: str) -> None:
        self.logger.info(f"[Agent: {self.name}] - [ID: {self.id}] - {message}")

    def _log_error(self, message: str) -> None:
        self.logger.error(f"[Agent: {self.name}] - [ID: {self.id}] - {message}")

    def _log_debug(self, message: str) -> None:
        self.logger.debug(f"[Agent: {self.name}] - [ID: {self.id}] - {message}")

    def _log_warning(self, message: str) -> None:
        self.logger.warning(f"[Agent: {self.name}] - [ID: {self.id}] - {message}")

    def _create_system_message(self, prompt: str) -> ChatMessage:
        """Create a system message with the given prompt"""
        return ChatMessage(role="system", content=prompt)

    def _get_output_schema(self) -> str:
        """Get JSON schema of output model if available"""
        if not self.structured_output:
            self.logger.warning(f"Output schema not found")
            return "[No specific output schema]."

        try:
            schema = self.structured_output.model_json_schema()
            self.logger.info(f"Parsed output schema successfully: {schema}")
            return json.dumps(schema, indent=4)
        except Exception as e:
            self.logger.error(f"Error getting output schema: {str(e)}")
            return "[No specific output schema]."

    @retry_on_json_parse_error()
    async def _output_parser(
        self,
        output: str,
    ) -> str:
        final_output = ""
        try:
            program = LLMTextCompletionProgram.from_defaults(
                output_parser=PydanticOutputParser(output_cls=self.structured_output),
                llm=self.llm._get_model(),
                prompt_template_str=output,
                verbose=True,
            )

            parsed_output = program()

            if isinstance(parsed_output, BaseModel):
                final_output = parsed_output.model_dump_json()
            else:
                self.logger.warning(
                    f"⚠️ Unexpected output type: {type(parsed_output)}. Attempting to clean."
                )
                final_output = clean_json_response(str(parsed_output))

        except Exception as e:
            self.logger.error(f"❌ Parsing failed: {str(e)}. Fallback to raw LLM call.")
            try:
                structured_schema_prompt = f"""{output}\nIf an output schema was provided, please ensure your response conforms to this structure:\n{self._get_output_schema()}"""
                fallback_output = await self.llm.achat(
                    query=structured_schema_prompt,
                    chat_history=self.chat_memory.get_all_memories(),
                )
                fallback_output = clean_json_response(fallback_output)
                final_output = self.structured_output.model_validate_json(
                    fallback_output
                ).model_dump_json()
            except Exception as e:
                self.logger.error(
                    f"❌ Fallback parsing failed: {str(e)}. Returning raw fallback output."
                )
                final_output = fallback_output

        return final_output

    def _format_tool_signatures(self) -> str:
        """Format all tool signatures into a string format LLM can understand"""
        if not self.tools:
            return (
                "No tools are available. Respond based on your general knowledge only."
            )

        tool_descriptions = []
        for tool in self.tools:
            metadata = tool.metadata
            parameters = metadata.get_parameters_dict()

            tool_descriptions.append(
                f"""
                Function: {metadata.name}
                Description: {metadata.description}
                Parameters: {json.dumps(parameters, indent=2)}
                """
            )

        return "\n".join(tool_descriptions)

    async def _argument_parser(
        self,
        task: str,
        tool_name: str,
        tool: FunctionTool,
        verbose: bool = False,
    ) -> Optional[Any]:
        prompt = f"""
        User task context: {task}
        Based on the User Task you need to generate the EXACT parameters to call the tool '{tool_name}'.
        Prioritize information from the context if it directly relates to the required parameters.
        Tool: {tool_name}
        Tool description: {tool.metadata.description}
        
        Tool specification (REQUIRED PARAMETERS and their types/descriptions):
        {json.dumps(tool.metadata.get_parameters_dict(), indent=2)}
        
        Important rules for generating parameters:
        1. ONLY generate parameters that are part of the 'Tool specification'.
        2. Ensure the parameter values strictly match the expected data types and formats.
        3. Extract necessary information from the 'User task context'.
        4. If a parameter is optional and not needed, omit it.
        5. If a required parameter cannot be confidently determined, use a placeholder like "[UNDEFINED_PARAMETER]" and explain why.

        Response format:
        {{
            "arguments": {{
                // parameter names and values matching the specification EXACTLY
            }}
        }}
        """
        try:
            if verbose:
                self._log_debug(f"Generating arguments for tool '{tool_name}'...")

            response = await self.llm.achat(
                query=prompt, chat_history=self.chat_memory.get_short_memories()
            )

            response = clean_json_response(response)
            params = json.loads(response)

            if "arguments" not in params or not isinstance(params["arguments"], dict):
                raise ValueError(
                    "LLM did not return arguments in the expected format: {'arguments': {...}}"
                )

            return params
        except Exception as e:
            if verbose:
                self._log_error(
                    f"An unexpected error occurred while generating arguments for tool '{tool_name}': {str(e)}"
                )
            return None

    async def _execute_tool(
        self,
        task: str,
        tool_name: str,
        requires_tool: bool,
        verbose: bool = False,
    ) -> Optional[Any]:
        """Execute a tool with better error handling and context-awareness"""
        if not requires_tool or not tool_name:
            return None
        tool = self.tools_dict.get(tool_name)
        if not tool:
            if verbose:
                self._log_warning(
                    f"Attempted to execute non-existent tool: {tool_name}"
                )
            return None
        result = None
        try:
            params = await self._argument_parser(task, tool_name, tool, verbose)
            if params and "arguments" in params:
                if verbose:
                    self._log_debug(
                        f"Executing tool '{tool_name}' with arguments: {params['arguments']}"
                    )
                result = await tool.acall(**params["arguments"])
                if verbose:
                    self._log_info(
                        f"Tool '{tool_name}' executed. Result: {str(result)[:200]}..."
                    )
                return result
            else:
                self._log_warning(
                    f"Could not generate valid arguments for tool '{tool_name}'. Skipping execution."
                )
                return result
        except Exception as e:
            if verbose:
                self._log_error(
                    f"Error generating arguments or executing tool '{tool_name}': {str(e)}"
                )
            if requires_tool:
                raise
            return result

    async def _generate_final_response(
        self,
        query: str,
        verbose: bool,
    ) -> str:
        """Generate a coherent final response of the results"""

        FINAL_RESPONSE_PROMPT = f"""\
        You are a helpful assistant whose role is to synthesize the information gathered from previous tasks to provide a comprehensive and concise answer to the user's query.
        User query: {query}
        Task Results: {self.chat_memory.get_short_memories()}
        Answer the user's query directly using the information provided in the Task Results.
        Be concise and ensure all critical information is included in your response.
        Do not include any introductory just answer normalization response.
        """

        if verbose:
            self._log_debug("Generating final_response...")

        try:
            # Make sure self._output_parser is defined in BaseAgent
            if not self.structured_output:
                result = await self.llm.achat(
                    query=FINAL_RESPONSE_PROMPT,
                    chat_history=self.chat_memory.get_long_memories(),
                )
            else:
                result = await self._output_parser(output=FINAL_RESPONSE_PROMPT)
            if verbose:
                self._log_info(
                    f"Final response generated successfully with final result: {result[:100]}..."
                    if len(str(result)) > 100
                    else f"Final response generated successfully with final result: {result}."
                )
            return result
        except Exception as e:
            if verbose:
                self._log_error(f"Error generating summary: {str(e)}")
            raise e

    @abstractmethod
    def chat(
        self,
        query: str,
        verbose: bool = False,
        chat_history: List[ChatMessage] = [],
        *args,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    async def achat(
        self,
        query: str,
        verbose: bool = False,
        chat_history: List[ChatMessage] = [],
        *args,
        **kwargs,
    ) -> str:
        """Main execution method that must be implemented by all agents"""
        pass

    @abstractmethod
    def stream_chat(
        self,
        query: str,
        verbose: bool = False,
        chat_history: Optional[List[ChatMessage]] = None,
        *args,
        **kwargs,
    ) -> Generator[str, None, None]:
        pass

    @abstractmethod
    async def astream_chat(
        self,
        query: str,
        verbose: bool = False,
        chat_history: Optional[List[ChatMessage]] = None,
        *args,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        pass

    def is_streaming_enabled(self) -> bool:
        return True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

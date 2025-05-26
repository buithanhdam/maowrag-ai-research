
from typing import Any, List, Optional, Dict, Union, Type
from llama_index.core.llms import ChatMessage
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel


# Base Types and Data Classes
class AgentType(Enum):
    DEFAULT = "DEFAULT"
    CODING = "CODING"
    REACT = "REACT"
    REFLECTION = "REFLECTION"
    PLANNING = "PLANNING"

class AgentIterationProcess(BaseModel):
    idx: int
    result: str
    time_taken: float
    token_factory: Dict[str, Any] = None

class AgentProcessingResult(BaseModel):
    session_id: str
    agent_id: str
    iteration_results: List[AgentIterationProcess]
    additional_params: Dict[str, Any] = field(default_factory=dict)

class AgentResponse:
    metadata: AgentProcessingResult
    message: str
    streaming: bool
    
    def __str__(self):
        return self.message
    def __repr__(self) -> str:
        return f"AgentResponse(message='{self.message[:50]}...', metadata={self.metadata})"
    @property
    def usage_metadata(self) -> dict:
        return self.metadata


class AgentCallbacks:
    def on_llm_new_token(self, token: str) -> None:
        pass
    
    def on_agent_start(self, agent_id: str) -> None:
        pass
    
    def on_agent_end(self, agent_id: str) -> None:
        pass


@dataclass
class AgentOptions:
    name: str
    description: str
    id: Optional[str] = None
    region: Optional[str] = None
    save_chat: bool = True
    callbacks: Optional[AgentCallbacks] = None
    structured_output: Optional[Type[BaseModel]] = None


class ChatMemory:
    def __init__(
        self,
        long_memories: List[ChatMessage] = None,
        short_memories: List[ChatMessage] = None,
        max_length: int = 10
    ):
        self.long_memories = long_memories or []
        self.short_memories = short_memories or []
        self.max_length = max_length
    def set_max_length(self, max_length: int) -> None:
        self.max_length = max_length
    def get_max_length(self) -> int:
        return self.max_length
    def add_long_memory(self, role: str, content: str) -> None:
        self.long_memories.append(ChatMessage(role=role, content=content))
    def set_initial_long_memories(self, long_memories: List[ChatMessage]):
        self.long_memories = long_memories
    def get_long_memories(self) -> List[ChatMessage]:
        return self.long_memories
    def add_short_memory(self, role: str, content: str) -> None:
        self.short_memories.append(ChatMessage(role=role, content=content))
        if len(self.short_memories) > self.max_length:
            # Keep the most recent short_memories
            self.short_memories = self.short_memories[-self.max_length:]
    def get_short_memories(self) -> List[ChatMessage]:
        return self.short_memories
    def reset_short_memories(self) -> None:
        self.short_memories = []
    def get_all_memories(self) -> List[ChatMessage]:
        return self.long_memories + self.short_memories

# For planning agent
@dataclass
class PlanStep:
    description: str = None
    requires_tool: bool = False
    tool_name: str = None
    completed: bool = False
    result: Any = None


class ExecutionPlan:
    def __init__(self):
        self.steps: List[PlanStep] = []
        self.current_step_idx = 0
        
    def add_step(self, step: PlanStep) -> None:
        self.steps.append(step)
    def get_steps(self) -> List[PlanStep]:
        return self.steps
    def get_num_steps(self) -> int:
        return len(self.steps)
    def get_current_step(self) -> Optional[PlanStep]:
        if self.current_step_idx < len(self.steps):
            return self.steps[self.current_step_idx]
        return None
        
    def mark_current_step_complete(self, result: Any = None) -> None:
        if self.current_step_idx < len(self.steps):
            self.steps[self.current_step_idx].completed = True
            self.steps[self.current_step_idx].result = result
    def mark_current_step_fail(self, result: Any = None) -> None:
        if self.current_step_idx < len(self.steps):
            self.steps[self.current_step_idx].completed = False
            self.steps[self.current_step_idx].result = result
    def is_complete(self) -> bool:
        return self.current_step_idx >= len(self.steps)
        
    def get_progress(self) -> str:
        completed = sum(1 for step in self.steps if step.completed)
        return f"Progress: {completed}/{len(self.steps)} steps completed"
import asyncio
from src.agents import (ReflectionAgent,
                        ReActAgent,
                        AgentOptions,
                        ManagerAgent)
from src.tools import get_weather_tool,search_web_tool
from src.llm import UnifiedLLM
from src.config import LLMType, global_config


async def test_reflection_async():
    llm= UnifiedLLM(model_name=LLMType.GEMINI)
    
    reflection_agent = ReflectionAgent(llm, AgentOptions(
        id="reflection1",
        name="Reflection Assistant",
        description="Helps with information base on LLM"
    ))
    async with reflection_agent as agent:
        result = await agent.achat(
            query="leonel messi most successful achievement in his career",
            verbose=1
        )
        print("reflection agent commplete: ",result)
async def test_planning_async():
    # Initialize agent
        # Create tools
    llm= UnifiedLLM(model_name=LLMType.GEMINI)
    planning_agent = ReActAgent(llm, AgentOptions(
        id="react1",
        name="Planning Assistant",
        description="Assists with project planning, task breakdown, and weather information"
    ),tools=[get_weather_tool,search_web_tool])
    
    async with planning_agent as agent:
        result = await agent.achat(
            query="what is new movie by disney and it profit around the world?",
            verbose=True
        )
        print("Planning agent commplete: ",result)

async def test_manager_agent():
    llm= UnifiedLLM(model_name=LLMType.GEMINI)
    
    reflection_agent = ReflectionAgent(llm, AgentOptions(
        id="reflection1",
        name="Reflection Assistant",
        description="Helps with information base on LLM"
    ))
    
    planning_agent = ReActAgent(llm, AgentOptions(
        id="react1",
        name="Planning Assistant",
        description="Assists with project planning, task breakdown, and weather information"
    ),tools=[get_weather_tool])
    
    
    async with ManagerAgent(llm, AgentOptions(
        name="Manager",
        description="Routes requests to specialized agents"
    )) as manager:
        manager.register_agent(reflection_agent)
        manager.register_agent(planning_agent)
        response = await manager.achat(
            query="Can you help me about today weather?",
            verbose=True
        )
        print("Manager agent commplete: ",response)
    
if __name__ == "__main__":
    asyncio.run(test_planning_async())
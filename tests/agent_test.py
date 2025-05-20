import asyncio
from src.agents import (ReflectionAgent,
                        PlanningAgent,
                        AgentOptions)
from src.tools import ToolManager
from src.llm import BaseLLM
from src.config import LLMProviderType, global_config


async def test_reflection_async():
    llm = BaseLLM(provider=LLMProviderType.GOOGLE)
    
    reflection_agent = ReflectionAgent(llm, AgentOptions(
        name="Reflection Assistant",
        description="Helps with information base on LLM"),
                                       tools=ToolManager.get_weather_tools()+ToolManager.get_search_tools()
    )
    async with reflection_agent as agent:
        result = await agent.achat(
            query="what is new movie by disney and it profit around the world?",
            verbose=1
        )
        print("reflection agent commplete: ",result)
async def test_planning_async():
    # Initialize agent
        # Create tools
    llm = BaseLLM(provider=LLMProviderType.GOOGLE)
    planning_agent = PlanningAgent(llm, AgentOptions(
        id="react1",
        name="Planning Assistant",
        description="Assists with project planning, task breakdown, and weather information"
    ),tools=ToolManager.get_weather_tools()+ToolManager.get_search_tools())
    
    async with planning_agent as agent:
        result = await agent.achat(
            query="what is new movie by disney and it profit around the world?",
            verbose=True
        )
        print("Planning agent commplete: ",result)

# async def test_manager_agent():
#     llm= UnifiedLLM(model_name=LLMType.GEMINI)
    
#     reflection_agent = ReflectionAgent(llm, AgentOptions(
#         id="reflection1",
#         name="Reflection Assistant",
#         description="Helps with information base on LLM"
#     ))
    
#     planning_agent = ReActAgent(llm, AgentOptions(
#         id="react1",
#         name="Planning Assistant",
#         description="Assists with project planning, task breakdown, and weather information"
#     ),tools=[get_weather_tool])
    
    
#     async with ManagerAgent(llm, AgentOptions(
#         name="Manager",
#         description="Routes requests to specialized agents"
#     )) as manager:
#         manager.register_agent(reflection_agent)
#         manager.register_agent(planning_agent)
#         response = await manager.achat(
#             query="Can you help me about today weather?",
#             verbose=True
#         )
#         print("Manager agent commplete: ",response)
    
if __name__ == "__main__":
    asyncio.run(test_planning_async())
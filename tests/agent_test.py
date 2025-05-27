import asyncio
from src.agents import (ReflectionAgent,
                        PlanningAgent,RouterAgent, ParallelAgent,
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
            query="""
            Could you list the Disney movies released in 2024, along with their estimated total worldwide profit?
            And what are the upcoming Disney animated movies scheduled for release in 2025, and are there any early projections for their potential global box office success?
            """,
            verbose=True,
            n_iterations=2
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
            query="""
            Could you list the Disney movies released in 2024, along with their estimated total worldwide profit?
            And what are the upcoming Disney animated movies scheduled for release in 2025, and are there any early projections for their potential global box office success?
            """,
            verbose=True,
            max_steps=3
        )
        print("Planning agent commplete: ",result)

async def test_multi_agent():
    llm = BaseLLM(provider=LLMProviderType.GOOGLE)

    planning_agent = PlanningAgent(
        llm,
        AgentOptions(
            name="Travel Agent",
            description="Agent can planning for travel, visit around the world"
        ),
        tools=ToolManager.get_weather_tools()+ToolManager.get_search_tools(),
        system_prompt="You are a travel assistant, you can help me with travel information"
    )
    
    reflection_agent = ReflectionAgent(
        llm=llm,
        options=AgentOptions(
            name="Food Agent",
            description="Helps with information base on Food"
        ),
        tools=ToolManager.get_weather_tools()+ToolManager.get_search_tools(),
        system_prompt="You are a food assistant, you can help me with food information"
    )
    
    multi_agent = ParallelAgent(
        llm,
        AgentOptions(
            name="Parallel Agent",
            description="Agent can help me with parallel tasks"
        ),
        tools=ToolManager.get_weather_tools()+ToolManager.get_search_tools(),
        system_prompt="You are a Manager agent, you can help me about travel, food"
    )
    # multi_agent = RouterAgent(
    #     llm,
    #     AgentOptions(
    #         name="Router Agent",
    #         description="Routes requests to specialized agents"
    #     ),
    #     tools=ToolManager.get_weather_tools()+ToolManager.get_search_tools(),
    #     system_prompt="You are a Manager agent, you can help me about travel, food"
    # )
    
    multi_agent._register_agent(reflection_agent)
    multi_agent._register_agent(planning_agent)
    response = await multi_agent.achat(
        query="Can you help me about healthy food and where to travel in India?",
        verbose=True
    )
    print("Manager agent commplete: ",response)
    
if __name__ == "__main__":
    asyncio.run(test_planning_async())
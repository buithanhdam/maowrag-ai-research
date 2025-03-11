import asyncio
from src.agents import (ReflectionAgent,
                        PlanningAgent,
                        AgentOptions,
                        ManagerAgent)
from src.tools.tool_manager import create_function_tool
from src.agents.llm import UnifiedLLM

def get_weather(location: str, unit: str = "celsius") -> dict:
        """Get current weather for a location
        
        Args:
            location: City name or coordinates
            unit: Temperature unit (celsius/fahrenheit)
        """
        # Implementation here
        return {
            "temperature": 25,
            "weather_description": "Sunny",
            "humidity": 60,
            "wind_speed": 10
        }

async def test_reflection_async():
    async with ReflectionAgent(llm= UnifiedLLM(model_name="gemini")) as agent:
        result = await agent.run(
            user_msg="leonel messi most successful achievement in his career",
            verbose=1
        )
        print("reflection agent commplete: ",result)
async def test_planning_async():
    # Initialize agent
        # Create tools
    weather_tool = create_function_tool(
        get_weather,
        name="get_weather",
        description="Get current weather information for a location"
    )
    async with PlanningAgent(llm= UnifiedLLM(model_name="gemini"),tools=[weather_tool]) as agent:
        result = await agent.run(
            task="What's the weather in Hanoi?",
            verbose=True
        )
        print("Planning agent commplete: ",result)

async def test_manager_agent():
    llm= UnifiedLLM(model_name="gemini")
    
    reflection_agent = ReflectionAgent(llm, AgentOptions(
        name="Reflection Assistant",
        description="Helps with information base on LLM"
    ))
    # Create tools
    weather_tool = create_function_tool(
        get_weather,
        name="get_weather",
        description="Get current weather information for a location"
    )
    
    planning_agent = PlanningAgent(llm, AgentOptions(
        name="Planning Assistant",
        description="Assists with project planning, task breakdown, and a weather tool"
    ),tools=[weather_tool])
    
    
    async with ManagerAgent(llm, AgentOptions(
        name="Manager",
        description="Routes requests to specialized agents"
    )) as manager:
        manager.register_agent(reflection_agent)
        manager.register_agent(planning_agent)
        response = await manager.run(
            query="Can you help me about today weather?",
            verbose=True
        )
        print("Manager agent commplete: ",response)
    
if __name__ == "__main__":
    asyncio.run(test_manager_agent())
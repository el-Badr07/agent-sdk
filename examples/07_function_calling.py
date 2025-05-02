"""Example demonstrating function calling with LLM provider adapters.

This example shows how to use the function calling capabilities of LLM provider adapters.
It demonstrates:
1. Defining functions that the LLM can call
2. Using the function_call method to let the LLM decide which function to call
3. Handling the function call results

Prerequisites:
- Install the SDK: pip install -e .
- Set up appropriate API keys in environment variables (e.g., OPENAI_API_KEY)
"""

import datetime
import json
import os
from typing import Any, Dict, List, Optional

import dotenv
from hotlm_core.schema import HumanMessage, SystemMessage
from hotlm_integrations import create_chat_model

# Load environment variables from a .env file if present
dotenv.load_dotenv()

# Sample functions that the LLM can call
def get_current_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """Get the current weather for a location."""
    # Simulate getting weather information
    # In a real application, this would call a weather API
    weather_data = {
        "location": location,
        "temperature": 22.5 if unit == "celsius" else 72.5,
        "unit": unit,
        "condition": "Sunny",
        "humidity": 45,
        "wind_speed": 10,
        "forecast": ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Sunny"]
    }
    return weather_data

def calculate_mortgage(principal: float, interest_rate: float, years: int) -> Dict[str, Any]:
    """Calculate monthly mortgage payment."""
    # Convert annual interest rate to monthly rate
    monthly_rate = interest_rate / 100 / 12
    # Number of monthly payments
    n_payments = years * 12
    # Calculate monthly payment using the mortgage formula
    if monthly_rate == 0:
        monthly_payment = principal / n_payments
    else:
        monthly_payment = principal * (monthly_rate * (1 + monthly_rate) ** n_payments) / ((1 + monthly_rate) ** n_payments - 1)
    
    total_payment = monthly_payment * n_payments
    total_interest = total_payment - principal
    
    return {
        "monthly_payment": round(monthly_payment, 2),
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_interest, 2),
        "principal": principal,
        "interest_rate": interest_rate,
        "years": years
    }

# Define the function schemas in OpenAI format
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use (celsius or fahrenheit)"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculate_mortgage",
        "description": "Calculate monthly mortgage payment and total interest paid",
        "parameters": {
            "type": "object",
            "properties": {
                "principal": {
                    "type": "number",
                    "description": "The loan amount in dollars"
                },
                "interest_rate": {
                    "type": "number",
                    "description": "The annual interest rate as a percentage (e.g., 5.5 for 5.5%)"
                },
                "years": {
                    "type": "integer",
                    "description": "The loan term in years"
                }
            },
            "required": ["principal", "interest_rate", "years"]
        }
    }
]

# Function dispatch mapping
function_map = {
    "get_current_weather": get_current_weather,
    "calculate_mortgage": calculate_mortgage
}

def handle_function_call(function_result):
    """Handle the result of a function call and execute the appropriate function."""
    function_name = function_result.get("function_name")
    if not function_name:
        # Model chose not to call a function
        return function_result.get("function_response", "No function called.")
    
    function_args = function_result.get("function_arguments", {})
    
    if function_name not in function_map:
        return f"Error: Function '{function_name}' not found."
    
    try:
        # Call the appropriate function
        function_to_call = function_map[function_name]
        result = function_to_call(**function_args)
        return result
    except Exception as e:
        return f"Error executing function '{function_name}': {str(e)}"

def run_example_with_function_calling(model, user_input):
    """Run an example using function calling capabilities."""
    # Define messages
    messages = [
        SystemMessage(content="You are a helpful assistant that can get current weather information and calculate mortgage payments."),
        HumanMessage(content=user_input)
    ]
    
    # Generate a response with function calling
    try:
        # Check if function calling is supported
        if not model.supports_function_calling():
            print(f"Function calling is not supported by {model._llm_type}")
            return
        
        # Let the LLM decide which function to call (if any)
        function_result = model.function_call(
            messages=messages,
            functions=functions
        )
        
        print(f"LLM recommended function: {function_result.get('function_name')}")
        if function_result.get('function_name'):
            print(f"With arguments: {json.dumps(function_result.get('function_arguments', {}), indent=2)}")
        
        # Execute the function if one was called
        result = handle_function_call(function_result)
        
        # Add the function result to messages and get a final response
        messages.append(SystemMessage(content=f"Function result: {json.dumps(result)}"))
        messages.append(HumanMessage(content="Can you explain this result to me?"))
        
        # Get a final response from the LLM
        final_result = model.invoke(messages)
        
        print("\nFunction Result:")
        print(json.dumps(result, indent=2))
        print("\nFinal LLM Response:")
        print(final_result.content)
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run function calling examples with LLM providers that support it."""
    print("Function Calling with LLM Provider Adapters Example")
    print("=================================================")
    
    # Check for API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your API key in the environment or a .env file.")
        return
    
    # Create OpenAI model that supports function calling
    print("\nCreating OpenAI model with function calling support...")
    model = create_chat_model(model_name="gpt-4")
    
    # Example 1: Weather query
    print("\nExample 1: Weather query")
    run_example_with_function_calling(
        model, 
        "What's the weather like in Seattle right now?"
    )
    
    # Example 2: Mortgage calculation
    print("\nExample 2: Mortgage calculation")
    run_example_with_function_calling(
        model,
        "How much would my monthly payment be for a $300,000 house with a 30-year mortgage at 5.5% interest rate?"
    )
    
    # Example 3: No function needed
    print("\nExample 3: No function needed")
    run_example_with_function_calling(
        model,
        "What are the different types of mortgage loans available?"
    )

if __name__ == "__main__":
    main()
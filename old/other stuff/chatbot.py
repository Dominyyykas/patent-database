import streamlit as st
from openai import OpenAI
import os
from function_calling import calculator, number_to_emoji_sum, handle_math_problem, get_available_apartments

# Set page configuration
st.set_page_config(
    page_title="ChatGPT Bot",
    page_icon="🤖"
)

# Initialize OpenAI client
client = OpenAI(api_key='YOUR API KEY')

# Define the available functions
functions = [
    {
        "name": "number_to_emoji_sum",
        "description": "Convert a list of integers to their emoji representation and append the sum as emojis.",
        "parameters": {
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "A list of integers to convert to emojis and sum."
                }
            },
            "required": ["numbers"]
        }
    },
    {
        "name": "handle_math_problem",
        "description": "Handle a mathematical problem by checking if an answer exists or saving it as pending.",
        "parameters": {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The mathematical problem to handle"
                }
            },
            "required": ["problem"]
        }
    },
    {
        "name": "get_available_apartments",
        "description": "Retrieve available apartments from the database based on filters.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Optional city filter"
                },
                "min_price": {
                    "type": "number",
                    "description": "Optional minimum price filter"
                },
                "max_price": {
                    "type": "number",
                    "description": "Optional maximum price filter"
                }
            }
        }
    }
]

# [1, 5 ,3 4, 5]


# Set up session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat title
st.title("ChatGPT Bot 🤖")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            functions=functions,
            function_call="auto",
            stream=True
        )
        
        # Stream the response
        full_response = ""
        function_call = None
        
        for chunk in response:
            if chunk.choices[0].delta.function_call:
                if function_call is None:
                    function_call = {
                        "name": chunk.choices[0].delta.function_call.name,
                        "arguments": chunk.choices[0].delta.function_call.arguments
                    }
                else:
                    function_call["arguments"] += chunk.choices[0].delta.function_call.arguments
            elif chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
        
        # Handle function calling
        if function_call:
            import json
            args = json.loads(function_call["arguments"])
            
            if function_call["name"] == "number_to_emoji_sum":
                result = number_to_emoji_sum(args["numbers"])
                full_response = f"Emoji sum: {result}"
            elif function_call["name"] == "handle_math_problem":
                result = handle_math_problem(args["problem"])
                full_response = result
            elif function_call["name"] == "get_available_apartments":
                result = get_available_apartments(
                    args.get("city"),
                    args.get("min_price"),
                    args.get("max_price")
                )
                if isinstance(result, list):
                    if not result:
                        full_response = "No apartments found matching your criteria."
                    else:
                        full_response = "Available apartments:\n\n"
                        for apt in result:
                            full_response += f"• {apt['title']} in {apt['city']}\n"
                            full_response += f"  Price: ${apt['price_per_night']}/night\n"
                            full_response += f"  {apt['bedrooms']} bedrooms, {apt['bathrooms']} bathrooms\n"
                            full_response += f"  Available: {apt['available_from']} to {apt['available_to']}\n\n"
                else:
                    full_response = result
        
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response}) 
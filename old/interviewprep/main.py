from dotenv import load_dotenv
import os
from openai import OpenAI
import sys

def test_openai_connection():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        sys.exit(1)
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Test API connection with a simple completion request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello! This is a test message."}
            ]
        )
        
        print("\n" + "="*50)
        print("✅ Successfully connected to OpenAI API!")
        print("="*50)
        print("\nModel used:", "gpt-3.5-turbo")
        print("\nAPI Response:")
        print("-"*20)
        print(response.choices[0].message.content)
        print("-"*20 + "\n")
        
    except Exception as e:
        print("\n❌ Error connecting to OpenAI API:", str(e))
        sys.exit(1)

if __name__ == "__main__":
    test_openai_connection() 
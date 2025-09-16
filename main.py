from src.graph.builder import build_graph
from src.graph.state import State
from src.model.llm import LLM
# Import your actual tool functions
from src.tools.math_tools import get_math_tool
from src.tools.search_tools import get_search_tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

def main():
    # Load variables from .env file
    load_dotenv()

    # Initialize the LLM.
    # NOTE: Corrected model name to 'gemini-1.5-flash' and removed the incorrect 'base_url'.
    # The library will handle the correct API endpoint automatically.
    # Khởi tạo trạng thái ban đầu
    llm = LLM(
        model="gemini-2.5-flash",
        temperature=0.5,
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url=["https://generativelanguage.googleapis.com/v1beta/openai/"]
    )

    # Initialize tools
    tools = [get_search_tool(), get_math_tool()]

    # Build the graph once
    graph = build_graph()

    # List to store the history of messages
    messages = []

    print("Chatbot is ready. Type 'exit' or 'quit' to end the conversation.")
    while True:
        # Get user input from the console
        question = input("You: ")

        # Check if the user wants to exit
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Append the new user message to the history
        messages.append(HumanMessage(content=question))

        # Define the initial state for this turn of the conversation
        initial_state: State = {
            "llm": llm,
            "tools": tools,
            "messages": messages,
        }

        # Run the graph with the current state
        final_state = graph.invoke(initial_state)

        # The graph should return the updated message list (including the AI's response)
        # We update our history for the next turn
        messages = final_state.get("messages", [])
        
        # Get the latest message from the assistant to print as the answer
        answer = messages[-1].content if messages else "Tôi không có đủ thông tin để trả lời."

        # Print the final result for this turn
        print(f"Bot: {answer}")


if __name__ == "__main__":
    main()
from src.graph.builder import build_graph
from src.graph.state import State
from src.model.llm import LLM
# Import your actual tool functions
from src.tools.math_tools import get_math_tool
from src.tools.search_tools import get_search_tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore
from langchain_community.embeddings import HuggingFaceEmbeddings
import os



def main():
    # Load variables from .env file
    load_dotenv()

    # Initialize the LLM.
    # The library will handle the correct API endpoint automatically.
    # Khởi tạo trạng thái ban đầu
    llm = LLM(
        model="gemini-2.5-flash",
        temperature=0.5,
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url=["https://generativelanguage.googleapis.com/v1beta/openai/"]
    )

    # Initialize the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")

    DB_URI = "postgresql://postgres:Quangquan1234%40@localhost:5432/postgres?sslmode=disable"

    with (
        PostgresStore.from_conn_string(
            DB_URI,
        ) as store,
        PostgresSaver.from_conn_string(DB_URI) as checkpointer,
    ):
        
        store.setup()
        checkpointer.setup()
        # Initialize tools
        tools = [get_search_tool(), get_math_tool()]

        # Build the graph once
        graph = build_graph(checkpointer=checkpointer, store=store)

        conversation_id = "1"
        bot_instruct_id = "1"

        config = {
            "configurable": {
                "llm": llm,
                "tools": tools,
                "thread_id": conversation_id,
                "user_id": bot_instruct_id,
            }
        }

        # List to store the history of messages
        # messages = []

        print(f"Chatbot is ready. Conversation ID: {conversation_id}")
        print("Chatbot is ready. Type 'exit' or 'quit' to end the conversation.")
        while True:
            # Get user input from the console
            question = input("You: ")

            # Check if the user wants to exit
            if question.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            # Append the new user message to the history
            input_message = HumanMessage(content=question)

            # Define the initial state for this turn of the conversation
            initial_state: State = {
                "messages": [input_message],
            }

            # Run the graph with the current state
            final_state = graph.invoke(initial_state, config)

            # The graph should return the updated message list (including the AI's response)
            # We update our history for the next turn
            answer = final_state.get("answer", [])

            # Print the final result for this turn
            print(f"Bot: {answer}")


if __name__ == "__main__":
    main()
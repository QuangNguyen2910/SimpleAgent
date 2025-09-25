from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from ..graph.state import State
from typing import Literal
from pydantic import BaseModel, Field

# --- Pydantic Schema for the LLM's decision ---
class MemoryDecision(BaseModel):
    """Quyết định về việc có cần cập nhật bộ nhớ dài hạn hay không."""
    reasoning: str = Field(description="Lý do cho quyết định, tóm tắt những gì cần thay đổi (nếu có).")
    decision: Literal["yes", "no"] = Field(description="Chỉ trả lời 'yes' nếu có thông tin mới, thay đổi, hoặc lỗi thời cần cập nhật, ngược lại là 'no'.")

# --- System Prompt for the Memory Checker ---
MEMORY_CHECKER_PROMPT = """Bạn là một hệ thống kiểm tra bộ nhớ.
Nhiệm vụ của bạn là đọc vài lượt hội thoại gần đây và quyết định xem có thông tin nào mới, đã thay đổi, hoặc đã lỗi thời về người dùng cần được cập nhật vào bộ nhớ dài hạn hay không.

Ví dụ:
- Nếu người dùng nói "Hãy nhớ rằng tôi thích pizza", bạn nên trả lời 'yes'.
- Nếu người dùng nói "Tôi không còn thích pizza nữa, giờ tôi thích burger", bạn nên trả lời 'yes'.
- Nếu người dùng chỉ hỏi một câu hỏi thông thường như "Thời tiết hôm nay thế nào?", bạn nên trả lời 'no'.

Chỉ tập trung vào các sự kiện, sở thích, hoặc thông tin cá nhân của người dùng.

Hãy xem xét đoạn hội thoại gần đây và đưa ra quyết định.
"""

MAX_TURNS_BEFORE_CHECK = 2 # Check memory every 2 turns

def memory_checker(state: State, config: RunnableConfig) -> State:
    """
    NODE: Kiểm tra xem có cần cập nhật bộ nhớ dài hạn không sau một số lượt hội thoại.
    """
    print("--- Thực hiện Node: memory_checker ---")

    # Increment the turn counter. It's part of the state so it persists.
    current_turns = state.get("memory_update_iter", 0) + 1
    state["memory_update_iter"] = current_turns
    
    # If it's not time to check yet, just pass through.
    if current_turns < MAX_TURNS_BEFORE_CHECK:
        print(f"Lượt hội thoại {current_turns}/{MAX_TURNS_BEFORE_CHECK}. Bỏ qua kiểm tra bộ nhớ.")
        return state

    print(f"Đã đạt đến lượt thứ {current_turns}. Bắt đầu kiểm tra bộ nhớ...")
    # Reset the counter for the next cycle
    state["memory_update_iter"] = 0
    
    # Get the last few messages for context
    messages = state["messages"][-6:]
    
    llm = config["configurable"]["llm"]
    structured_llm = llm.with_structured_output(MemoryDecision)
    
    prompt_messages = [SystemMessage(content=MEMORY_CHECKER_PROMPT)] + messages
    
    try:
        response: MemoryDecision = structured_llm.invoke(prompt_messages)
        print(f"Quyết định kiểm tra bộ nhớ: '{response.decision}'. Lý do: {response.reasoning}")

        state["update_memory"] = response.decision

    except Exception as e:
        print(f"LỖI trong memory_checker: {e}")
        state["update_memory"] = "no"  # Fallback to 'no' on error

    return state
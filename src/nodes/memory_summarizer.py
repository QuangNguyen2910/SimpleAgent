from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from ..graph.state import State
from pydantic import BaseModel, Field

# --- Pydantic Schema for the Summary Output ---
class MemorySummary(BaseModel):
    """Một bản tóm tắt ngắn gọn về thông tin cần cập nhật trong bộ nhớ."""
    summary: str = Field(description="Một hoặc hai câu mô tả chính xác thông tin mới hoặc thay đổi về người dùng cần được ghi nhớ. Ví dụ: 'Người dùng bây giờ thích burger thay vì pizza.'")

# --- System Prompt for the Summarizer ---
MEMORY_SUMMARIZER_PROMPT = """Bạn là một hệ thống tóm tắt thông tin.
Nhiệm vụ của bạn là đọc kỹ đoạn hội thoại gần đây và trích xuất chính xác thông tin cá nhân mới hoặc đã thay đổi của người dùng.

Hãy tạo ra một bản tóm tắt ngắn gọn, rõ ràng để một agent khác có thể dựa vào đó cập nhật bộ nhớ.

Chỉ tập trung vào sự thật và yêu cầu cụ thể mà người dùng yêu cầu sau này bạn cần áp dụng để có thể lưu trữ được.
"""

def memory_summarizer(state: State, config: RunnableConfig) -> State:
    """
    NODE: Tóm tắt thông tin cần cập nhật vào bộ nhớ từ cuộc hội thoại.
    """
    print("--- Thực hiện Node: memory_summarizer ---")

    messages = state["messages"][-6:]
    llm = config["configurable"]["llm"]
    structured_llm = llm.with_structured_output(MemorySummary)
    
    prompt_messages = [SystemMessage(content=MEMORY_SUMMARIZER_PROMPT)] + messages
    
    try:
        response: MemorySummary = structured_llm.invoke(prompt_messages)
        summary_text = response.summary
        print(f"Tóm tắt bộ nhớ được tạo: '{summary_text}'")
        
        # Store the summary in the state for the next node
        state["memory_summary"] = summary_text
        
    except Exception as e:
        print(f"LỖI trong memory_summarizer: {e}")
        # If summarization fails, we clear the summary to prevent errors downstream
        state["memory_summary"] = None

    return state
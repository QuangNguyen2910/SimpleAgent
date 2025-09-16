# File: nodes/simple_answerer.py

from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage
from ..graph.state import State
from typing import List

# Merged system prompt for simple chatbot (Không thay đổi)
SYSTEM_PROMPT = """
Bạn là một trợ lý trò chuyện thân thiện, trả lời các câu hỏi của người dùng một cách ngắn gọn, chính xác và hữu ích dựa trên kiến thức chung hoặc ngữ cảnh từ cuộc trò chuyện gần đây. Không sử dụng công cụ bên ngoài.
Hãy trả lời bằng một câu hoặc đoạn ngắn, không cần giải thích trừ khi được yêu cầu. Nếu không có đủ thông tin, trả lời "Tôi không có đủ thông tin để trả lời."
"""

def simple_answerer(state: State) -> State:
    """NODE: Trả lời câu hỏi đơn giản như một chatbot thông thường."""
    print("--- Thực hiện Node: simple_answerer ---")
    
    messages = state["messages"][-4:] if len(state["messages"]) >= 4 else state["messages"]
    
    prompt_messages: List[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT)
    ] + messages
    
    # --- THAY ĐỔI LỚN: Sử dụng `invoke` thay vì `call_straight` ---
    # `invoke` là phương thức tiêu chuẩn của LangChain để gọi chat model.
    # Nó trả về một đối tượng BaseMessage (thường là AIMessage), không phải chuỗi thô.
    response = state["llm"].invoke(prompt_messages)

    
    print(f"Trả lời: {response}")
    
    # Cập nhật state
    state["answer"] = response
    # Thêm toàn bộ đối tượng AIMessage vào lịch sử, đây là cách làm đúng.
    state["messages"].append(HumanMessage(content=response))
    
    return state
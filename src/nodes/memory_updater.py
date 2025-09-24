# File: nodes/simple_answerer.py

from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from ..graph.state import State
from typing import Literal, Dict, Any, List
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
import uuid

# Merged system prompt for simple chatbot (Không thay đổi)
SYSTEM_PROMPT_MEMORY_UPDATER = """
Bạn là một người tốt trong việc trích xuất nội dung, yêu cầu cá nhân của người dùng từ cuộc hội thoại để lưu trữ về sau.

Một số thông tin hiện tại từ người dùng:
{user_info}
"""

# --- Định nghĩa Schema cho Output ---
# Chúng ta tạo một Pydantic model để định nghĩa cấu trúc JSON mà LLM phải trả về.
class NewInstruction(BaseModel):
    """Tóm tắt lại yêu cầu của người dùng qua đoạn hội thoại."""
    new_instruction: str = Field(
        description="Yêu cầu ngắn gọn của người dùng theo hội thoại."
    )

def memory_updater(state: State, config: RunnableConfig, store: BaseStore) -> State:
    """NODE: Trả lời câu hỏi đơn giản như một chatbot thông thường."""
    print("--- Thực hiện Node: memory_updater ---")

    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)
    memories = store.search(namespace, query=str(state["messages"][-1].content), limit=3)
    user_info = "\n".join([d.value["data"] for d in memories])
    print(user_info)

    messages = state["messages"][-6:] if len(state["messages"]) >= 6 else state["messages"]

    # Store new memories if the user asks the model to remember
    last_message = messages[-1]

    if "remember" in last_message.content.lower() or "nhớ" in last_message.content.lower():    
        prompt_messages: List[BaseMessage] = [
            SystemMessage(content=SYSTEM_PROMPT_MEMORY_UPDATER.format(user_info=user_info))
        ] + messages
        
        # --- Sử dụng with_structured_output và invoke ---
        # 1. Tạo một instance LLM mới được "ràng buộc" với schema Decision của chúng ta.
        llm = config["configurable"]["llm"]
        structured_llm = llm.with_structured_output(NewInstruction)
        
        # 2. Gọi LLM. LangChain sẽ tự động xử lý việc ép LLM trả về JSON và parse nó.
        # Không còn `try-except` để parse JSON nữa!
        try:
            response_object = structured_llm.invoke(prompt_messages)
            new_instruction = response_object.new_instruction
        except Exception as e:
            # Xử lý lỗi nếu LLM không thể trả về đúng định dạng sau nhiều lần thử
            print(f"LỖI: LLM không thể tạo output có cấu trúc. Lỗi: {e}. Sử dụng fallback.")
            return state

        print(f"Yêu cầu mới từ người dùng: {new_instruction}")

        store.put(namespace, str(uuid.uuid4()), {"data": new_instruction})  
    
    return state
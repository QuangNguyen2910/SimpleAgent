# File: nodes/selector.py

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Literal

from ..graph.state import State

# --- PROMPTS CHO NODE SELECTOR (Không thay đổi) ---
SELECTOR_SYSTEM_PROMPT = """Bạn là chuyên gia phân loại câu hỏi của người dùng để quyết định cách xử lý.

Dựa trên câu hỏi của người dùng, hãy quyết định xem nên trả lời theo cách 'normal' (sử dụng kiến thức nội bộ, không cần công cụ) hay 'deep_research' (cần dùng công cụ để tìm kiếm, tính toán, v.v.).

**HƯỚNG DẪN:**
- "normal": Nếu câu hỏi là lời chào, hỏi đáp thông thường, hoặc có thể trả lời trực tiếp mà không cần tra cứu.
- "deep_research": Nếu câu hỏi yêu cầu tìm kiếm web, tính toán phức tạp, hoặc bất kỳ hành động nào cần công cụ bên ngoài.

Ví dụ:
- Câu hỏi: "Chào bạn, bạn khỏe không?" → "normal"
- Câu hỏi: "Thủ đô của Pháp là gì?" → "normal" (kiến thức phổ thông)
- Câu hỏi: "Giá cổ phiếu Tesla hôm nay là bao nhiêu?" → "deep_research"
- Câu hỏi: "Tính căn bậc hai của 529." → "deep_research"

Trả về quyết định của bạn.
"""

# --- THAY ĐỔI LỚN 1: Định nghĩa Schema cho Output ---
# Chúng ta tạo một Pydantic model để định nghĩa cấu trúc JSON mà LLM phải trả về.
class Decision(BaseModel):
    """Một quyết định về cách xử lý câu hỏi của người dùng."""
    decision: Literal["normal", "deep_research"] = Field(
        description="Lựa chọn phải là 'normal' hoặc 'deep_research'."
    )

def select_node(state: State) -> State:
    """NODE SELECTOR: Quyết định cách xử lý dựa trên câu hỏi."""
    print("--- Thực hiện Node: select_node ---")
    
    question = ""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            question = message.content
            break
    
    if not question:
        print("LỖI: Không tìm thấy HumanMessage trong state['messages']")
        state["decision"] = "normal"  # Fallback
        return state
    
    messages = [
        SystemMessage(content=SELECTOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Câu hỏi: {question}")
    ]
    
    # --- THAY ĐỔI LỚN 2: Sử dụng with_structured_output và invoke ---
    # 1. Tạo một instance LLM mới được "ràng buộc" với schema Decision của chúng ta.
    structured_llm = state["llm"].with_structured_output(Decision)
    
    # 2. Gọi LLM. LangChain sẽ tự động xử lý việc ép LLM trả về JSON và parse nó.
    # Không còn `try-except` để parse JSON nữa!
    try:
        response_object = structured_llm.invoke(messages)
        decision = response_object.decision
    except Exception as e:
        # Xử lý lỗi nếu LLM không thể trả về đúng định dạng sau nhiều lần thử
        print(f"LỖI: LLM không thể tạo output có cấu trúc. Lỗi: {e}. Sử dụng fallback.")
        decision = "normal" # Fallback an toàn

    print(f"Quyết định: {decision}")
    
    # Cập nhật state (đổi 'decision' thành 'decision' để phù hợp với builder.py)
    state["decision"] = decision
    
    return state
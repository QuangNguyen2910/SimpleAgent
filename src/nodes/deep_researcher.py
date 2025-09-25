# Tệp: nodes/deep_researcher.py

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage 
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union
from langchain_core.runnables import RunnableConfig
from ..graph.state import State
from langgraph.store.base import BaseStore
import uuid

# --- 1. Định nghĩa các Lược đồ Hành động (bắt chước bind_tools) ---

class ToolCall(BaseModel):
    """Một yêu cầu có cấu trúc để gọi một công cụ cụ thể với các đối số."""
    name: str = Field(..., description="Tên của công cụ cần gọi (ví dụ: 'search_web', 'evaluate_expression').")
    arguments: Dict[str, Any] = Field(..., description="Một từ điển các đối số cho công cụ.")

class FinalAnswer(BaseModel):
    """Câu trả lời cuối cùng sẽ được trả về cho người dùng."""
    answer: str = Field(..., description="Câu trả lời cuối cùng toàn diện cho câu hỏi của người dùng.")

# --- 2. Định nghĩa Lược đồ ReAct Chính ---
# Đây là điểm mấu chốt. Nó buộc LLM phải cung cấp cả lý luận và một hành động có cấu trúc.

class ReActStep(BaseModel):
    """
    Đầu ra của mô hình cho một bước duy nhất, chứa lý luận và hành động đã chọn.
    Hành động phải là một danh sách các lệnh gọi công cụ hoặc là câu trả lời cuối cùng.
    """
    reasoning: str = Field(..., description="Một quá trình suy nghĩ chi tiết, từng bước. Giải thích kế hoạch của bạn, những gì bạn biết và những gì bạn cần tìm ra.")
    action: Union[List[ToolCall], FinalAnswer] = Field(..., description="Hành động cần thực hiện. Đây sẽ là MỘT trong hai: một danh sách các lệnh gọi công cụ HOẶC câu trả lời cuối cùng.")


# --- 3. Prompt Hệ thống để hướng dẫn LLM ---

REACT_HYBRID_PROMPT = """Bạn là một trợ lý nghiên cứu rất thông minh. Mục tiêu của bạn là trả lời câu hỏi của người dùng bằng cách chia nó thành một loạt các bước.

Bạn hoạt động trong một vòng lặp:
1.  **Lý luận:** Đầu tiên, bạn suy nghĩ. Phân tích yêu cầu, thông tin có sẵn và kế hoạch của bạn.
2.  **Hành động:** Dựa trên lý luận của bạn, bạn chọn MỘT trong hai hành động có thể:
    a. **Gọi Công cụ:** Nếu bạn cần thêm thông tin, hãy cung cấp một danh sách một hoặc nhiều đối tượng `ToolCall`.
    b. **Câu trả lời cuối cùng:** Nếu bạn có tất cả thông tin cần thiết, hãy cung cấp `FinalAnswer`.

**Các công cụ có sẵn:**
- `search_web(query: str)`: Sử dụng công cụ này để tìm kiếm thông tin trên internet.
- `evaluate_expression(expression: str)`: Sử dụng công cụ này cho các phép tính toán học.

**Đầu ra của bạn PHẢI LUÔN LUÔN là một đối tượng JSON duy nhất hợp lệ với lược đồ được yêu cầu.**

Một số thông tin từ người dùng:
{user_info}

Bắt đầu.
"""

# --- Các Node và Cạnh của LangGraph ---

def call_agent_and_parse(state: State, config: RunnableConfig, store: BaseStore) -> State:
    """Node gọi LLM, được cấu trúc để xuất ra một đối tượng ReActStep."""
    print("--- Thực hiện Node: call_agent_and_parse (Hybrid) ---")

    # ... (phần lấy user_info không thay đổi) ...
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")
    memories = store.search(namespace, query=str(state["messages"][-1].content))
    user_info = "\n".join([d.value["data"] for d in memories])

    llm = config["configurable"]["llm"]
    llm_with_structure = llm.with_structured_output(ReActStep)
    
    messages = state["messages"][-6:] if len(state["messages"]) >= 6 else state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages.insert(0, SystemMessage(content=REACT_HYBRID_PROMPT.format(user_info=user_info)))

    response: ReActStep = llm_with_structure.invoke(messages)
    
    print(f"\n🤔 LÝ LUẬN: {response.reasoning}")
    
    # Kiểm tra loại hành động và cập nhật trạng thái
    if isinstance(response.action, FinalAnswer):
        print(f"✅ HÀNH ĐỘỘNG: Câu trả lời cuối cùng")
        # Thêm AIMessage chứa lý luận và câu trả lời cuối cùng
        final_ai_message = AIMessage(content=f"Lý luận:\n{response.reasoning}\n\nCâu trả lời cuối cùng:\n{response.action.answer}")
        state["messages"].append(final_ai_message)
        state["answer"] = response.action.answer
        state["parsed_action"] = None # Tín hiệu để kết thúc vòng lặp
    else: # Đây là một danh sách các đối tượng ToolCall
        print(f"🛠️ HÀNH ĐỘNG: Gọi Công cụ")
        # *** THAY ĐỔI CHÍNH Ở ĐÂY ***
        # Tạo tool_calls với ID duy nhất cho mỗi lệnh gọi
        tool_calls = []
        for tool_call_action in response.action:
            tool_calls.append({
                "name": tool_call_action.name,
                "args": tool_call_action.arguments,
                "id": str(uuid.uuid4()) # Tạo ID duy nhất
            })
            print(f"- Công cụ: {tool_call_action.name}, Đối số: {tool_call_action.arguments}, ID: {tool_calls[-1]['id']}")

        # Tạo một AIMessage chứa cả lý luận và các lệnh gọi công cụ có cấu trúc
        ai_message_with_tools = AIMessage(
            content=response.reasoning, # Chỉ chứa lý luận ở đây
            tool_calls=tool_calls
        )
        state["messages"].append(ai_message_with_tools)
        state["parsed_action"] = tool_calls # Chuyển các lệnh gọi công cụ có cấu trúc (với ID)

    return state


def execute_tool(state: State, config: RunnableConfig) -> State:
    """Node thực thi các lệnh gọi công cụ và trả về các ToolMessage."""
    print("--- Thực hiện Node: execute_tool (Hybrid) ---")
    
    tool_calls: List[Dict] = state["parsed_action"]
    known_tools = {tool.name: tool for tool in config["configurable"]["research_tools"]} if config["configurable"]["research_tools"] else {}
    
    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        
        tool_function = known_tools.get(tool_name)
        if not tool_function:
            observation = f"Lỗi: Không tìm thấy công cụ '{tool_name}'."
        else:
            try:
                observation = tool_function.invoke(tool_args)
            except Exception as e:
                observation = f"Lỗi khi thực thi công cụ {tool_name}: {e}"
        
        print(f"👁️ QUAN SÁT (từ {tool_name}):\n{observation}")
        
        # *** KEY FIX IS HERE ***
        # Add the 'name' parameter to the ToolMessage constructor.
        tool_messages.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call_id,
            name=tool_name  # <-- This is the required addition for Gemini
        ))

    state["messages"].extend(tool_messages)
    state["parsed_action"] = None 
    return state
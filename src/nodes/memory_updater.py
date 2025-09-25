import uuid
from typing import List, Dict
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from ..graph.state import State
from langgraph.store.base import BaseStore

# --- 1. NEW, more focused prompt ---
MEMORY_UPDATER_PROMPT = """Bạn là một agent quản lý bộ nhớ.
Bạn đã nhận được một nhiệm vụ về việc cập nhật bộ nhớ.

Kế hoạch của bạn là thực thi nhiệm vụ này một cách chính xác:
-  Nếu có thông tin cũ tương hỗ với thông tin mới, sử dụng `delete_memory` với ID bạn tìm được để xóa thông tin cũ.
-  Sử dụng `write_memory` để lưu thông tin mới, chính xác như trong nhiệm vụ.

**Các công cụ có sẵn:**
- `write_memory(content: str)`
- `delete_memory(memory_id: str)`

Một số thông tin/yêu cầu trước đây từ người dùng:
{user_info}

Hãy bắt đầu.
"""

# --- 2. The Agent Logic (Updated to use the summary) ---
def memory_updater(state: State, config: RunnableConfig, store: BaseStore) -> State:
    """Node gọi LLM để thực hiện các tác vụ quản lý bộ nhớ DỰA TRÊN TÓM TẮT."""
    print("--- Thực hiện Node: memory_updater ---")

    query = state.get("memory_summary")
    if not query:
        print("Cảnh báo: Không có tóm tắt bộ nhớ. Bỏ qua cập nhật.")
        state["parsed_action"] = None # Signal completion
        return state

    user_id = config["configurable"]["user_id"]
    memory_tools = config["configurable"]["memory_tools"]
    known_tools = {tool.name: tool for tool in memory_tools} if memory_tools else {}
    namespace = (user_id, "memories")
    query = state["memory_summary"]
    memories = store.search(namespace, query=query, limit=5)
    user_info = "\n".join([f"ID: {d.key}, Nội dung: {d.value['data']}" for d in memories])

    llm = config["configurable"]["llm"]
    llm_with_tools = llm.bind_tools(memory_tools)
    
    # The context is now just the prompt with the summary, not the whole message history
    prompt = MEMORY_UPDATER_PROMPT.format(user_info=user_info)
    messages = [SystemMessage(content=prompt)] + [HumanMessage(content=f"Thông tin cần lưu vào bộ nhớ: {query}")]
    
    response = llm_with_tools.invoke(messages)
    print(f"\nPhản hồi từ memory_updater: {response}")

    if not response.tool_calls:
        print("Cảnh báo: memory_updater không yêu cầu gọi công cụ nào.")
        return state

    tool_calls: List[Dict] = response.tool_calls
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_object = known_tools.get(tool_name)
        if not tool_object:
            print(f"Lỗi: Không tìm thấy công cụ '{tool_name}'.")
        else:
            try:
                # Get the actual Python function from the tool object
                raw_function = tool_object.func
                
                # Call the function directly, providing the extra args it needs
                result = raw_function(
                    **tool_args,      # Unpacks {'memory_id': '1'}
                    user_id=user_id,  # Add the user_id
                    store=store       # Add the store object
                )
                print(f"Thực thi thành công công cụ {tool_name}: {result}")
            except Exception as e:
                print(f"Lỗi khi thực thi công cụ {tool_name}: {e}")
    
    return state
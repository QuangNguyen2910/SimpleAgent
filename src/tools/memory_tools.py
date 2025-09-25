import uuid
from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.config import get_store

# --- Tool Input Schemas ---

class ReadMemoryInput(BaseModel):
    query: str = Field(description="Chủ đề hoặc câu hỏi cần tìm kiếm trong bộ nhớ.")

class WriteMemoryInput(BaseModel):
    content: str = Field(description="Nội dung thông tin cần ghi nhớ.")

class DeleteMemoryInput(BaseModel):
    memory_id: str = Field(description="ID duy nhất của mẩu tin cần xóa.")

# --- Tool Functions ---

# def read_memory(query: str, config: RunnableConfig, store: BaseStore) -> str:
#     """
#     Dùng để đọc lại những điều người dùng đã yêu cầu bạn ghi nhớ trước đây.
#     Công cụ này tìm kiếm trong bộ nhớ dài hạn của bạn để tìm thông tin liên quan đến một chủ đề.
#     Nó sẽ trả về cả nội dung và ID của mẩu tin.
#     """
#     print(f"--- Đang thực thi Công cụ Đọc Bộ nhớ với truy vấn: '{query}' ---")
#     store = get_store(config)
#     user_id = config["configurable"]["user_id"]
#     namespace = (user_id, "memories")

#     # Search for relevant memories
#     try:
#         memories = store.search(namespace, query=query, limit=3)
#     except Exception as e:
#         return f"Lỗi khi tìm kiếm trong bộ nhớ: {e}"

#     if not memories:
#         return "Không tìm thấy thông tin nào liên quan trong bộ nhớ."

#     # Format the results to be readable by the LLM, including the ID for deletion
#     formatted_results = "\n\n".join(
#         [f"ID: {doc.key}\nNội dung: {doc.value.get('data', 'N/A')}" for doc in memories]
#     )
#     return f"Thông tin tìm thấy trong bộ nhớ:\n{formatted_results}"

def write_memory(content: str, user_id: str, store: BaseStore) -> str:
    """
    Dùng để ghi nhớ một thông tin mới mà người dùng cung cấp.
    """
    print(f"--- Đang thực thi Công cụ Ghi Bộ nhớ với nội dung: '{content}' ---")
    namespace = (user_id, "memories")
    memory_id = str(uuid.uuid4())

    try:
        store.put(namespace, memory_id, {"data": content})
        return f"Đã ghi nhớ thành công thông tin: '{content}'"
    except Exception as e:
        return f"Lỗi khi ghi vào bộ nhớ: {e}"

def delete_memory(memory_id: str, user_id: str, store: BaseStore) -> str:
    """
    Dùng để xóa một mẩu tin cụ thể khỏi bộ nhớ khi nó đã cũ hoặc sai.
    """
    print(f"--- Đang thực thi Công cụ Xóa Bộ nhớ với ID: '{memory_id}' ---")
    namespace = (user_id, "memories")
    
    try:
        store.delete(namespace, [memory_id])
        return f"Đã xóa thành công mẩu tin với ID: {memory_id}"
    except Exception as e:
        return f"Lỗi khi xóa khỏi bộ nhớ: {e}"


# --- Function to get all memory tools ---

def get_memory_tools() -> List[StructuredTool]:
    """Trả về một danh sách các StructuredTool để quản lý bộ nhớ."""
    return [
        # StructuredTool.from_function(
        #     name="read_memory",
        #     func=read_memory,
        #     description="Tìm kiếm và đọc thông tin từ bộ nhớ dài hạn.",
        #     args_schema=ReadMemoryInput
        # ),
        StructuredTool.from_function(
            name="write_memory",
            func=write_memory,
            description="Lưu một thông tin mới vào bộ nhớ dài hạn.",
            args_schema=WriteMemoryInput
        ),
        StructuredTool.from_function(
            name="delete_memory",
            func=delete_memory,
            description="Xóa một thông tin đã cũ hoặc sai khỏi bộ nhớ dài hạn bằng ID của nó.",
            args_schema=DeleteMemoryInput
        ),
    ]
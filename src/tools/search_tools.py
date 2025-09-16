# File: tools/search_tools.py

import os
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tavily import TavilyClient
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# --- 1. Khởi tạo Tavily Client ---
# Code sẽ tự động đọc API key từ biến môi trường TAVILY_API_KEY
try:
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
except KeyError:
    raise Exception("Lỗi: Biến môi trường TAVILY_API_KEY chưa được thiết lập. Vui lòng thêm API key của bạn.")


# --- 2. Cập nhật Mô tả và Schema cho công cụ ---

# Mô tả này ngắn gọn và rõ ràng hơn cho LLM.
_TOOL_DESCRIPTION = (
    "Công cụ tìm kiếm trên internet để lấy thông tin về các sự kiện, con người, địa điểm, hoặc bất kỳ kiến thức thực tế nào. "
    "Không sử dụng cho các phép tính toán học."
)

class SearchInput(BaseModel):
    query: str = Field(description="Một câu truy vấn tìm kiếm rõ ràng và cụ thể.")


# --- 3. Viết lại hàm search_web để gọi API của Tavily ---

def search_web(query: str) -> str:
    """
    Tìm kiếm thông tin trên web bằng Tavily API dựa trên một truy vấn.
    """
    print(f"--- Đang thực thi Công cụ Tìm kiếm Tavily với truy vấn: '{query}' ---")
    
    try:
        # Gọi API của Tavily. Bạn có thể tùy chỉnh các tham số khác như max_results.
        search_results = tavily_client.search(
            query=query, 
            search_depth="basic", # "basic" cho tốc độ, "advanced" cho chi tiết
            max_results=5
        )
    except Exception as e:
        return f"Đã xảy ra lỗi khi gọi Tavily API: {e}"

    # Tavily trả về kết quả trong key 'results'.
    results = search_results.get("results", [])
    
    if not results:
        return "Không tìm thấy kết quả nào từ Tavily."
        
    # Định dạng kết quả thành một chuỗi duy nhất, súc tích để LLM xử lý.
    # Tavily thường trả về 'content' là bản tóm tắt rất tốt.
    formatted_results = "\n\n".join(
        [f"Nguồn: {res.get('url', 'N/A')}\nNội dung: {res.get('content', '')}" for res in results]
    )
    return formatted_results


# --- 4. Hàm get_search_tool không thay đổi ---

def get_search_tool():
    """Trả về một StructuredTool để tìm kiếm web bằng Tavily."""
    return StructuredTool.from_function(
        name="search_web",
        func=search_web,
        description=_TOOL_DESCRIPTION,
        args_schema=SearchInput
    )
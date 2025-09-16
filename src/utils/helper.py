import re
from typing import List

def convert_to_markdown_table(data: dict) -> str:
    """
    Converts a dictionary containing a list-of-lists table into a Markdown table string.

    The function assumes the input dictionary has a key "table" which holds a list of lists.
    The first inner list is treated as the table header.

    Args:
        data (dict): A dictionary with a key "table".
                     e.g., {"table": [["h1", "h2"], ["r1c1", "r1c2"]]}

    Returns:
        str: A string containing the formatted Markdown table. Returns an empty string
             if the input is invalid or the table is empty.
    """
    # 1. Validate input and extract the table data
    if not isinstance(data, dict) or "table" not in data or not data["table"]:
        return ""

    table_data = data["table"]
    # 2. Separate header from the rest of the rows
    header = table_data[0]
    rows = table_data[1:]
    num_columns = len(header)


    
    # 3. Build the Markdown header line
    # Example: | Header 1 | Header 2 |
    header_line = "| " + " | ".join(str(cell) for cell in header) + " |"

    # 4. Build the Markdown separator line
    # Example: |----------|----------|
    separator_line = "| " + " | ".join(["---"] * num_columns) + " |"

    # 5. Build the data row lines
    row_lines = []
    for row in rows:
        # Ensure all cells are strings for the join operation
        row_str = [str(cell) for cell in row]
        row_lines.append("| " + " | ".join(row_str) + " |")

    # 6. Combine all parts into the final table string
    markdown_table = [header_line, separator_line] + row_lines

    return "\n".join(markdown_table)

def _get_column_values(row_identifier: str, table_data: List[List]) -> List[float]:
    """Helper function to extract numeric values from a specified row in the table."""
    if not table_data:
        raise ValueError("Table data not found. Cannot perform table operations.")

    values_str = []
    for row in table_data[1:]:
        if row and row[0].strip().lower() == row_identifier.strip().lower():
            values_str.extend(row[1:])
            break
    
    if not values_str:
         raise ValueError(f"Metric row identifier '{row_identifier}' not found in the table.")

    values = []
    number_pattern = re.compile(r'-?[\d,]+(?:\.\d+)?(?:\s*[%$KMBT]|\s*[a-zA-Z]+)?')
    for item in values_str:
        if item is None or str(item).strip() in ["NA", "-", ""]:
            continue
        try:
            s_item = str(item).strip()
            match = number_pattern.search(s_item)
            if match:
                number_str = match.group(0)
                cleaned_str = re.sub(r'[^\d.-]', '', number_str)
                value = float(cleaned_str)
                values.append(value)
        except (ValueError, TypeError):
            continue

    if not values:
        raise ValueError(f"No numeric data found for metric row '{row_identifier}'.")

    return values

def execute_program(program, table_data=None):
    """
    Execute a program string in the exe_ans format, including table functions.

    Args:
        program (str): A program string with operations separated by commas
                      e.g., "subtract(248.36, const_100), divide(#0, const_100)"
                      or "table_max(P/E (x)), add(#0, const_10)"
        table_data (List[List], optional): Table data for table operations.
                                         First row is header, subsequent rows are data.

    Returns:
        str: The final result as a string with 4 decimal places
    """

    results = []

    if not program or not program.strip():
        raise ValueError("Empty program")

    program = program.replace('\n', ' ').replace('\r', ' ')
    operations = []
    current_op = ""
    paren_count = 0

    for char in program:
        if char == '(':
            paren_count += 1
            current_op += char
        elif char == ')':
            paren_count -= 1
            current_op += char
        elif char == ',' and paren_count == 0:
            if current_op.strip():
                operations.append(current_op.strip())
            current_op = ""
        else:
            current_op += char

    if current_op.strip():
        operations.append(current_op.strip())

    if not operations:
        raise ValueError("No valid operations found")

    for operation in operations:
        if '(' in operation and ')' in operation:
            func_name = operation.split('(')[0].strip()
            args_str = operation[operation.index('(')+1:operation.rindex(')')].strip()

            if not args_str:
                args = []
            else:
                if func_name in ['table_max', 'table_min', 'table_sum', 'table_average']:
                    # Split args but preserve row identifier with spaces/parentheses
                    args = [arg.strip() for arg in args_str.split(',') if arg.strip()]
                    # Filter out 'none' (case-insensitive)
                    args = [arg for arg in args if arg.lower() != 'none']
                else:
                    args = [arg.strip() for arg in args_str.split(',') if arg.strip()]

            if func_name in ['table_max', 'table_min', 'table_sum', 'table_average']:
                if not table_data:
                    raise ValueError(f"Table data required for {func_name} function")
                
                if len(args) < 1:
                    raise ValueError(f"{func_name} requires at least one argument (row_identifier)")
                
                row_identifier = args[0].strip('"\'')
                values = _get_column_values(row_identifier, table_data)
                
                if func_name == 'table_max':
                    result = max(values)
                elif func_name == 'table_min':
                    result = min(values)
                elif func_name == 'table_sum':
                    result = sum(values)
                elif func_name == 'table_average':
                    result = sum(values) / len(values) if values else 0.0
                
                results.append(result)
                
            else:
                parsed_args = []
                number_pattern = re.compile(r'-?[\d,]+(?:\.\d+)?(?:\s*[%$KMBT]|\s*[a-zA-Z]+)?')
                for arg in args:
                    if arg.startswith('#'):
                        index = int(arg[1:])
                        if index >= len(results):
                            raise ValueError(f"Invalid reference #{index}: only {len(results)} results available")
                        parsed_args.append(results[index])
                    elif arg.startswith('const_'):
                        tm_value = float(arg.split("_")[1])
                        parsed_args.append(tm_value)
                    else:
                        try:
                            match = number_pattern.search(arg)
                            if match:
                                number_str = match.group(0)
                                cleaned_str = re.sub(r'[^\d.-]', '', number_str)
                                value = float(cleaned_str)
                                # Convert percentage to decimal if % is present
                                if '%' in number_str:
                                    value = value / 100
                                parsed_args.append(value)
                            else:
                                raise ValueError(f"Invalid number: {arg}")
                        except (ValueError, TypeError):
                            raise ValueError(f"Invalid number: {arg}")

                if func_name == 'add':
                    if len(parsed_args) != 2:
                        raise ValueError(f"add requires exactly 2 arguments, got {len(parsed_args)}")
                    result = parsed_args[0] + parsed_args[1]
                elif func_name == 'subtract':
                    if len(parsed_args) != 2:
                        raise ValueError(f"subtract requires exactly 2 arguments, got {len(parsed_args)}")
                    result = parsed_args[0] - parsed_args[1]
                elif func_name == 'multiply':
                    if len(parsed_args) != 2:
                        raise ValueError(f"multiply requires exactly 2 arguments, got {len(parsed_args)}")
                    result = parsed_args[0] * parsed_args[1]
                elif func_name == 'divide':
                    if len(parsed_args) != 2:
                        raise ValueError(f"divide requires exactly 2 arguments, got {len(parsed_args)}")
                    if parsed_args[1] == 0:
                        raise ValueError("Division by zero")
                    result = parsed_args[0] / parsed_args[1]
                else:
                    raise ValueError(f"Unknown function: {func_name}")

                results.append(result)
        else:
            raise ValueError(f"Invalid operation format: {operation}")

    if not results:
        raise ValueError("No results generated")

    return f"{results[-1]:.4f}"

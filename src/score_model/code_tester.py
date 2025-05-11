import re
import threading
import time
from typing import Dict, Any, List, Tuple, Optional, Union

def extract_all_solutions(code: str) -> List[Tuple[str, str]]:
    """
    Extract all solutions and their corresponding function names from the code.
    Returns a list of tuples (function_name, solution_code)
    """
    
    solutions = []
    # Split the code by task prompts
    tasks = re.split(r'You are an expert Python programmer, and here is your task:', code)
    
    # Skip the first element if it's empty
    tasks = [task for task in tasks if task.strip()]

    for task in tasks:
        if "Please provide your solution:" in task and "Corrected Solution: " not in task:
            parts = task.split("Please provide your solution:")
            if len(parts) < 2:
                continue
                
            solution_part = parts[1].strip()
            
            # Clean up triple quotes at the beginning
            solution_part = re.sub(r'^[\'\"][\'\"][\'\"]', '', solution_part)
            solution_part = re.sub(r'^```python', '', solution_part)
            solution_part = re.sub(r'^```', '', solution_part)
            
            # Clean up triple quotes at the end
            solution_part = re.sub(r'[\'\"][\'\"][\'\"]$', '', solution_part)
            solution_part = re.sub(r'```$', '', solution_part)
            
            # Find function definitions in the cleaned solution
            func_matches = re.finditer(r'def\s+(\w+)\s*\(([^)]*)\):', solution_part)
            
            for func_match in func_matches:
                func_name = func_match.group(1)
                func_start = func_match.start()
                
                # Find the function body by tracking indentation
                lines = solution_part[func_start:].split('\n')
                func_lines = [lines[0]]  # Start with the definition line
                
                # Process the remaining lines to extract the function body
                for i in range(1, len(lines)):
                    line = lines[i]
                    # If line is empty or indented, it's part of the function
                    if not line.strip() or line.startswith(' ') or line.startswith('\t'):
                        func_lines.append(line)
                    else:
                        # Found a non-indented line, which means end of function
                        # But check if it's a decorator or another part of the function
                        if line.strip().startswith('@') or line.strip().startswith('return '):
                            func_lines.append(line)
                        else:
                            break
                
                # Join the function lines
                clean_func = '\n'.join(func_lines).strip()
                
                # Remove any print statements or test calls at the end
                clean_func_lines = clean_func.split('\n')
                final_lines = []
                for line in clean_func_lines:
                    # Skip lines with print statements or assertion calls
                    if 'print(' not in line and 'assert ' not in line:
                        final_lines.append(line)
                
                final_func = '\n'.join(final_lines).strip()
                solutions.append((func_name, final_func))
                
                # If we found at least one function, we can stop looking
                break
        elif "Corrected Solution: " in task:  
            parts = task.split("Corrected Solution: ")
            if len(parts) < 2:
                continue
                
            solution_part = parts[1].strip()
            
            # Clean up triple quotes at the beginning
            solution_part = re.sub(r'^[\'\"][\'\"][\'\"]', '', solution_part)
            solution_part = re.sub(r'^```python', '', solution_part)
            solution_part = re.sub(r'^```', '', solution_part)
            
            # Clean up triple quotes at the end
            solution_part = re.sub(r'[\'\"][\'\"][\'\"]$', '', solution_part)
            solution_part = re.sub(r'```$', '', solution_part)
            
            # Find function definitions in the cleaned solution
            func_matches = re.finditer(r'def\s+(\w+)\s*\(([^)]*)\):', solution_part)
            
            for func_match in func_matches:
                func_name = func_match.group(1)
                func_start = func_match.start()
                
                # Find the function body by tracking indentation
                lines = solution_part[func_start:].split('\n')
                func_lines = [lines[0]]  # Start with the definition line
                
                # Process the remaining lines to extract the function body
                for i in range(1, len(lines)):
                    line = lines[i]
                    # If line is empty or indented, it's part of the function
                    if not line.strip() or line.startswith(' ') or line.startswith('\t'):
                        func_lines.append(line)
                    else:
                        # Found a non-indented line, which means end of function
                        # But check if it's a decorator or another part of the function
                        if line.strip().startswith('@') or line.strip().startswith('return '):
                            func_lines.append(line)
                        else:
                            break
                
                # Join the function lines
                clean_func = '\n'.join(func_lines).strip()
                
                # Remove any print statements or test calls at the end
                clean_func_lines = clean_func.split('\n')
                final_lines = []
                for line in clean_func_lines:
                    # Skip lines with print statements or assertion calls
                    if 'print(' not in line and 'assert ' not in line:
                        final_lines.append(line)
                
                final_func = '\n'.join(final_lines).strip()
                solutions.append((func_name, final_func))
                
                # If we found at least one function, we can stop looking
                break
    return solutions

def get_function_name_from_test(test: str) -> Optional[str]:
    """
    Extract the function name from a test assertion.
    """
    # Look for function calls in the test
    match = re.search(r'assert\s+(\w+)\(', test)
    if match:
        return match.group(1)
    return None

def safe_execute_code(extracted_codes_list, tests, timeout: int = 5):
    """
    Execute extracted code against tests and return performance metrics.
    
    Args:
        extracted_codes_list: List of extracted code solutions in the format from extract_all_solutions
        tests: List of test assertions or list of lists of test assertions
        timeout: Maximum execution time in seconds
        
    Returns:
        Dict containing success status, pass rate, and detailed results
    """
    import threading
    import re
    if not extracted_codes_list or all(not code for code in extracted_codes_list):
        return {"success": False, "error": "No solutions found in the code", "pass_rate": 0.0}
        
    # Flatten the extracted codes into a single list of (function_name, code) tuples
    all_solutions = []
    for solutions in extracted_codes_list:
        if solutions:  # Check if solutions is not empty
            for solution in solutions:
                all_solutions.append(solution)
    
    if not all_solutions:
        return {"success": False, "error": "No valid solutions found", "pass_rate": 0.0, "rewards": 0.0}
        
    # Handle tests that are in a list of lists format (e.g., [['assert1', 'assert2'], ['assert3']])
    flat_tests = []
    if tests and isinstance(tests, list):
        for test_item in tests:
            if isinstance(test_item, list):
                flat_tests.extend(test_item)
            else:
                flat_tests.append(test_item)
    else:
        flat_tests = tests
        
    # Use the flattened tests list going forward
    tests = flat_tests
    
    # Extract function names from tests
    function_names_in_tests = []
    for test in tests:
        func_name = get_function_name_from_test(test)
        if func_name:
            function_names_in_tests.append(func_name)
    
    if not function_names_in_tests:
        return {"success": False, "error": "Could not parse function names from tests", "pass_rate": 0.0}
    
    # Group tests by the function they're testing
    tests_by_function = {}
    for test in tests:
        func_name = get_function_name_from_test(test)
        if func_name:
            if func_name not in tests_by_function:
                tests_by_function[func_name] = []
            tests_by_function[func_name].append(test)
    
    # Run each solution against its corresponding tests
    results = {"success": True, "results": [], "total_tests": 0, "passed_tests": 0}
    
    for func_name, tests_list in tests_by_function.items():
        # Find the corresponding solution for this function
        solution_code = None
        for sol_name, sol_code in all_solutions:
            if sol_name == func_name:
                solution_code = sol_code
                break
        
        # Skip if we don't have code for this function
        if not solution_code:
#             print(f"No solution found for function: {func_name}")
            continue
        
        function_tests = tests_list
        results["total_tests"] += len(function_tests)
        
        exec_globals = {'__builtins__': __builtins__}
        test_result = {
            "function": func_name,
            "passed_tests": 0,
            "total_tests": len(function_tests),
            "errors": []
        }
        
        def run_tests():
            nonlocal test_result
            try:
                # Add necessary imports for regex
                exec("import re", exec_globals)
                
                # Execute the solution code to define the function
                exec(solution_code, exec_globals)
                
                # Run each test for this function
                for test in function_tests:
                    try:
                        exec(test, exec_globals)
                        test_result["passed_tests"] += 1
                        results["passed_tests"] += 1
                    except Exception as e:
                        test_result["errors"].append(f"Test failed: {test} - Error: {str(e)}")
                
            except Exception as e:
                test_result["errors"].append(f"Function execution error: {str(e)}")
        
        # Run tests in a thread with timeout
        thread = threading.Thread(target=run_tests)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            test_result["errors"].append("Code execution timed out")
        
        # Add results
        results["results"].append(test_result)
    
    # Calculate pass rate
    if results["total_tests"] > 0:
        results["pass_rate"] = results["passed_tests"] / results["total_tests"]
    else:
        results["pass_rate"] = 0.0
    
    # Update overall success
    results["success"] = results["passed_tests"] > 0 and results["passed_tests"] == results["total_tests"]
    
    # Add rewards field to be compatible with existing code
    results["rewards"] = results["pass_rate"]
    
#     print(results["pass_rate"])
    
    return results
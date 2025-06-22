#!/usr/bin/env python3
"""
C Function Parser for extracting individual functions from C code
"""
import re
from typing import List, Dict, Optional, Tuple

class CFunctionParser:
    """Parser to extract individual C functions from full C code"""
    
    def __init__(self):
        # Pattern to match function definitions
        # Matches: return_type function_name(parameters) { ... }
        self.function_pattern = re.compile(
            r'(?:^|\n)\s*(?:static\s+|inline\s+|extern\s+)*'  # Optional modifiers
            r'(?:void|int|char|float|double|long|short|unsigned|signed|struct\s+\w+|\w+\s*\*?)\s*'  # Return type
            r'(\w+)\s*'  # Function name (captured)
            r'\([^)]*\)\s*'  # Parameters
            r'\{',  # Opening brace
            re.MULTILINE | re.DOTALL
        )
        
        # Pattern for main function specifically
        self.main_pattern = re.compile(
            r'(?:^|\n)\s*(?:int\s+)?main\s*\([^)]*\)\s*\{',
            re.MULTILINE | re.DOTALL
        )
    
    def find_function_boundaries(self, code: str, start_pos: int) -> Optional[Tuple[int, int]]:
        """
        Find the start and end positions of a function given its opening brace position
        
        Args:
            code: The full C code
            start_pos: Position of the opening brace
            
        Returns:
            Tuple of (start_line_pos, end_pos) or None if not found
        """
        # Find the actual start of the function (before the opening brace)
        lines = code[:start_pos].split('\n')
        function_start_line = len(lines) - 1
        
        # Look backwards to find the function signature start
        while function_start_line > 0:
            line = lines[function_start_line].strip()
            if line and not line.startswith('//') and not line.startswith('/*'):
                # Check if this line contains function signature elements
                if any(keyword in line for keyword in ['int', 'void', 'char', 'float', 'double', 'struct']) or '(' in line:
                    break
            function_start_line -= 1
        
        # Calculate actual start position
        actual_start = sum(len(line) + 1 for line in lines[:function_start_line])
        
        # Find matching closing brace
        brace_count = 0
        i = start_pos
        
        while i < len(code):
            char = code[i]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return (actual_start, i + 1)
            i += 1
        
        return None
    
    def extract_functions(self, code: str) -> List[Dict[str, any]]:
        """
        Extract all functions from C code
        
        Args:
            code: Full C source code
            
        Returns:
            List of dictionaries containing function information
        """
        functions = []
        
        # Remove comments first to avoid false matches
        cleaned_code = self.remove_comments(code)
        
        # Find all function matches
        for match in self.function_pattern.finditer(cleaned_code):
            function_name = match.group(1)
            start_brace_pos = match.end() - 1  # Position of opening brace
            
            # Find function boundaries
            boundaries = self.find_function_boundaries(cleaned_code, start_brace_pos)
            
            if boundaries:
                start_pos, end_pos = boundaries
                function_code = cleaned_code[start_pos:end_pos].strip()
                
                # Calculate line numbers
                lines_before = cleaned_code[:start_pos].count('\n')
                lines_in_function = function_code.count('\n') + 1
                
                functions.append({
                    'name': function_name,
                    'code': function_code,
                    'start_line': lines_before + 1,
                    'end_line': lines_before + lines_in_function,
                    'is_main': function_name == 'main',
                    'char_start': start_pos,
                    'char_end': end_pos
                })
        
        # Sort functions by their position in the code
        functions.sort(key=lambda f: f['char_start'])
        
        return functions
    
    def remove_comments(self, code: str) -> str:
        """Remove C-style comments from code"""
        # Remove single-line comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        return code
    
    def find_vulnerable_functions(self, code: str) -> List[Dict[str, any]]:
        """
        Find functions that are likely to contain vulnerabilities
        
        Args:
            code: Full C source code
            
        Returns:
            List of functions ordered by vulnerability likelihood
        """
        functions = self.extract_functions(code)
        
        # Define vulnerability indicators
        vulnerability_keywords = [
            'strcpy', 'strcat', 'sprintf', 'gets', 'scanf',
            'system', 'exec', 'popen', 'malloc', 'free',
            'buffer', 'input', 'user', 'argv', 'getenv'
        ]
        
        # Score functions based on vulnerability indicators
        for func in functions:
            score = 0
            func_code_lower = func['code'].lower()
            
            # Check for vulnerability keywords
            for keyword in vulnerability_keywords:
                score += func_code_lower.count(keyword)
            
            # Bonus for main function (often contains user input handling)
            if func['is_main']:
                score += 5
            
            # Bonus for functions with array/pointer operations
            score += func_code_lower.count('[')  # Array access
            score += func_code_lower.count('*')  # Pointer operations
            
            func['vulnerability_score'] = score
        
        # Sort by vulnerability score (highest first)
        functions.sort(key=lambda f: f['vulnerability_score'], reverse=True)
        
        return functions
    
    def get_best_function_for_analysis(self, code: str) -> Dict[str, any]:
        """
        Get the best function for vulnerability analysis
        
        Args:
            code: Full C source code
            
        Returns:
            Dictionary with the best function for analysis
        """
        vulnerable_functions = self.find_vulnerable_functions(code)
        
        if not vulnerable_functions:
            # If no functions found, return the original code
            return {
                'name': 'full_code',
                'code': code,
                'start_line': 1,
                'end_line': len(code.split('\n')),
                'is_main': False,
                'vulnerability_score': 0,
                'extraction_method': 'no_functions_found'
            }
        
        # Return the function with highest vulnerability score
        best_function = vulnerable_functions[0]
        best_function['extraction_method'] = 'vulnerability_scoring'
        
        return best_function
    
    def get_function_summary(self, functions: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Get a summary of all extracted functions
        
        Args:
            functions: List of function dictionaries
            
        Returns:
            Summary dictionary
        """
        if not functions:
            return {
                'total_functions': 0,
                'main_function_present': False,
                'average_vulnerability_score': 0,
                'function_names': []
            }
        
        return {
            'total_functions': len(functions),
            'main_function_present': any(f['is_main'] for f in functions),
            'average_vulnerability_score': sum(f.get('vulnerability_score', 0) for f in functions) / len(functions),
            'function_names': [f['name'] for f in functions],
            'most_vulnerable': functions[0]['name'] if functions else None
        }

def extract_c_function(code: str) -> Tuple[str, Dict[str, any]]:
    """
    Convenience function to extract the best C function for analysis
    
    Args:
        code: Full C source code
        
    Returns:
        Tuple of (extracted_function_code, extraction_info)
    """
    parser = CFunctionParser()
    best_function = parser.get_best_function_for_analysis(code)
    all_functions = parser.extract_functions(code)
    
    extraction_info = {
        'selected_function': best_function,
        'all_functions': parser.get_function_summary(all_functions),
        'total_extracted': len(all_functions)
    }
    
    return best_function['code'], extraction_info

# Test the parser
if __name__ == "__main__":
    test_code = """
#include <stdio.h>
#include <string.h>

int safe_function(int x) {
    return x * 2;
}

void vulnerable_function(char* input) {
    char buffer[10];
    strcpy(buffer, input);  // Buffer overflow vulnerability
    printf("%s", buffer);
}

int main(int argc, char** argv) {
    if (argc > 1) {
        vulnerable_function(argv[1]);
    }
    return 0;
}
"""
    
    parser = CFunctionParser()
    functions = parser.extract_functions(test_code)
    
    print("Extracted Functions:")
    for func in functions:
        print(f"- {func['name']} (lines {func['start_line']}-{func['end_line']})")
        print(f"  Vulnerability score: {func.get('vulnerability_score', 0)}")
        print()
    
    best_function = parser.get_best_function_for_analysis(test_code)
    print(f"Best function for analysis: {best_function['name']}")
    print(f"Function code:\n{best_function['code']}") 
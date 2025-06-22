#!/usr/bin/env python3
"""
Test script to demonstrate the improved analysis formatting
"""

def format_analysis_explanation(parsed_response, is_vulnerable):
    """Format the parsed Gemini response into a readable explanation"""
    explanation_parts = []
    
    # Add summary
    summary = parsed_response.get("summary", "")
    if summary:
        explanation_parts.append(f"🔍 **Analysis Summary:**\n{summary}")
    
    if is_vulnerable:
        # Add vulnerability info
        vuln_types = parsed_response.get("vulnerability_types", [])
        if vuln_types:
            types_str = ", ".join(vuln_types)
            explanation_parts.append(f"⚠️ **Vulnerability Types:** {types_str}")
        
        severity = parsed_response.get("severity", "unknown")
        severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
        explanation_parts.append(f"{severity_emoji} **Severity:** {severity.title()}")
        
        # Add recommendations
        recommendations = parsed_response.get("recommendations", [])
        if recommendations:
            rec_text = "\n".join([f"• {rec}" for rec in recommendations[:3]])  # Limit to 3 recommendations
            explanation_parts.append(f"💡 **Recommendations:**\n{rec_text}")
    else:
        # Add security practices for secure code
        practices = parsed_response.get("security_practices", [])
        if practices:
            practices_text = "\n".join([f"• {practice}" for practice in practices[:3]])
            explanation_parts.append(f"✅ **Good Security Practices:**\n{practices_text}")
        
        recommendations = parsed_response.get("recommendations", [])
        if recommendations:
            rec_text = "\n".join([f"• {rec}" for rec in recommendations[:2]])
            explanation_parts.append(f"💡 **Suggestions:**\n{rec_text}")
    
    return "\n\n".join(explanation_parts)

def demo_vulnerable_analysis():
    """Demo of formatted vulnerable code analysis"""
    print("🔴 VULNERABLE CODE ANALYSIS DEMO")
    print("="*50)
    
    # Sample response that would come from Gemini
    sample_response = {
        "summary": "This code contains buffer overflow and command injection vulnerabilities.",
        "vulnerability_types": ["Buffer Overflow", "Command Injection"],
        "severity": "high",
        "recommendations": [
            "Use bounds-checked functions like strncpy instead of strcpy",
            "Sanitize user input before passing to system calls",
            "Implement input validation and length checking"
        ]
    }
    
    formatted = format_analysis_explanation(sample_response, True)
    print(formatted)
    print("\nLength:", len(formatted), "characters")

def demo_secure_analysis():
    """Demo of formatted secure code analysis"""
    print("\n🟢 SECURE CODE ANALYSIS DEMO")
    print("="*50)
    
    # Sample response for secure code
    sample_response = {
        "summary": "This code follows security best practices with proper input validation.",
        "security_practices": [
            "Uses bounds-checked string functions",
            "Validates input length before processing",
            "Proper memory management with null checks"
        ],
        "recommendations": [
            "Consider adding additional error handling",
            "Document security assumptions"
        ]
    }
    
    formatted = format_analysis_explanation(sample_response, False)
    print(formatted)
    print("\nLength:", len(formatted), "characters")

def compare_old_vs_new():
    """Compare old raw output vs new formatted output"""
    print("\n📊 OLD vs NEW COMPARISON")
    print("="*50)
    
    old_raw_output = """This C code (CWE761_Free_Pointer_Not_at_Start_of_Buffer__char_console_81a.cpp) represents the 'source' part of a CWE-761 test case. The vulnerability 'Free Pointer not at Start of Buffer' occurs when `free()` is called on a pointer that does not point to the beginning of a dynamically allocated memory block. This file primarily handles memory allocation and data input, and then passes the allocated buffer to another component (the 'sink') for deallocation. Here's a breakdown of the code's security aspects: 1. **Memory Allocation and Initialization**: Both `bad()` and `goodB2G()` functions correctly allocate 100 bytes of memory using `malloc(100*sizeof(char))` and ensure it's null-terminated at the beginning (`data[0] = '\\0'`). This is a proper setup for a character buffer. 2. **Safe Input Handling**: Input is read from `stdin` using `fgets()`, which is a secure function for input, as it prevents buffer overflows by allowing the specification of the maximum number of characters to read."""
    
    new_formatted = """🔍 **Analysis Summary:**
Buffer overflow vulnerability detected in memory management functions.

⚠️ **Vulnerability Types:** Buffer Overflow, Memory Management

🔴 **Severity:** High

💡 **Recommendations:**
• Use bounds-checked functions like strncpy instead of strcpy
• Implement proper null pointer checks before memory operations
• Add input validation for buffer size limits"""
    
    print("OLD FORMAT (Raw Gemini):")
    print(f"Length: {len(old_raw_output)} characters")
    print(old_raw_output[:200] + "...\n")
    
    print("NEW FORMAT (Structured):")
    print(f"Length: {len(new_formatted)} characters")
    print(new_formatted)

if __name__ == "__main__":
    demo_vulnerable_analysis()
    demo_secure_analysis() 
    compare_old_vs_new()
    
    print("\n✨ BENEFITS OF NEW FORMAT:")
    print("• Concise and scannable")
    print("• Consistent structure")
    print("• Visual hierarchy with emojis")
    print("• Limited length prevents UI overflow")
    print("• Key information highlighted") 
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void vulnerableFunction(char *userInput)
{
    char buffer[64];
    strcpy(buffer, userInput); // Buffer overflow vulnerability
    printf("Input: %s\n", buffer);
}

void anotherVulnerableFunction()
{
    char cmd[256];
    printf("Enter command: ");
    gets(cmd);   // Dangerous function - buffer overflow
    system(cmd); // Command injection vulnerability
}

int riskyMemoryFunction()
{
    char *ptr = malloc(100);
    // Missing null check
    strcpy(ptr, "Some data");

    // Use after free vulnerability
    free(ptr);
    printf("Data: %s\n", ptr); // Use after free

    return 0;
}

int main()
{
    char input[256];
    printf("Enter some text: ");
    scanf("%s", input); // No bounds checking

    vulnerableFunction(input);
    anotherVulnerableFunction();
    riskyMemoryFunction();

    return 0;
}
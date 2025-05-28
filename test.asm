
extern bar_func
extern func

section .data
    arg1 dd 3
    arg2 dd 4
    arg3 dd 6
    arg4 dd 2


section .text
    global _start
    global print_string
    global func

_start:
    push dword [arg1]  ; 将 arg1 的值压栈
    push dword [arg2]   ; 将 arg2 的值压栈
    call bar_func       ; 调用 bar_func
    add esp, 8
    push arg3
    push arg4
    call func

    add esp, 16
    mov ebx, 0
    mov eax, 1
    int 0x80

print_string:

 mov edx, [esp+8]
 mov ecx, [esp+4]
 mov ebx, 1
 mov eax, 4
 int 0x80
 ret



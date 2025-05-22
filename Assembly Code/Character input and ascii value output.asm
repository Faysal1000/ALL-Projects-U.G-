.model small
.stack 100h
.data  
    imsg db "Enter a character: $"
    m db 13,10,"The ascii value of the character is= $"  
.code
main proc
    mov ax, @data
    mov ds, ax
    
    lea dx, imsg
    mov ah, 09h
    int 21h
    
    mov ah, 01h
    int 21h  
    push ax
    
    mov ah, 09h
    lea dx, m
    int 21h
    
    pop ax
    call print_decimal
    
    
    exit:
    mov ah, 4ch
    int 21h
    main endp



print_decimal proc
    xor ah, ah
    mov cx, 0
    mov bx, 10

convert_loop:
    xor dx, dx
    div bx
    push dx
    inc cx
    cmp ax, 0
    jne convert_loop

print_d_loop:
    pop dx
    add dl, '0'
    mov ah, 02h
    int 21h
    loop print_d_loop    
    
    mov dl, 'd'
    int 21h  
    ret        
print_decimal endp 
end main
    
    
    
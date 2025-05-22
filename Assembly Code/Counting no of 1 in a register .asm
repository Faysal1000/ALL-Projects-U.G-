.model small
.stack 100h
.data  
    imsg db "Enter 3 character: $"
    r db 13,10,"The reverse order is: $"  
.code
main proc
    mov ax, @data
    mov ds, ax
    
    lea dx, imsg
    mov ah, 09h
    int 21h
    
    mov ah, 01h
    mov cx, 3
    
    input:
    int 21h
    push ax
    loop input 
    
    lea dx, r
    mov ah, 09h
    int 21h
    
    mov cx,3
    print:
    pop ax
    mov dl, al
    mov ah, 02h
    int 21h
    loop print
    
    
    exit:
    mov ah, 4ch
    int 21h
    main endp
end main
    
    
    
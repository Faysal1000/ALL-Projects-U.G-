.model small
.stack 100h
.data  
    imsg db "Enter one Hex digit (0 or 1): $"
    msg db 13,10,"You entered: $" 
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
    
    lea dx, msg
    mov ah, 09h
    int 21h
    
    pop ax
    mov dl, al
    mov ah, 02h
    int 21h
    
    mov ah, 4ch
    int 21h
    main endp
end main
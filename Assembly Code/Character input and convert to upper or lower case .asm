.model small
.stack 100h
.data  
    imsg db "Enter 1 digit Decimal (0-9): $"
    e db 13,10,"The number is even. $"  
    o db 13,10,"The number is odd. $"
.code
main proc
    mov ax, @data
    mov ds, ax
    
    lea dx, imsg
    mov ah, 09h
    int 21h
    

    mov ah, 01h
    int 21h
    
    sub al, '0'
    test al, 1
    jz even
    jmp odd
    
    even:
    lea dx, e
    mov ah, 09h
    int 21h
    jmp exit
    
    odd:
    lea dx, o
    mov ah, 09h
    int 21h
    jmp exit
    
    exit:
    mov ah, 4ch
    int 21h
    main endp
end main
    
    
    
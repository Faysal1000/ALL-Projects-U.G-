.model small
.stack 100h
.data  
    imsg db "Enter lower (a-z) or upper (A-Z) case letter : $"
    u db 13,10,"The Upper case is:  $"  
    l db 13,10,"The Lower case is: $"
.code
main proc
    mov ax, @data
    mov ds, ax
    
    lea dx, imsg
    mov ah, 09h
    int 21h
    
    mov ah, 01h
    int 21h
    
    cmp al, 'a'
    push ax
     
    jge lower
    jmp upper
    
    lower:
    lea dx, u
    mov ah, 09h
    int 21h 
    pop ax
    and al, 0DFh
    mov ah, 02h
    mov dl, al
    int 21h
    jmp exit
    
    upper:
    lea dx, l
    mov ah, 09h
    int 21h
    pop ax
    or al, 20h
    mov ah, 02h
    mov dl, al
    int 21h
    jmp exit
    
    
    
    exit:
    mov ah, 4ch
    int 21h
    main endp
end main
    
    
    
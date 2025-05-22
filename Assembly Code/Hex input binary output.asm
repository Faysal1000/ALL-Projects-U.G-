.model small
.stack 100h
.data  
    imsg db "Enter 8 bit binary value: $"
    msg db 13,10,"Hex eqivalent is: $" 
.code
main proc
    mov ax, @data
    mov ds, ax
    
    lea dx, imsg
    mov ah, 09h
    int 21h
    
    mov ah, 01h
    mov bl, 0
    mov cx, 8

input:
    int 21h
    shl bl, 1
    sub al, '0'
    or bl, al
    loop input
    
    push ax
    
    lea dx, msg
    mov ah, 09h
    int 21h
    
    pop ax
    mov al, bl
    call print_hex
    
    mov ah, 4ch
    int 21h
    main endp  

print_hex proc
    mov bl, al
    mov ah, al
    shr ah, 4
    mov al, ah
    call convert_hex

    mov al, bl
    and al, 0Fh
    call convert_hex

    mov dl, 'h'
    mov ah, 02h
    int 21h 
    ret
print_hex endp    

convert_hex proc
    cmp al, 10
    jl digit_case
    add al, 'A' - 10
    jmp print_char

digit_case:
    add al, '0'

print_char:
    mov dl, al
    mov ah, 02h
    int 21h
    ret
convert_hex endp
end main
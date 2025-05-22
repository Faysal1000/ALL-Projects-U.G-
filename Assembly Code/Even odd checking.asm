.model small
.stack 100h
.data  
    imsg db "Enter 1 digit Hex: $"
    msg db 13,10,"Binary eqivalent is: $" 
.code
main proc
    mov ax, @data
    mov ds, ax
    
    lea dx, imsg
    mov ah, 09h
    int 21h
    
    mov ah, 01h
    mov bl, 0
    mov cx, 1

input:
    int 21h   
    call convert_bin 
    mov bl, al
    loop input 
    
    mov al, bl
    push ax
    
    lea dx, msg
    mov ah, 09h
    int 21h
    
    pop ax
    mov al, bl 
    mov cx, 4
    call print_binary
    
    mov ah, 4ch
    int 21h
    main endp 


print_binary proc
    mov bl, al
    shl bl,4
print_loop:
    rol bl, 1
    jnc zero
    mov dl, '1'
    jmp print
zero:
    mov dl, '0'
print:
    mov ah, 02h
    int 21h
    loop print_loop
    mov dl, 'b'
    int 21h
    ret
print_binary endp    

convert_bin proc
    cmp al, 'A'
    jge letter
    jmp digit
    
    letter:
    sub al, 'A' 
    add al, 10d
   
    digit:
    sub al, '0'
    
    ret
    convert_bin endp
    
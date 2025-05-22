.model small
.stack 100h
.data 
    o db ?
    r db ?
    s  db ' = $'
    m1 db 13, 10, "The reversed number is Greater than 100$"
    m2 db 13, 10, "The reversed number is less than 100$"
    m3 db 13, 10, "The reversed number is equal to 100$"
    m4 db 13, 10, "The original number is = $"
    m5 db 13, 10, "The reverse number is = $"
    m6 db "Please Enter 8 bit Binary Value: $"

.code
main proc
    mov ax, @data
    mov ds, ax
    
    lea dx, m6
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

    mov cx, 8
    mov al, bl
    mov o, al
    mov bh, 0

reverse:
    shl al, 1
    rcr bh, 1
    loop reverse

    mov al, bh
    mov r, al
    cmp al, 100
    ja greater
    jb less
    je equal

greater:
    lea dx, m1
    mov ah, 09h
    int 21h
    jmp display

less:
    lea dx, m2
    mov ah, 09h
    int 21h
    jmp display

equal:
    lea dx, m3
    mov ah, 09h
    int 21h
    jmp display

display:
    lea dx, m4
    mov ah, 09h
    int 21h
           
    mov cl, 8
    mov al, o
    call print_binary
    
    lea dx, s
    mov ah, 09h
    int 21h 

    mov al, o
    call print_decimal 
    
    lea dx, s
    mov ah, 09h
    int 21h 

    mov al, o
    call print_hex
           
    lea dx, m5
    mov ah, 09h
    int 21h
        
    mov cl, 8
    mov al, r
    call print_binary
    
    lea dx, s
    mov ah, 09h
    int 21h 
        
    mov al, r
    call print_decimal 
     
    lea dx, s
    mov ah, 09h
    int 21h 

    mov al, r
    call print_hex

    mov ah, 4Ch
    int 21h
main endp   

print_binary proc
    mov bl, al
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

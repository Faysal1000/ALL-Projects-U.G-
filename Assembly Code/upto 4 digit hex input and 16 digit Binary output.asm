.model small
.stack 100h

.data
    m1 db "Enter up to 4-digit Hex no: $"
    m2 db 13, 10, "Invalid, Try Again: $"
    m3 db 13, 10, "Binary No is: $"
    i  db 5 dup('$')        ; buffer for up to 4 digits

.code
main proc
    mov ax, @data
    mov ds, ax 
    
    lea dx, m1
    mov ah, 09h
    int 21h

input_start:
    lea si, i              ; SI points to input buffer
    xor cx, cx             ; CX = 0, count of valid digits

input:
    mov ah, 01h
    int 21h                ; read char into AL

    cmp al, 13             ; check for Enter key
    je convert

    call validation        ; convert AL to numeric
    jc input_start         ; if invalid, restart

    mov [si], al           ; store converted digit
    inc si
    inc cx                 ; increment count

    cmp cx, 4
    je convert
    jmp input

convert:
    lea si, i              ; point to first digit
    xor bx, bx             ; clear BX to build final number

convert_loop:
    shl bx, 4              ; make room for next hex digit
    mov al, [si]
    inc si
    xor ah, ah             ; clear ah before storing
    or bx, ax              ; OR in the new digit
    loop convert_loop
    
print:
    lea dx, m3
    mov ah, 09h
    int 21h
    mov cx, 16

print_result:
    rol bx, 1              ; rotate bit into carry
    jc digit_one
    mov dl, '0'
    jmp print_out

digit_one:
    mov dl, '1'

print_out:
    mov ah, 02h
    int 21h
    loop print_result

exit:
    mov ah, 4Ch
    int 21h
main endp

; ===== Validation Procedure =====
; AL: input character
; returns converted value in AL if valid
; sets carry flag on invalid input

validation proc
    cmp al, '0'
    jb invalid
    cmp al, '9'
    jbe digit

    cmp al, 'A'
    jb invalid
    cmp al, 'F'
    jbe hexletter

invalid:
    lea dx, m2
    mov ah, 09h
    int 21h
    stc
    ret

digit:
    sub al, '0'
    clc
    ret

hexletter:
    sub al, 'A'
    add al, 10
    clc
    ret
validation endp

end main

.model small
.stack 100h

.data
    m1 db "Enter up to 16-bit binary number: $"
    m2 db 13, 10, "Invalid input. Try Again: $"
    m3 db 13, 10, "Hexadecimal is: $"
    bin_input db 17 dup('$')     ; 16 bits max + null terminator

.code
main proc
    mov ax, @data
    mov ds, ax     
    
    lea dx, m1
    mov ah, 09h
    int 21h

input_start:
    lea si, bin_input
    xor cx, cx             ; CX = 0 (bit counter)
    
input:
    mov ah, 01h
    int 21h                ; read char into AL

    cmp al, 13             ; Enter pressed?
    je convert

    cmp al, '0'
    je valid_bit
    cmp al, '1'
    je valid_bit

invalid:
    lea dx, m2
    mov ah, 09h
    int 21h
    jmp input_start

valid_bit:
    mov [si], al
    inc si
    inc cx

    cmp cx, 16             ; Max 16 bits allowed
    je convert
    jmp input

convert:
    lea si, bin_input
    xor bx, bx             ; BX = accumulator

build_number:
    shl bx, 1              ; shift left to make room for bit
    mov al, [si]
    inc si
    cmp al, '1'
    jne skip_or
    or bx, 1
skip_or:
    loop build_number

print:
    lea dx, m3
    mov ah, 09h
    int 21h
    
    mov ax, bx
    call print_hex

exit:
    mov ah, 4Ch
    int 21h
main endp   


print_hex proc
    ; Input: AX contains 16-bit number
    ; Output: Prints 4 hex digits + 'h'

    push ax
    mov cx, ax          ; Copy original value to CX for safe keeping

    ; Print highest nibble (bits 12-15)
    mov ax, cx
    shr ax, 12
    and al, 0Fh
    call convert_hex

    ; Next nibble (bits 8-11)
    mov ax, cx
    shr ax, 8
    and al, 0Fh
    call convert_hex

    ; Next nibble (bits 4-7)
    mov ax, cx
    shr ax, 4
    and al, 0Fh
    call convert_hex

    ; Lowest nibble (bits 0-3)
    mov ax, cx
    and al, 0Fh
    call convert_hex

    ; Append 'h'
    mov dl, 'h'
    mov ah, 02h
    int 21h

    pop ax
    ret
print_hex endp

; ---------- Converts lower 4 bits in AL to ASCII hex digit ----------
convert_hex proc
    cmp al, 10
    jl digit_case       ; If less than 10, it's 0-9
    add al, 'A' - 10     ; For A-F
    jmp print_char

digit_case:
    add al, '0'          ; For 0-9

print_char:
    mov dl, al           ; Move digit to DL for printing
    mov ah, 02h
    int 21h
    ret
convert_hex endp

end main

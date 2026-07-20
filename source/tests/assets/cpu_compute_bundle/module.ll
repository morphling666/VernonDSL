define i32 @__vernon_cpu_fill(ptr %invocation) {
entry:
  %arguments = load ptr, ptr %invocation, align 1
  %buffer = load ptr, ptr %arguments, align 1
  %gid_x_address = getelementptr i8, ptr %arguments, i64 8
  %gid_y_address = getelementptr i8, ptr %arguments, i64 12
  %gid_z_address = getelementptr i8, ptr %arguments, i64 16
  %x = load i32, ptr %gid_x_address, align 1
  %y = load i32, ptr %gid_y_address, align 1
  %z = load i32, ptr %gid_z_address, align 1
  %z_stride = mul i32 %z, 6
  %y_stride = mul i32 %y, 3
  %zy = add i32 %z_stride, %y_stride
  %index = add i32 %zy, %x
  %index64 = zext i32 %index to i64
  %destination = getelementptr float, ptr %buffer, i64 %index64
  %x_value = uitofp i32 %x to float
  %y_tens = mul i32 %y, 10
  %y_value = uitofp i32 %y_tens to float
  %z_hundreds = mul i32 %z, 100
  %z_value = uitofp i32 %z_hundreds to float
  %xy_value = fadd float %x_value, %y_value
  %value = fadd float %xy_value, %z_value
  store float %value, ptr %destination, align 1
  ret i32 0
}

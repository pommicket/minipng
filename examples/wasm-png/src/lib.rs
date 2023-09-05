#[no_mangle]
pub unsafe fn decode_png(input: *const u8, input_len: usize, output: *mut u8, output_len: usize) -> usize {
	let input = core::slice::from_raw_parts(input, input_len);
	let output = core::slice::from_raw_parts_mut(output, output_len);
	
	let decoder = png::Decoder::new(input);
	let Ok(mut reader) = decoder.read_info() else {
		return 0;
	};
	let required_bytes = reader.output_buffer_size();
	if output_len >= required_bytes {
		if reader.next_frame(output).is_err() {
			return 0;
		}
	}
	required_bytes
}

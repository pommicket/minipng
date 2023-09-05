#[no_mangle]
pub unsafe fn decode_png(input: *const u8, input_len: usize, output: *mut u8, output_len: usize) -> usize {
	let input = core::slice::from_raw_parts(input, input_len);
	let output = core::slice::from_raw_parts_mut(output, output_len);
	let Ok(header) = tiny_png::decode_png_header(input) else {
		return 0;
	};
	let required_bytes = header.required_bytes();
	if output_len >= required_bytes {
		if tiny_png::decode_png(input, output).is_err() {
			return 0;
		}
	}
	required_bytes
}

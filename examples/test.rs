fn main() {
	let mut data = &include_bytes!("test.png")[..];
	let header = tiny_png::read_png_header(&mut data).unwrap();
	let mut buf = vec![0; header.required_bytes()];
	let result = tiny_png::read_png(&mut data, Some(&header), &mut buf);
	println!("{result:?}");
}

fn main() {
	let mut my_buffer = vec![0; 1 << 20]; // hope this is big enough!
	let mut png = &include_bytes!("image.png")[..];
	let image = tiny_png::read_png(&mut png, None, &mut my_buffer).expect("bad PNG");
	println!("{}×{} image", image.width(), image.height());
	let pixels = image.pixels();
	println!(
		"top-left pixel is #{:02x}{:02x}{:02x}",
		pixels[0], pixels[1], pixels[2]
	);
	// (^ this only makes sense for RGB 8bpc images)
}

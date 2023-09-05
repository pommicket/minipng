//! this file is just meant for profiling

use std::hint::black_box;

fn main() {
	if cfg!(debug_assertions) {
		eprintln!("this should only be run in release mode");
	} else {
		println!("pid = {}", std::process::id());
		let large_image = black_box(std::fs::read("benches/large.png").unwrap());

		for _ in 0..100 {
			let mut png = &large_image[..];
			let header = tiny_png::read_png_header(&mut png).unwrap();
			let mut buf = vec![0; header.required_bytes()];
			let data = tiny_png::read_png(&mut png, Some(&header), &mut buf).unwrap();
			std::hint::black_box(data);
		}
	}
}
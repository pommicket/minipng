use crate::{decode_png, decode_png_header, ColorType, Error};
use std::{fs, sync::Mutex};

#[derive(Debug)]
enum Flaw {
	ErrorFromValidPNG,
	NoErrorFromInvalidPNG,
	DecodedMismatch,
	ConvertedMismatch,
}

impl core::fmt::Display for Flaw {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		match self {
			Self::ErrorFromValidPNG => write!(f, "decoding valid PNG gives error"),
			Self::NoErrorFromInvalidPNG => write!(f, "decoding invalid PNG gives no error"),
			Self::DecodedMismatch => write!(f, "image incorrectly decoded"),
			Self::ConvertedMismatch => write!(f, "image incorrectly converted"),
		}
	}
}

const LARGE_BUF: Mutex<Vec<u8>> = Mutex::new(vec![]);
fn test_bytes(bytes: &[u8]) -> Result<(), Flaw> {
	let decoder = png::Decoder::new(bytes);
	let mut is_valid = true;
	if let Ok(mut reader) = decoder.read_info() {
		let mut png_buf = vec![0; reader.output_buffer_size()];
		if let Ok(png_header) = reader.next_frame(&mut png_buf) {
			let png_bytes = &png_buf[..png_header.buffer_size()];

			let tiny_header = match decode_png_header(bytes) {
				Ok(h) => h,
				Err(Error::UnsupportedInterlace) => return Ok(()),
				Err(_) => return Err(Flaw::ErrorFromValidPNG),
			};
			let mut tiny_buf = vec![0; tiny_header.required_bytes_rgba8bpc()];
			let mut image =
				decode_png(bytes, &mut tiny_buf).map_err(|_| Flaw::ErrorFromValidPNG)?;
			let tiny_bytes = image.pixels();
			if png_bytes != tiny_bytes {
				return Err(Flaw::DecodedMismatch);
			}
			let (_, mut data) = png_decoder::decode(bytes).unwrap();
			if matches!(image.color_type(), ColorType::Gray | ColorType::Rgb) {
				// pretend there's no tRNS chunk.
				// there shouldnt be one who the fucks stupid idea was that.
				for i in 0..data.len() / 4 {
					data[4 * i + 3] = 255;
				}
			}
			image.convert_to_rgba8bpc().unwrap();
			if data != image.pixels() {
				return Err(Flaw::ConvertedMismatch);
			}
		} else {
			is_valid = false;
		}
	} else {
		is_valid = false;
	}

	if !is_valid && decode_png(bytes, &mut LARGE_BUF.lock().unwrap()).is_ok() {
		return Err(Flaw::NoErrorFromInvalidPNG);
	}
	Ok(())
}

fn test_images_in_dir(dir: &str) {
	for entry in fs::read_dir(dir).unwrap() {
		let entry = entry.unwrap();
		let r#type = entry.file_type().unwrap();
		let path = entry.path().to_str().unwrap().to_string();
		if r#type.is_file() {
			if let Err(flaw) = test_bytes(&std::fs::read(&path).unwrap()) {
				panic!("flaw for file {path}: {flaw}");
			}
		} else if r#type.is_dir() {
			test_images_in_dir(&path);
		}
		println!("{path} ... \x1b[32mok\x1b[0m");
	}
}

#[test]
fn test_images() {
	*LARGE_BUF.lock().unwrap() = vec![0; 10 << 20];

	test_images_in_dir("test");
}

#[test]
fn test_bad_png() {
	let mut data = &b"hello"[..];
	let err = decode_png_header(&mut data).unwrap_err();
	assert!(matches!(err, Error::NotPng));
}
#[test]
fn test_buffer_too_small() {
	let png = &include_bytes!("../test/ouroboros.png")[..];
	let mut buffer = [0; 128];
	let err = decode_png(png, &mut buffer[..]).unwrap_err();
	assert!(matches!(err, Error::BufferTooSmall));
}

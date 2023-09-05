use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn run_benches(c: &mut Criterion) {
	let large_image = black_box(include_bytes!("large.png"));
	let small_image = black_box(include_bytes!("small.png"));

	let mut group = c.benchmark_group("large-image");
	group.sample_size(50);

	group.bench_function("tiny-png", |b| {
		b.iter(|| {
			let png = &large_image[..];
			let header = tiny_png::read_png_header(png).unwrap();
			let mut buf = vec![0; header.required_bytes()];
			let data = tiny_png::read_png(png, &mut buf).unwrap();
			std::hint::black_box(data);
		})
	});

	group.bench_function("png", |b| {
		b.iter(|| {
			let png = &large_image[..];
			let decoder = png::Decoder::new(png);
			let mut reader = decoder.read_info().unwrap();
			let mut png_buf = vec![0; reader.output_buffer_size()];
			reader.next_frame(&mut png_buf).unwrap();
			std::hint::black_box(png_buf);
		})
	});
	group.finish();

	let mut group = c.benchmark_group("small-image");
	group.sample_size(1000);
	group.bench_function("tiny-png", |b| {
		b.iter(|| {
			let png = &small_image[..];
			let header = tiny_png::read_png_header(png).unwrap();
			let mut buf = vec![0; header.required_bytes()];
			let data = tiny_png::read_png(png, &mut buf).unwrap();
			std::hint::black_box(data);
		})
	});
	group.bench_function("png", |b| {
		b.iter(|| {
			let png = &small_image[..];
			let decoder = png::Decoder::new(png);
			let mut reader = decoder.read_info().unwrap();
			let mut png_buf = vec![0; reader.output_buffer_size()];
			reader.next_frame(&mut png_buf).unwrap();
			std::hint::black_box(png_buf);
		})
	});
	group.finish();
}

criterion_group!(benches, run_benches);
criterion_main!(benches);

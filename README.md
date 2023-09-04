## tiny-png

Tiny Rust PNG decoder.

This decoder can be used without `std` or `alloc` by disabling the `std` feature (enabled by default).

## Goals

- Correctly decode all non-interlaced PNG files (if a non-interlaced PNG is decoded incorrectly, please report it as a bug)
- Small code size
- No panics (if any function panics, please report it as a bug).
- No unsafe code

## Non-goals

- Adam7 interlacing (interlaced PNGs are rare, and this would require additional code complexity
  — but if you want to add it behind a feature gate, feel free to)
- Sacrificing code size a lot for speed (except maybe with a feature enabled)
- Checking block CRCs (increased code complexity probably isn't worth it,
  and there's already Adler32 checksums for IDAT chunks)

## Example usage

Basic usage:

```rust
let mut buffer = vec![0; 1 << 20]; // hope this is big enough!
let mut png = &include_bytes!("image.png")[..];
let image = tiny_png::read_png(&mut png, None, &mut buffer).expect("bad PNG");
println!("{}×{} image", image.width(), image.height());
let pixels = image.pixels();
println!("top-left pixel is #{:02x}{:02x}{:02x}", pixels[0], pixels[1], pixels[2]);
// (^ this only makes sense for RGB 8bpc images)
```

Allocate the right number of bytes:

```rust
let mut png = &include_bytes!("image.png")[..];
let header = tiny_png::read_png_header(&mut png).expect("bad PNG");
println!("need {} bytes of memory", header.required_bytes());
let mut buffer = vec![0; header.required_bytes()];
let image = tiny_png::read_png(&mut png, Some(&header), &mut buffer).expect("bad PNG");
println!("{}×{} image", image.width(), image.height());
let pixels = image.pixels();
println!("top-left pixel is #{:02x}{:02x}{:02x}", pixels[0], pixels[1], pixels[2]);
// (^ this only makes sense for RGB 8bpc images)
```

## Features

- `std` (default: enabled) — use standard library. enabling it allows you to read from `BufReader<File>` but
   adds `std` as a dependency.
- `adler` (default: disabled) — check Adler-32 checksums. slightly increases code size and
  slightly decreases performance but verifies integrity of PNG files.

## Development

A `pre-commit` git hook is provided to run `cargo fmt` and `cargo clippy`. You can install it with:

```sh
ln -s ../../pre-commit .git/hooks/
```

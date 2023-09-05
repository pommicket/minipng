## tiny-png

Tiny Rust PNG decoder.

This decoder can be used without `std` or `alloc` by disabling the `std` feature (enabled by default).

Also it has tiny code size (e.g. 5x smaller `.wasm.gz` size compared to the `png` crate — see `check-size.sh`).

## Goals

- Correctly decode all valid non-interlaced PNG files (on 32-bit platforms, some very large images
  might fail because of `usize::MAX`).
- Small code size &amp; complexity
- No dependencies other than `core`
- No panics
- Minimal if any unsafe code

## Non-goals

- Adam7 interlacing (increases code complexity and interlaced PNGs are rare anyways)
- Significantly sacrificing code size for speed (except maybe with a feature enabled)
- Checking block CRCs (increases code complexity
  and there’s already Adler32 checksums for IDAT chunks)

## Example usage

Basic usage:

```rust
let mut buffer = vec![0; 1 << 20]; // hope this is big enough!
let mut png = &include_bytes!("../examples/image.png")[..];
let image = tiny_png::read_png(&mut png, None, &mut buffer).expect("bad PNG");
println!("{}×{} image", image.width(), image.height());
let pixels = image.pixels();
println!("top-left pixel is #{:02x}{:02x}{:02x}", pixels[0], pixels[1], pixels[2]);
// (^ this only makes sense for RGB 8bpc images)
```

More complex example that allocates the right number of bytes:

```rust
let mut png = &include_bytes!("../examples/image.png")[..];
let header = tiny_png::read_png_header(&mut png).expect("bad PNG");
let mut buffer = vec![0; header.required_bytes_rgba8bpc()];
let mut image = tiny_png::read_png(&mut png, Some(&header), &mut buffer).expect("bad PNG");
image.convert_to_rgba8bpc();
assert!(png.is_empty(), "extra data after PNG image end");
println!("{}×{} image", image.width(), image.height());
let pixels = image.pixels();
println!("top-left pixel is #{:02x}{:02x}{:02x}{:02x}", pixels[0], pixels[1], pixels[2], pixels[3]);
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

## License

> Zero-Clause BSD
> 
> Permission to use, copy, modify, and/or distribute this software for
> any purpose with or without fee is hereby granted.
> 
> THE SOFTWARE IS PROVIDED “AS IS” AND THE AUTHOR DISCLAIMS ALL
> WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES
> OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE
> FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY
> DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
> AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
> OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

(Note: all the test PNG images are either in the U.S. public domain or CC0-licensed.)

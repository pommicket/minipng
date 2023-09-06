## minipng

[crates.io page](https://crates.io/crates/minipng) &middot; [repository](https://github.com/pommicket/minipng)

Tiny Rust PNG decoder with no dependencies (not even `std` or `alloc`)
and tiny code size (e.g. &gt;8 times smaller `.wasm.gz` size compared to the `png` crate — see `check-size.sh`).

## Goals

- Correctly decode all valid non-interlaced PNG files (on ≤32-bit platforms, some very large images
  might fail because of `usize::MAX`).
- Small code size &amp; complexity
- No dependencies other than `core`
- No panics
- Minimal unsafe code

## Non-goals

- Adam7 interlacing (increases code complexity and interlaced PNGs are rare anyways)
- significantly sacrificing code size/complexity for speed
- checking block CRCs (increases code complexity
  and there’s already Adler32 checksums for IDAT chunks)
- ancillary chunks (tEXt, iCCP, etc.)
- correctly handling non-indexed image with tRNS chunk (who uses this?)
- animated PNGs

## Example usage

Basic usage:

```rust
let mut buffer = vec![0; 1 << 20]; // hope this is big enough!
let png = &include_bytes!("../examples/image.png")[..];
let image = minipng::decode_png(png, &mut buffer).expect("bad PNG");
println!("{}×{} image", image.width(), image.height());
let pixels = image.pixels();
println!("top-left pixel is #{:02x}{:02x}{:02x}", pixels[0], pixels[1], pixels[2]);
// (^ this only works for RGB(A) 8bpc images)
```

More complex example that allocates the right number of bytes and handles all color formats:

```rust
let png = &include_bytes!("../examples/image.png")[..];
let header = minipng::decode_png_header(png).expect("bad PNG");
let mut buffer = vec![0; header.required_bytes_rgba8bpc()];
let mut image = minipng::decode_png(png, &mut buffer).expect("bad PNG");
image.convert_to_rgba8bpc();
println!("{}×{} image", image.width(), image.height());
let pixels = image.pixels();
println!("top-left pixel is #{:02x}{:02x}{:02x}{:02x}", pixels[0], pixels[1], pixels[2], pixels[3]);
```

## Features

- `adler` (default: disabled) — check Adler-32 checksums. slightly increases code size and
  slightly decreases performance but verifies integrity of PNG files.

## Development

A `pre-commit` git hook is provided to run `cargo fmt` and `cargo clippy`. You can install it with:

```sh
ln -s ../../pre-commit .git/hooks/
```

## Testing

All PNG files in the `test` directory are tested by `cargo t` (NOTE: `cargo test`
doesn‘t log as much progress because there’s no way of dynamically generating
tests and no way of enabling `--nocapture` by default *grumble grumble*).

## Performance

Benchmarks (see `cargo bench`) show that `minipng` is about 50% slower than the `png` crate
for large images, but faster than `png` for small images.

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


#!/bin/sh

cd examples/wasm-tiny-png && cargo b --release && cd ../.. && wasm-opt -Oz examples/wasm-tiny-png/target/wasm32-unknown-unknown/release/wasm_tiny_png.wasm -o target/tiny_png.wasm && gzip -f target/tiny_png.wasm || exit 1
cd examples/wasm-png && cargo b --release && cd ../.. && wasm-opt -Oz examples/wasm-png/target/wasm32-unknown-unknown/release/wasm_png.wasm -o target/png.wasm && gzip -f target/png.wasm || exit 1

wc -c target/png.wasm.gz target/tiny_png.wasm.gz | head -n2

#!/bin/sh

cd examples/wasm-minipng && cargo b --release && cd ../.. && wasm-opt --strip-debug -Oz examples/wasm-minipng/target/wasm32-unknown-unknown/release/wasm_minipng.wasm -o target/minipng.wasm && gzip -k -f target/minipng.wasm || exit 1
cd examples/wasm-png && cargo b --release && cd ../.. && wasm-opt --strip-debug -Oz examples/wasm-png/target/wasm32-unknown-unknown/release/wasm_png.wasm -o target/png.wasm && gzip -k -f target/png.wasm || exit 1

wc -c target/png.wasm.gz target/minipng.wasm.gz | head -n2

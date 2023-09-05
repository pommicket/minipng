#!/bin/sh

# test all feature sets
for mode in '' '--release'; do
	cargo test $mode || exit 1
	cargo test $mode --features adler || exit 1
done

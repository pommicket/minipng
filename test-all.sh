#!/bin/sh

# test all feature sets
for mode in '' '--release'; do
	cargo t $mode
	cargo t $mode --no-default-features
	cargo t $mode --features adler
	cargo t $mode --no-default-features --features adler
done

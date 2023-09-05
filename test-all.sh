#!/bin/sh

# test all feature sets
for mode in '' '--release'; do
	cargo t $mode || exit 1
	cargo t $mode --no-default-features || exit 1
	cargo t $mode --features adler || exit 1
	cargo t $mode --no-default-features --features adler || exit 1
done

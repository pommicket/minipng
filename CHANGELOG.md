## 0.1.1

- add overflow check for chunk length.
  this could have resulted in debug-only panics for maliciously crafted images.
- add “impossible compressed size” check which slightly mitigates the
  problem of a malicious image causing you to allocate a shitton of memory.

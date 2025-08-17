# StringTape

A memory-efficient string storage library compatible with Apache Arrow's string array format.
Stores multiple strings in a contiguous memory layout using offset-based indexing, similar to Arrow's `String` and `LargeString` arrays.

- __Apache Arrow Compatible__: Uses i32/i64 offsets like Arrow's String/LargeString arrays
- __Memory Efficient__: All strings stored in a single contiguous buffer
- __Type Safe__: Generic over offset types (i32 for 32-bit, i64 for 64-bit)
- __`no_std` Support__: Can be used in embedded environments
- __Zero Dependencies__: Pure Rust implementation

## Quick Start

```rust
use stringtape::{StringTape32, StringTapeError};

// Create a new StringTape with 32-bit offsets
let mut tape = StringTape32::new();
tape.push("hello")?;
tape.push("world")?;

assert_eq!(tape.len(), 2);
assert_eq!(&tape[0], "hello");
assert_eq!(tape.get(1), Some("world"));

// Iterate over strings
for s in &tape {
    println!("{}", s);
}

// Build from iterator
let tape2: StringTape32 = ["a", "b", "c"].into_iter().collect();
assert_eq!(tape2.len(), 3);
# Ok::<(), StringTapeError>(())
```

## Memory Layout

StringTape uses the same memory layout as Apache Arrow string arrays:

```text
Data buffer:    [h,e,l,l,o,w,o,r,l,d]
Offset buffer:  [0, 5, 10]
```

## API Overview

### Creation

```rust
// Empty tape
let tape = StringTape32::new();

// Pre-allocated capacity
let tape = StringTape32::with_capacity(1024, 100)?; // 1KB data, 100 strings

// From iterator
let tape: StringTape32 = ["a", "b", "c"].into_iter().collect();
```

### Adding Strings

```rust
let mut tape = StringTape32::new();

// Single string
tape.push("hello")?;

// Multiple strings
tape.extend(["world", "foo", "bar"])?;
```

### Accessing Strings

```rust
// By index (panics if out of bounds)
let s = &tape[0];

// Safe access
let s = tape.get(0); // Returns Option<&str>

// Iteration
for s in &tape {
    println!("{}", s);
}

// Collect to Vec
let strings: Vec<&str> = tape.iter().collect();
```

### Capacity Management

```rust
let mut tape = StringTape32::new();

// Check sizes
println!("Strings: {}", tape.len());
println!("Data bytes: {}", tape.data_len());
println!("Data capacity: {}", tape.data_capacity());

// Reserve space
tape.reserve(1024, 100)?; // 1KB data, 100 strings

// Clear contents
tape.clear();

// Truncate
tape.truncate(5); // Keep first 5 strings
```

### Apache Arrow Interop

```rust
let mut tape = StringTape32::new();
tape.push("hello")?;
tape.push("world")?;

// Get raw pointers for Arrow compatibility
let (data_ptr, offsets_ptr, data_len, string_count) = tape.as_raw_parts();

// These can be used to construct Arrow arrays without copying
```

## Offset Types

StringTape supports two offset types:

- __`StringTape32`__ (alias for `StringTape<i32>`): Compatible with Arrow's `String` arrays, supports up to 2GB of string data
- __`StringTape64`__ (alias for `StringTape<i64>`): Compatible with Arrow's `LargeString` arrays, supports virtually unlimited string data

```rust
use stringtape::{StringTape32, StringTape64};

let small_tape = StringTape32::new(); // For < 2GB of data
let large_tape = StringTape64::new(); // For >= 2GB of data
```

## no_std Support

StringTape can be used in `no_std` environments:

```toml
[dependencies]
stringtape = { version = "0.1", default-features = false }
```

In `no_std` mode:
- Requires `alloc` for dynamic allocation
- All functionality is preserved
- Error types implement `Display` but not `std::error::Error`

## Testing

Run tests for both `std` and `no_std` configurations:

```bash
# Test with std (default)
cargo test

# Test without std
cargo test --no-default-features

# Test both configurations
cargo test && cargo test --no-default-features

# Run documentation tests
cargo test --doc

# Test with all features
cargo test --all-features
```

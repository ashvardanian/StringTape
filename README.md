# StringTape

A memory-efficient string storage library compatible with [Apache Arrow](https://arrow.apache.org/)'s string array format.
Stores multiple strings in a contiguous memory layout using offset-based indexing, similar to Arrow's `String` and `LargeString` arrays.

- __Apache Arrow Compatible__: Matching `String` and `LargeString` arrays
- __Memory Efficient__: All strings stored in two contiguous buffers
- __Zero-Copy Views__: Efficient slicing with `[i..n]` range syntax
- __Zero Dependencies__: Pure Rust implementation, with `no_std` support

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

### Basic Operations

```rust
use stringtape::StringTape32;

let mut tape = StringTape32::new();
tape.push("hello")?;                    // Append one string
tape.extend(["world", "foo"])?;         // Append an array
assert_eq!(&tape[0], "hello");          // Direct indexing
assert_eq!(tape.get(1), Some("world")); // Safe access

for s in &tape { // Iterate
    println!("{}", s);
}

// Construct from an iterator
let tape2: StringTape32 = ["a", "b", "c"].into_iter().collect();
```

### Views and Slicing

```rust
let view = tape.view();              // View entire tape
let subview = tape.subview(1, 3)?;   // Items [1, 3)
let nested = subview.subview(0, 1)?; // Nested subviews
let raw_bytes = &tape.view()[1..3];  // Raw byte slice

// Views have same API as tapes
assert_eq!(subview.len(), 2);
assert_eq!(&subview[0], "world");
```

### Memory Management

```rust
// Pre-allocate capacity
let tape = StringTape32::with_capacity(1024, 100)?; // 1KB data, 100 strings

// Monitor usage
println!("Items: {}, Data: {} bytes", tape.len(), tape.data_len());

// Modify
tape.clear();           // Remove all items
tape.truncate(5);       // Keep first 5 items

// Custom allocators
use allocator_api2::alloc::Global;
let tape = StringTape::new_in(Global);
```

### Apache Arrow Interop

True zero-copy conversion to/from Arrow arrays:

```rust
// StringTape → Arrow (zero-copy)
let (data_slice, offsets_slice) = tape.arrow_slices();
let data_buffer = Buffer::from_slice_ref(data_slice);
let offsets_buffer = OffsetBuffer::new(ScalarBuffer::new(
    Buffer::from_slice_ref(offsets_slice), 0, offsets_slice.len()
));
let arrow_array = StringArray::new(offsets_buffer, data_buffer, None);

// Arrow → StringTapeView (zero-copy)
let view = unsafe {
    StringTapeView32::from_raw_parts(
        arrow_array.values(),
        arrow_array.offsets().as_ref(),
    )
};
```

## `no_std` Support

StringTape can be used in `no_std` environments:

```toml
[dependencies]
stringtape = { version = "0.1", default-features = false }
```

In `no_std` mode:

- All functionality is preserved
- Requires `alloc` for dynamic allocation
- Error types implement `Display` but not `std::error::Error`

## Testing

Run tests for both `std` and `no_std` configurations:

```bash
cargo test                          # Test with std (default)
cargo test --doc                    # Test documentation examples
cargo test --no-default-features    # Test without std
cargo test --all-features           # Test with all features enabled
```

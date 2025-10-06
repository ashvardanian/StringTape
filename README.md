# StringTape

![StringTape banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/StringTape.png?raw=true)

Memory-efficient collection classes for variable-length strings, co-located on a contiguous "tape".

- Convertible to __[Apache Arrow](https://arrow.apache.org/)__ `String`/`LargeString` & `Binary`/`LargeBinary` arrays
- Compatible with __UTF-8 & binary__ strings in Rust via `CharsTape` and `BytesTape`
- Usable in `no_std` and with custom allocators for GPU & embedded use cases
- Sliceable into __zero-copy__ borrow-checked views with `[i..n]` range syntax

Why?

```rust
let doc = fs::read_to_string("enwik9.txt")?;                                // 1.0 GB
let _ = Vec::<String>::from_iter(doc.split_whitespace());                   // + 7.1 GB
let _ = CharsTapeAuto::from_iter(doc.split_whitespace());                   // + 1.3 GB
let _ = Vec::<&[u8]>::from_iter(doc.split_whitespace().map(str::as_bytes)); // + 1.9 GB
let _ = CharsSlicesAuto::from_iter_and_data(                                // + 0.7 GB
    doc.split_whitespace(),
    Cow::Borrowed(doc.as_bytes()),
);
```

Tape classes copy data into contiguous buffers for cache-friendly iteration.
Slices classes reference existing data without copies.

## Quick Start

```rust
use stringtape::{CharsTapeI32, BytesTapeI32, StringTapeError};

// Create a new CharsTape with 32-bit offsets
let mut tape = CharsTapeI32::new();
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
let tape2: CharsTapeI32 = ["a", "b", "c"].into_iter().collect();
assert_eq!(tape2.len(), 3);

// Binary data with BytesTape
let mut bytes = BytesTapeI32::new();
bytes.push(b"hi")?;
assert_eq!(&bytes[0], b"hi");

# Ok::<(), StringTapeError>(())
```

## Memory Layout

`CharsTape` and `BytesTape` use the same memory layout as Apache Arrow string and binary arrays:

```text
Data buffer:    [h,e,l,l,o,w,o,r,l,d]
Offset buffer:  [0, 5, 10]
```

## API Overview

### Basic Operations

```rust
use stringtape::CharsTapeI32;

let mut tape = CharsTapeI32::new();
tape.push("hello")?;                    // Append one string
tape.extend(["world", "foo"])?;         // Append an array
assert_eq!(&tape[0], "hello");          // Direct indexing
assert_eq!(tape.get(1), Some("world")); // Safe access

for s in &tape { // Iterate
    println!("{}", s);
}

// Construct from an iterator
let tape2: CharsTapeI32 = ["a", "b", "c"].into_iter().collect();
```

`BytesTape` provides the same interface for arbitrary byte slices.

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
let tape = CharsTapeI32::with_capacity(1024, 100)?; // 1KB data, 100 strings

// Monitor usage
println!("Items: {}, Data: {} bytes", tape.len(), tape.data_len());

// Modify
tape.clear();           // Remove all items
tape.truncate(5);       // Keep first 5 items

// Custom allocators
use allocator_api2::alloc::Global;
let tape = CharsTape::new_in(Global);
```

### Apache Arrow Interop

True zero-copy conversion to/from Arrow arrays:

```rust
// CharsTape → Arrow (zero-copy)
let (data_slice, offsets_slice) = tape.arrow_slices();
let data_buffer = Buffer::from_slice_ref(data_slice);
let offsets_buffer = OffsetBuffer::new(ScalarBuffer::new(
    Buffer::from_slice_ref(offsets_slice), 0, offsets_slice.len()
));
let arrow_array = StringArray::new(offsets_buffer, data_buffer, None);

// Arrow → CharsTapeView (zero-copy)
let view = unsafe {
    CharsTapeViewI32::from_raw_parts(
        arrow_array.values(),
        arrow_array.offsets().as_ref(),
    )
};
```

`BytesTape` works the same way with Arrow `BinaryArray`/`LargeBinaryArray` types.

### Unsigned Offsets

In addition to the signed offsets (`i32`/`i64` via `CharsTapeI32`/`CharsTapeI64`),
the library also supports unsigned offsets (`u32`/`u64`) when you prefer non-negative indexing:

- `CharsTapeU32`, `CharsTapeU64`
- `BytesTapeU32`, `BytesTapeU64`
- `CharsTapeViewU32<'_>`, `CharsTapeViewU64<'_>`
- `BytesTapeViewU32<'_>`, `BytesTapeViewU64<'_>`

Note, that unsigned offsets cannot be converted to/from Arrow arrays.

## `no_std` Support

StringTape can be used in `no_std` environments:

```toml
[dependencies]
stringtape = { version = "2", default-features = false }
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
cargo clippy --lib -- -D warnings   # Lint the library code
cargo fmt --all -- --check          # Check code formatting
```

To reproduce memory usage numbers mentioned above, run:

```bash
/usr/bin/time -f "Vec<String>: %M KB | %E" cargo run --release --quiet --bin bench_vec_string -- enwik9.txt
/usr/bin/time -f "Vec<&[u8]>: %M KB | %E" cargo run --release --quiet --bin bench_vec_slice -- enwik9.txt
/usr/bin/time -f "CharsSlicesAuto: %M KB | %E" cargo run --release --quiet --bin bench_chars_slices -- enwik9.txt
/usr/bin/time -f "CharsTapeAuto: %M KB | %E" cargo run --release --quiet --bin bench_chars_tape -- enwik9.txt
```

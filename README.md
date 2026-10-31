# StringTape

![StringTape banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/StringTape.png?raw=true)

Memory-efficient collection classes for variable-length strings, co-located on contiguous "tapes".

- Convertible to __[Apache Arrow](https://arrow.apache.org/)__ `String`/`LargeString` & `Binary`/`LargeBinary` arrays
- Compatible with __UTF-8 & binary__ strings in Rust via `CharsTape` and `BytesTape`
- Usable in `no_std` and with custom __allocators__ for GPU & embedded use cases
- Sliceable into __zero-copy__ borrow-checked views with `subview(i..n)`

__Why not use `Vec<String>`?__
The first reason is about __RAM__!
In addition to storing 3 pointers for every `String` instance, your underlying memory allocator may store 2-4 pointers per heap entry.
Assuming most English words are under 8 bytes long, storing an extra 6 pointers on a 64-bit machine results in a 7x memory usage amplification:

```rust,ignore
let doc = fs::read_to_string("enwik9.txt")?;    // 1.0 GB
let words = doc.split_whitespace();             // ~ 160 M words
let buffers = words.map(str::as_bytes);

let _ = Vec::<String>::from_iter(words);        // + 7.1 GB copied ❌
CharsTapeAuto::new().extend(words)?;            // + 1.3 GB copied ✅
let _ = Vec::<&[u8]>::from_iter(buffers);       // + 1.9 GB copy-less ⚠️
let _ = BytesCowsAuto::from_iter_and_data(      // + 0.7 GB copy-less ✅
    buffers,
    Cow::Borrowed(doc.as_bytes()),
);
```

> "Tape" classes copy data into contiguous buffers for cache-friendly iteration.
> "Cows" classes reference existing data without copies.

The second reason is about __NUMA & GPUs__!
When designing high-performance systems, memory affinity is crucial.
On 2-socket servers, developers should avoid accessing memory allocated on the other socket.
On GPU-equipped servers, data should be transferred to GPU memory in large chunks to hide PCIe and software latency.
With individual `String` instances scattered around the heap, this is impossible.
With tapes and cows, this is easy: pass any allocator implementing [`core::alloc::Allocator`](https://doc.rust-lang.org/core/alloc/trait.Allocator.html).

## Quick Start

```rust
use stringtape::{CharsTapeI32, CharsTapeAuto, CharsCowsAuto, StringTapeError};
use std::borrow::Cow;

// Create a new CharsTape with 32-bit offsets
let mut tape = CharsTapeI32::new();
tape.push("hello")?;
tape.push("world")?;

assert_eq!(tape.len(), 2);
assert_eq!(tape.get(1), Some("world"));
assert_eq!(tape.get(2), None);

// Iterate over strings
for s in &tape {
    println!("{}", s);
}

// Append many strings; the offsets widen as the data grows
let mut tape2 = CharsTapeAuto::new();
tape2.extend(["a", "b", "c"])?;
assert_eq!(tape2.len(), 3);

// Zero-copy slices referencing existing data
let data = "hello world";
let cows = CharsCowsAuto::from_iter_and_data(
    data.split_whitespace(),
    Cow::Borrowed(data.as_bytes()),
)?;
assert_eq!(cows.len(), 2);
# Ok::<(), StringTapeError>(())
```

## Memory Layout

`CharsTape` and `BytesTape` use the same memory layout as Apache Arrow string and binary arrays:

```text
Data buffer:    [h,e,l,l,o,w,o,r,l,d]
Offset buffer:  [0, 5, 10]
```

`CharsCows` and `BytesCows` use packed `(offset, length)` entries to reference substrings in shared data:

```text
Shared data:    [h,e,l,l,o, ,w,o,r,l,d, ,f,o,o]
Entries:        [(0, 5), (6, 5), (12, 3)]  // "hello", "world", "foo"
```

The `#[repr(C, packed(1))]` layout eliminates padding between offset and length fields.
Similar to [SoA vs AoS](https://en.wikipedia.org/wiki/AoS_and_SoA), there is a trade-off.
That said, in extremely high-throughput applications, like hashing billions of very short strings, the runtime may be dominated by unaligned memory accesses and the additional level of indirection.
Consider switching to `CharsTape` or `BytesTape` in such cases.

## API Overview

### Basic Operations

```rust
use stringtape::{CharsTapeI32, StringTapeError};

let mut tape = CharsTapeI32::new();
tape.push("hello")?;                    // Append one string
tape.extend(["world", "foo"])?;         // Append an array
assert_eq!(tape.get(1), Some("world")); // Access by index
assert!(tape.contains("hello"));        // Membership test

for s in &tape {                        // Forward iteration
    println!("{}", s);
}
for s in tape.iter().rev() {            // Reverse iteration
    println!("{}", s);
}

// Construct from any iterator
let mut tape2 = CharsTapeI32::new();
tape2.extend(["a", "b", "c"])?;

// Compare in-place
assert!(tape > tape2);                  // Lexicographic comparison, "hello" > "a"

// Utility methods
tape.first();                           // First element
tape.last();                            // Last element
assert_eq!(tape.pop(), Some("foo"));    // Remove and return the last
tape.shrink_to_fit()?;                  // Reduce memory usage
# Ok::<(), StringTapeError>(())
```

`BytesTape` provides the same interface over `&[u8]` instead of `&str`.

Tapes do not sort.
Reordering a tape would move the payload, which is exactly what the layout exists to avoid.
Convert to a cows collection instead — it reorders compact `(offset, length)` entries and leaves the bytes untouched:

```rust
use stringtape::{CharsTapeI32, CharsCows, Reorderable, StringTapeError};

let mut tape = CharsTapeI32::new();
tape.extend(["zebra", "apple", "banana"])?;

let mut cows = CharsCows::<i32, u16>::try_from(&tape)?;
cows.sort();
cows.sort_by(|first, second| first.len().cmp(&second.len()));

// An external SIMD argument sort can compute the order instead
cows.permute(&[2, 0, 1])?;
# Ok::<(), StringTapeError>(())
```

`CharsCows`, `BytesCows`, `CharsCowsAuto` and `BytesCowsAuto` all implement the `Reorderable` trait, which is what keeps their reordering surfaces identical.

### Views and Slicing

```rust
use stringtape::{CharsTapeI32, StringTapeError};

let mut tape = CharsTapeI32::new();
tape.extend(["hello", "world", "foo"])?;

let spanned = &tape[1..3];                      // Text of items [1, 3) in one slice
assert_eq!(spanned, "worldfoo");

let view = tape.view();                         // View entire tape; views are `Copy`
assert!(view.subview(2..9).is_none());          // Checked slicing: out of range is `None`

if let Some(subview) = view.subview(1..) {      // Items [1, 3); also `1..3`, `..=2`, `..`
    assert_eq!(subview.len(), 2);               // Views have same API as tapes
    assert_eq!(subview.get(0), Some("world"));
    assert_eq!(&subview[..], "worldfoo");       // Ranges index views too
}
# Ok::<(), StringTapeError>(())
```

### Memory Management

```rust
use stringtape::{CharsTape, CharsTapeI32, StringTapeError};

// Pre-allocate capacity
let mut tape = CharsTapeI32::with_capacity(1024, 100)?; // 1KB data, 100 strings
assert!(tape.capacity() >= 100);

// Monitor usage
println!("Items: {}, Data: {} bytes", tape.len(), tape.data_len());

// Modify
tape.truncate(5);           // Keep first 5 items
tape.clear();               // Remove all items
tape.reserve(4096, 64)?;    // Size up for the next batch in one step

// Custom allocators
use std::alloc::Global;
let in_global: CharsTapeI32 = CharsTape::new_in(Global);
# Ok::<(), StringTapeError>(())
```

### Apache Arrow Interop

True zero-copy conversion to/from Arrow arrays:

```rust,ignore
// CharsTape → Arrow (zero-copy)
let data_buffer = Buffer::from_slice_ref(tape.data());
let offsets_buffer = OffsetBuffer::new(ScalarBuffer::new(
    Buffer::from_slice_ref(tape.offsets()), 0, tape.offsets().len()
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

### Auto Type Selection

Auto variants automatically select the most memory-efficient types based on data size:

```rust
use stringtape::{CharsTapeAuto, CharsCowsAuto, StringTapeError};
use std::borrow::Cow;

let data = "hello world";
let strings = data.split_whitespace();

// CharsTapeAuto: starts with I32 offsets, widening to U32 and U64 as data grows
let mut tape = CharsTapeAuto::new();
tape.extend(strings.clone())?;
assert_eq!(tape.type_name(), "CharsTape<i32>");

// CharsCowsAuto: selects offset (U32/U64) and length (U8/U16/U32) types
let cows = CharsCowsAuto::from_iter_and_data(strings, Cow::Borrowed(data.as_bytes()))?;
assert_eq!(cows.type_name(), "CharsCows<u32, u8>");
# Ok::<(), StringTapeError>(())
```

Available: `CharsTapeAuto`, `BytesTapeAuto`, `CharsCowsAuto`, `BytesCowsAuto`.

### Unsigned Offsets

Unsigned offsets (`u32`/`u64`) are available via `CharsTapeU32`, `CharsTapeU64`, `BytesTapeU16`, `BytesTapeU32`, `BytesTapeU64` and corresponding views.
These cannot be converted to/from Arrow arrays.

### Standard Traits

Tapes, views and cows implement `Debug`, `PartialEq`, `Eq`, `PartialOrd`, `Ord`, `Hash`, `IntoIterator`, `Send` and `Sync`.
Tapes also implement `Default`, views are `Clone` and `Copy`, and every iterator is an `ExactSizeIterator` and a `DoubleEndedIterator`.

```rust
use stringtape::{CharsTapeI32, StringTapeError};
use std::collections::HashMap;

let mut tape1 = CharsTapeI32::new();
tape1.push("hello")?;

let mut tape2 = CharsTapeI32::new();    // Copy explicitly, as copies can fail
tape2.extend(&tape1)?;
assert_eq!(tape1, tape2);               // PartialEq, Eq
assert!(tape1 <= tape2);                // PartialOrd, Ord

// Use in HashMap/HashSet
let mut map = HashMap::new();
map.insert(tape1, 42);                  // Hash

// Reverse iteration
for s in tape2.iter().rev() {           // DoubleEndedIterator
    println!("{}", s);
}
# Ok::<(), StringTapeError>(())
```

## Error Handling

No method panics, aborts on allocation failure, or silently drops an item.
Range indexing is the one exception and follows `str`: `&tape[1..3]` panics on an inverted or out-of-bounds range, while `subview` returns `None`.
Tapes and cows implement neither `FromIterator`, `Extend` nor `Clone`, because those signatures can't report a failure.
A read that can miss returns `Option`, so `get` past the end or `subview` outside the tape is `None`.
Anything that allocates, validates UTF-8, or narrows a size into the `Offset` or `Length` type returns `Result<_, StringTapeError>`, and the variant says which of those failed.
A refused `push` leaves the tape unchanged, and a refused `extend` keeps the items before the one that failed.
`pop` returns the string it removed rather than a success flag.
Two exceptions remain:

- The stable `sort_by` and `sort_by_key` on cows use the standard library's stable sort, whose scratch buffer aborts if it cannot be allocated; `sort` is unstable, in place and allocation-free.
- `StringTapeError` implements `Display`, so the standard `to_string` aborts if the global heap is exhausted.

## `no_std` Support

StringTape can be used in `no_std` environments:

```toml
[dependencies]
stringtape = { version = "3", default-features = false }
```

In `no_std` mode:

- All functionality is preserved
- Requires `alloc` for dynamic allocation
- `StringTapeError` still implements `core::error::Error`

## Testing

Run tests for both `std` and `no_std` configurations, on a nightly toolchain until Rust 1.100 is stable:

```bash
cargo test                                                              # Test with std, README included
cargo test --no-default-features                                        # Test without std
cargo build --lib --no-default-features --target thumbv7em-none-eabihf  # Build for bare metal
cargo miri test --lib                                                   # Check unsafe code under Miri
cargo clippy --all-targets -- -D warnings                               # Lint every target
cargo fmt --check                                                       # Check code formatting
```

To reproduce memory usage numbers mentioned above, pull the dataset for the ["Large Text Compression Benchmark"](https://mattmahoney.net/dc/textdata.html), and run:

```bash
/usr/bin/time -f "Vec<String>: %M KB | %E" cargo run --release --quiet --bin bench_vec_string -- enwik9.txt
/usr/bin/time -f "Vec<&[u8]>: %M KB | %E" cargo run --release --quiet --bin bench_vec_slice -- enwik9.txt
/usr/bin/time -f "CharsCowsAuto: %M KB | %E" cargo run --release --quiet --bin bench_chars_slices -- enwik9.txt
/usr/bin/time -f "CharsTapeAuto: %M KB | %E" cargo run --release --quiet --bin bench_chars_tape -- enwik9.txt
```

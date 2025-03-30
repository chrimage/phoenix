# sqlite-vec in Python

## Installation

Install the `sqlite-vec` package using pip:

```bash
pip install sqlite-vec
```

## Basic Usage

```python
import sqlite3
import sqlite_vec

# Connect to a database and load sqlite-vec extension
db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

# Check version
vec_version, = db.execute("select vec_version()").fetchone()
print(f"vec_version={vec_version}")
```

## Working with Vectors

### Lists of Floats

Convert Python lists to compact BLOB format using `serialize_float32()`:

```python
from sqlite_vec import serialize_float32

embedding = [0.1, 0.2, 0.3, 0.4]
result = db.execute('select vec_length(?)', [serialize_float32(embedding)])
print(result.fetchone()[0])  # 4
```

### NumPy Arrays

Use NumPy arrays directly, ensuring they are cast to 32-bit floats:

```python
import numpy as np
embedding = np.array([0.1, 0.2, 0.3, 0.4])
db.execute(
    "SELECT vec_length(?)", [embedding.astype(np.float32)]
)
```

## SQLite Version Considerations

- Recommended: SQLite version 3.41 or higher
- Check current version: `python -c 'import sqlite3; print(sqlite3.sqlite_version)'`

### Upgrading SQLite

Options include:
- Compiling your own SQLite version
- Using `pysqlite3` package
- Upgrading Python version

## MacOS Specific Note

MacOS blocks SQLite extensions by default. If you encounter an `AttributeError`, use Homebrew's Python version or explore alternative workarounds.
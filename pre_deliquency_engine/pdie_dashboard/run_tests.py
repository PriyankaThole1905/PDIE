"""Windows-safe test runner with UTF-8 encoding."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Now run the tests
exec(open('test_engine.py', encoding='utf-8').read())

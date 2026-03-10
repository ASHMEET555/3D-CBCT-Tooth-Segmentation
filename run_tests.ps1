# Run tests with proper environment settings to avoid OpenBLAS memory errors
$env:OPENBLAS_NUM_THREADS=1
$env:MKL_NUM_THREADS=1
$env:NUMBA_DISABLE_JIT=1

& ".\.venv\Scripts\python.exe" -m pytest tests/test_preprocessing.py -v @args

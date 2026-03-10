@echo off
REM Run tests with proper environment settings to avoid OpenBLAS memory errors
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMBA_DISABLE_JIT=1

python -m pytest tests/test_preprocessing.py -v %*

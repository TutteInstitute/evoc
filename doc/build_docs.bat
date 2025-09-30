@echo off
REM Documentation build script for EVoC (Windows)

echo Building EVoC Documentation
echo ==========================

REM Check if we're in the right directory
if not exist "source\conf.py" (
    echo Error: Run this script from the doc directory
    exit /b 1
)

REM Check if virtual environment exists, create if needed
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements
echo Installing documentation requirements...
pip install -r requirements.txt

REM Install EVoC in development mode
echo Installing EVoC in development mode...
pip install -e ..\..

REM Clean previous build
echo Cleaning previous build...
make clean

REM Build HTML documentation
echo Building HTML documentation...
make html

if %ERRORLEVEL% equ 0 (
    echo Documentation built successfully!
    echo Open build\html\index.html in your browser to view
) else (
    echo Build failed with errors
    exit /b 1
)

echo Build complete!

@echo off
REM Check if requirements.txt exists
IF NOT EXIST requirements.txt (
    echo requirements.txt not found!
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Check if the installation was successful
IF ERRORLEVEL 1 (
    echo Failed to install dependencies.
    exit /b 1
)

REM Run the Flask application
echo Running the application...
python -m src.main

REM Pause the command window to see output
pause

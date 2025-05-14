@echo off
echo Starting MedAssyst Backend...
echo.
echo Make sure Ollama is running with the Mykes/medicus model installed (hf.co/Mykes/medicus:Q4_K_M)
echo.

python -m uvicorn main:app --reload

echo.
echo Backend server stopped.

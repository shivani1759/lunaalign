@echo off
title LunaAlign Interactive Web Dashboard
echo ========================================================
echo Starting LunaAlign Lunar Image Registration Dashboard...
echo ========================================================
echo.
echo Opening browser at http://localhost:8080...
start http://localhost:8080
python app.py --port 8080
pause

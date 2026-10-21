@echo off
chcp 65001 >nul
echo ========================================
echo   训练工具独立模式启动
echo ========================================
echo.

REM 检查 Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请确保 Python 已安装并加入 PATH
    pause
    exit /b 1
)

REM 启动
python standalone_training.py %*

pause

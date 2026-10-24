@echo off
rem Start the lab on Windows: lab   (Ctrl+C stops it; run lab again to restart)
rem   lab mcp    runs the MCP server an agent (Claude Code, Codex, ...) connects to
rem Same as ./lab on Mac and Linux.
cd /d "%~dp0"
if /i "%~1"=="mcp" (
  ".venv\Scripts\python.exe" -m image_to_3dlab.mcp_server %2 %3 %4 %5 %6 %7 %8 %9
  exit /b
)
".venv\Scripts\python.exe" viewer\serve.py --auto %*

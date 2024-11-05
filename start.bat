@echo off
set PORT=8080
set HOST=0.0.0.0

uvicorn main:app --host %HOST% --port %PORT% --forwarded-allow-ips '*'
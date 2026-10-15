@echo off
rem Builds uiprobe.exe with MSVC, or with g++ (MinGW) when MSVC is not installed:
rem   build.bat [include directory of the Vulkan headers]
setlocal
set VULKAN_HEADERS=%~1
if "%VULKAN_HEADERS%"=="" set VULKAN_HEADERS=%TEMP%\vmaf-vulkan-build\Vulkan-Headers\include
cd /d %~dp0
set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
if exist "%VSWHERE%" (
  for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSDIR=%%i
)
if defined VSDIR (
  call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
  cl /nologo /O2 /EHsc /I "%VULKAN_HEADERS%" uiprobe.cpp /Fe:uiprobe.exe || exit /b 1
) else (
  g++ -O2 -static -I "%VULKAN_HEADERS%" uiprobe.cpp -o uiprobe.exe || exit /b 1
)

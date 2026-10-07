@echo off
rem Builds vvdec and vvdecapp with the reconstruction on the GPU (Vulkan): MSVC,
rem Ninja, Release, into build\vulkan; the programs land in bin\release-static.
rem
rem   scripts\build_vulkan.bat [slangc.exe] [include directory of the Vulkan headers]
rem
rem The defaults are the Slang compiler and Vulkan headers that libvmaf-fast's
rem fast\scripts\build_vmaf_vulkan.ps1 keeps in %TEMP%\vmaf-vulkan-build.
setlocal
set ROOT=%~dp0..
set SLANGC=%~1
if "%SLANGC%"=="" set SLANGC=%TEMP%\vmaf-vulkan-build\slang-2026.10.2\bin\slangc.exe
set VULKAN_HEADERS=%~2
if "%VULKAN_HEADERS%"=="" set VULKAN_HEADERS=%TEMP%\vmaf-vulkan-build\Vulkan-Headers\include

set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSDIR=%%i
if "%VSDIR%"=="" (
  echo Visual Studio with the C++ tools was not found
  exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

cmake -S "%ROOT%" -B "%ROOT%\build\vulkan" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVVDEC_ENABLE_VULKAN=ON ^
      "-DVVDEC_SLANGC=%SLANGC%" "-DVVDEC_VULKAN_HEADERS=%VULKAN_HEADERS%" || exit /b 1
cmake --build "%ROOT%\build\vulkan" || exit /b 1

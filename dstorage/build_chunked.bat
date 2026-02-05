@echo off
setlocal

set CL_EXE=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe
set LINK_EXE=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\link.exe
set VS_INC=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\include
set VS_LIB=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\lib\x64
set DS_INC=C:\Users\danie\Documents\treasure\UsersdanieDocumentsDirectStorage\Samples\packages\Microsoft.Direct3D.DirectStorage.1.3.0\native\include
set SDK_UM=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um
set SDK_SHARED=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared
set SDK_UCRT=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt
set SDK_WINRT=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\winrt
set SDK_LIB_UM=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64
set SDK_LIB_UCRT=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64

echo === Step 1: Compile C++ ===
"%CL_EXE%" /c /O2 /EHsc /std:c++17 /DWIN32_LEAN_AND_MEAN /DNOMINMAX /DDSTORAGE_EXPORTS /I"%VS_INC%" /I"%DS_INC%" /I"%SDK_UM%" /I"%SDK_SHARED%" /I"%SDK_UCRT%" /I"%SDK_WINRT%" /Fo:"native\dstorage_loader.obj" "native\dstorage_loader.cpp"
if errorlevel 1 (echo COMPILE FAILED & exit /b 1)
echo COMPILE OK

echo === Step 2: Link DLL ===
"%LINK_EXE%" /DLL /OUT:"dstorage_loader.dll" /IMPLIB:"native\dstorage_loader.lib" /LIBPATH:"%VS_LIB%" /LIBPATH:"%SDK_LIB_UM%" /LIBPATH:"%SDK_LIB_UCRT%" dxgi.lib d3d12.lib ole32.lib kernel32.lib ucrt.lib msvcrt.lib "native\dstorage_loader.obj"
if errorlevel 1 (echo LINK FAILED & exit /b 1)
echo LINK OK

echo === Step 3: Build Go ===
go build -v ./...
if errorlevel 1 (echo GO BUILD FAILED & exit /b 1)
echo GO BUILD OK

echo === Step 4: Run Tests ===
go test -v -count=1
echo === DONE ===

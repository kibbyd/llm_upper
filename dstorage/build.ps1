# PowerShell build script for Ollama DirectStorage
# Run in PowerShell: .\build.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Ollama DirectStorage Build Script"
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# All paths
$clExe = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe"
$linkExe = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\link.exe"
$vsInc = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\include"
$vsLib = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\lib\x64"
$dsInc = "C:\Users\danie\Documents\treasure\UsersdanieDocumentsDirectStorage\Samples\packages\Microsoft.Direct3D.DirectStorage.1.3.0\native\include"
$dsLib = "C:\Users\danie\Documents\treasure\UsersdanieDocumentsDirectStorage\Samples\packages\Microsoft.Direct3D.DirectStorage.1.3.0\native\lib\x64"
$sdkUm = "C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um"
$sdkShared = "C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared"
$sdkUcrt = "C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt"
$sdkWinrt = "C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\winrt"
$sdkLibUm = "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64"
$sdkLibUcrt = "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64"

$buildDir = "C:\Users\danie\Documents\ollama\ml\backend\ggml\dstorage"
$cppFile = "$buildDir\native\dstorage_loader.cpp"
$objFile = "$buildDir\native\dstorage_loader.obj"
$dllFile = "$buildDir\dstorage_loader.dll"
$libFile = "$buildDir\native\dstorage_loader.lib"

# Step 1: Compile C++ to .obj
Write-Host "Step 1: Compiling C++..." -ForegroundColor Yellow

$compileArgs = @(
    "/c",
    "/O2",
    "/EHsc",
    "/std:c++17",
    "/DWIN32_LEAN_AND_MEAN",
    "/DNOMINMAX",
    "/DDSTORAGE_EXPORTS",
    "/I`"$vsInc`"",
    "/I`"$dsInc`"",
    "/I`"$sdkUm`"",
    "/I`"$sdkShared`"",
    "/I`"$sdkUcrt`"",
    "/I`"$sdkWinrt`"",
    "/Fo:`"$objFile`"",
    "`"$cppFile`""
)

$process = Start-Process -FilePath $clExe -ArgumentList $compileArgs -NoNewWindow -Wait -PassThru -RedirectStandardOutput "build_stdout.txt" -RedirectStandardError "build_stderr.txt"
Get-Content "build_stdout.txt"
Get-Content "build_stderr.txt"

if ($process.ExitCode -ne 0) {
    Write-Host "C++ compilation FAILED" -ForegroundColor Red
    exit 1
}

Write-Host "C++ compiled OK" -ForegroundColor Green
Write-Host ""

# Step 2: Link to DLL
Write-Host "Step 2: Linking DLL..." -ForegroundColor Yellow

$linkArgs = @(
    "/DLL",
    "/OUT:`"$dllFile`"",
    "/IMPLIB:`"$libFile`"",
    "/LIBPATH:`"$vsLib`"",
    "/LIBPATH:`"$sdkLibUm`"",
    "/LIBPATH:`"$sdkLibUcrt`"",
    "dxgi.lib",
    "d3d12.lib",
    "ole32.lib",
    "kernel32.lib",
    "ucrt.lib",
    "msvcrt.lib",
    "`"$objFile`""
)

$process = Start-Process -FilePath $linkExe -ArgumentList $linkArgs -NoNewWindow -Wait -PassThru -RedirectStandardOutput "link_stdout.txt" -RedirectStandardError "link_stderr.txt"
Get-Content "link_stdout.txt"
Get-Content "link_stderr.txt"

if ($process.ExitCode -ne 0) {
    Write-Host "Linking FAILED" -ForegroundColor Red
    exit 1
}

Write-Host "DLL linked OK" -ForegroundColor Green
Write-Host ""

# Step 3: Build Go with the DLL
Write-Host "Step 3: Building Go..." -ForegroundColor Yellow

Set-Location $buildDir
go build -v 2>&1

Write-Host ""

# Step 4: Run tests
Write-Host "Step 4: Running tests..." -ForegroundColor Yellow
go test -v 2>&1

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "BUILD COMPLETE" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Files created:"
Write-Host "  $objFile"
Write-Host "  $dllFile"
Write-Host "  $libFile"
Write-Host ""
Write-Host "Press any key..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

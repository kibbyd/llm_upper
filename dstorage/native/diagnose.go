// DirectStorage diagnostic - checks exactly what's failing
// Run: go run native/diagnose.go

//go:build ignore

package main

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

func main() {
	fmt.Println("=== DirectStorage Diagnostic ===")
	fmt.Println()

	// Check Windows build
	ntdll := syscall.MustLoadDLL("ntdll.dll")
	rtlGetVersion := ntdll.MustFindProc("RtlGetVersion")
	type OSVersionInfo struct {
		Size      uint32
		Major     uint32
		Minor     uint32
		Build     uint32
		Platform  uint32
		CSDString [128]uint16
	}
	var ver OSVersionInfo
	ver.Size = uint32(unsafe.Sizeof(ver))
	rtlGetVersion.Call(uintptr(unsafe.Pointer(&ver)))
	fmt.Printf("Windows: %d.%d.%d\n", ver.Major, ver.Minor, ver.Build)
	if ver.Build < 22000 {
		fmt.Println("ERROR: Need Windows 11 (build 22000+)")
	} else {
		fmt.Println("OK: Windows 11+")
	}
	fmt.Println()

	// Check D3D12
	fmt.Println("--- D3D12 ---")
	d3d12, err := syscall.LoadDLL("d3d12.dll")
	if err != nil {
		fmt.Println("ERROR: d3d12.dll not found:", err)
	} else {
		fmt.Println("OK: d3d12.dll loaded")
		d3d12.Release()
	}

	// Check DXGI
	dxgi, err := syscall.LoadDLL("dxgi.dll")
	if err != nil {
		fmt.Println("ERROR: dxgi.dll not found:", err)
	} else {
		fmt.Println("OK: dxgi.dll loaded")
		dxgi.Release()
	}
	fmt.Println()

	// Check DirectStorage system DLL
	fmt.Println("--- DirectStorage ---")
	dsDLL, err := syscall.LoadDLL("dstorage.dll")
	if err != nil {
		fmt.Println("WARN: System dstorage.dll not found (trying local)")
	} else {
		fmt.Println("OK: System dstorage.dll loaded")
		dsDLL.Release()
	}

	// Check our local copy
	localDS := `C:\Users\danie\Documents\ollama\ml\backend\ggml\dstorage\dstorage.dll`
	if _, err := os.Stat(localDS); err == nil {
		fmt.Println("OK: Local dstorage.dll exists")
	} else {
		fmt.Println("ERROR: Local dstorage.dll missing")
	}

	localCore := `C:\Users\danie\Documents\ollama\ml\backend\ggml\dstorage\dstoragecore.dll`
	if _, err := os.Stat(localCore); err == nil {
		fmt.Println("OK: Local dstoragecore.dll exists")
	} else {
		fmt.Println("ERROR: Local dstoragecore.dll missing")
	}
	fmt.Println()

	// Try loading our DLL
	fmt.Println("--- Our DLL ---")
	loaderPath := `C:\Users\danie\Documents\ollama\ml\backend\ggml\dstorage\dstorage_loader.dll`
	loader, err := syscall.LoadDLL(loaderPath)
	if err != nil {
		fmt.Println("ERROR: Can't load dstorage_loader.dll:", err)
		return
	}
	fmt.Println("OK: dstorage_loader.dll loaded")

	procAvail, err := loader.FindProc("ds_loader_available")
	if err != nil {
		fmt.Println("ERROR: ds_loader_available not exported:", err)
		return
	}
	fmt.Println("OK: ds_loader_available found")

	ret, _, _ := procAvail.Call()
	fmt.Printf("\nds_loader_available() = %d\n", ret)

	// Get the HRESULT
	procHR, err := loader.FindProc("ds_loader_get_hresult")
	if err == nil {
		hr, _, _ := procHR.Call()
		fmt.Printf("Last HRESULT = 0x%08X\n", uint32(hr))
	}

	if ret == 1 {
		fmt.Println("RESULT: DirectStorage IS available!")

		// Try creating a loader
		procCreate, _ := loader.FindProc("ds_loader_create")
		handle, _, _ := procCreate.Call()
		if handle != 0 {
			fmt.Println("ds_loader_create() SUCCESS!")
			procDestroy, _ := loader.FindProc("ds_loader_destroy")
			procDestroy.Call(handle)
		} else {
			fmt.Println("ds_loader_create() FAILED")
			if procHR != nil {
				hr, _, _ := procHR.Call()
				fmt.Printf("Create HRESULT = 0x%08X\n", uint32(hr))
			}
		}
	} else {
		fmt.Println("RESULT: DirectStorage NOT available")
		fmt.Println()
		fmt.Println("The HRESULT above tells us exactly what failed.")
		fmt.Println("0x00000000 = D3D12 failed before DirectStorage was tried")
		fmt.Println("0x80070005 = Access denied (try Run as Administrator)")
		fmt.Println("0x80004002 = Interface not supported")
		fmt.Println("0x80004005 = General failure")
	}

	// Check NVIDIA
	fmt.Println()
	fmt.Println("--- GPU Info ---")
	fmt.Println("Run 'nvidia-smi' in a terminal to check GPU driver version")

	loader.Release()
}

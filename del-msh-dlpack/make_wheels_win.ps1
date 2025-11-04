
# ####################
# cleanup files

$root = Get-Location

Write-Host "Cleaning .pyd files and __pycache__ folders under $root ..." -ForegroundColor Cyan

# --- .pyd ファイルを削除 ---
Get-ChildItem -Path $root -Recurse -Include *.pyd -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
    Write-Host "Deleted file: $($_.FullName)" -ForegroundColor DarkGray
}

# --- __pycache__ フォルダを削除 ---
Get-ChildItem -Path $root -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Deleted folder: $($_.FullName)" -ForegroundColor DarkGray
}

Write-Host "Cleanup completed!" -ForegroundColor Green

# ################
# make dist folder

$dir = "..\target\wheels\dist"
if (-not (Test-Path $dir)) {
    mkdir $dir | Out-Null
}

# ###############

$cuvers = @(
    @("cu121", "v12.1"),
    @("cu123", "v12.3"),
    @("cu126", "v12.6")
)
$venvs = @("venv311", "venv312", "venv313")

$path_org = $env:PATH

foreach ($cu in $cuvers) {

    $cutag = $($cu[0])
    $env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$($cu[1])"
    $env:PATH = "$env:CUDA_HOME\bin;$env:CUDA_HOME\libnvvp;" + $path_org
    echo $env:PATH
    nvcc --version

    foreach ($venv in $venvs) {
        echo "---------------------------------------"
        # activate virtual env
        & ..\..\$venv\Scripts\Activate.ps1
        python --version
        python -m pip install -U pip maturin
        # compile
        & ..\..\$venv\Scripts\maturin.exe build --release --features cuda
        # rename
        $whl = Get-ChildItem ..\target\wheels\*.whl | Select-Object -First 1
        if (-not $whl) { throw "wheel not found" }
        echo $whl
        $newname = $whl.Name -replace '(^del_msh_dlpack-\d+\.\d+\.\d+)', "`$1+$cutag"
        Copy-Item $whl.FullName -Destination ("..\target\wheels\dist\" + $newname)
        del $whl
        deactivate
    }

}


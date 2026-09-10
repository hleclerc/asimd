# EVERY TEST, AT EVERY MSVC /arch: LEVEL.
#
# The Windows counterpart of `run_all_isa.sh`. MSVC does not take `-msse2` / `-mavx2`; on x64 it
# has exactly four levels, and the baseline one has no flag at all because SSE2 is architectural
# there.
#
#   pwsh tests/run_msvc.ps1                 # from a Developer PowerShell, or after vcvars64
#   pwsh tests/run_msvc.ps1 -RunAvx512      # only if this machine actually has AVX-512
#
# WHY /Zc:preprocessor. MSVC's default preprocessor is the traditional one, which mishandles
# __VA_ARGS__ expansion -- and `tests/check.h` is built on variadic macros, because a template
# argument list carries commas the preprocessor does not respect. The conforming preprocessor is
# opt-in and this is what opts in.
#
# AVX-512 IS COMPILED BUT NOT RUN by default: GitHub's hosted runners do not guarantee it, and a
# binary built with /arch:AVX512 faults on a machine without it. Compiling still exercises every
# AVX-512 backend, which is most of what we want from CI.
param( [switch]$RunAvx512 )

$ErrorActionPreference = "Stop"
$here  = Split-Path -Parent $MyInvocation.MyCommand.Path
$inc   = Join-Path $here "..\src"
$tmp   = Join-Path $env:TEMP ( "asimd_" + [guid]::NewGuid().ToString("N").Substring(0,8) )
New-Item -ItemType Directory -Path $tmp | Out-Null

$tests = @( "test_ops", "test_split", "test_selection", "test_x86_ops", "test_x86_dispatch" )
# label, extra flags, run it?
$levels = @(
    @{ name = "SSE2 (x64 baseline)"; flags = @();                 run = $true          },
    @{ name = "AVX";                 flags = @("/arch:AVX");      run = $true          },
    @{ name = "AVX2";                flags = @("/arch:AVX2");     run = $true          },
    @{ name = "AVX512";              flags = @("/arch:AVX512");   run = [bool]$RunAvx512 }
)

$common = @( "/std:c++20", "/O2", "/EHsc", "/permissive-", "/Zc:preprocessor", "/W3", "/nologo" )
$failed = $false

foreach ( $lvl in $levels ) {
    $suffix = if ( $lvl.run ) { "" } else { "   (compiled, not run)" }
    Write-Host "===================== $($lvl.name)$suffix ====================="
    foreach ( $t in $tests ) {
        $exe = Join-Path $tmp "$t.exe"
        $src = Join-Path $here "$t.cpp"
        $out = & cl @common @($lvl.flags) "/I$inc" "/I$here" $src "/Fe:$exe" "/Fo:$tmp\" 2>&1
        if ( $LASTEXITCODE -ne 0 ) {
            Write-Host "  BUILD FAILED  $t"
            $out | Select-String -Pattern "error" | Select-Object -First 5 | ForEach-Object { Write-Host "      $_" }
            $failed = $true
            continue
        }
        $warn = $out | Select-String -Pattern ": warning "
        if ( $warn ) {
            Write-Host "  warnings in ${t}:"
            $warn | Select-Object -First 5 | ForEach-Object { Write-Host "      $_" }
        }
        if ( $lvl.run ) {
            $res = & $exe
            if ( $LASTEXITCODE -ne 0 ) { $failed = $true }
            $res | Select-String -Pattern "^$t|FAIL|broken" | ForEach-Object { Write-Host "  $_" }
        }
    }
}

Remove-Item -Recurse -Force $tmp
if ( $failed ) { Write-Host "`nSOMETHING FAILED"; exit 1 }
Write-Host "`nall MSVC levels: ok"

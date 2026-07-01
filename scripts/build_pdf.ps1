$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$latexDir = Join-Path $root "docs\latex"
$buildDir = Join-Path $latexDir "build"
$outDir = Join-Path $root "output\pdf"

New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Push-Location $latexDir
try {
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory $buildDir breakpoint_whitepaper.tex
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory $buildDir breakpoint_whitepaper.tex
}
finally {
  Pop-Location
}

Copy-Item (Join-Path $buildDir "breakpoint_whitepaper.pdf") (Join-Path $outDir "breakpoint_system_design.pdf") -Force
Write-Host "Wrote output/pdf/breakpoint_system_design.pdf"

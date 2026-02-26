function fastCompile {
    $files = Get-ChildItem -Path "src_cpp" -Filter "*.cpp" | Where-Object { $_.Name -ne "binding.cpp" } | ForEach-Object { $_.FullName }
    g++ -O3 -march=native -std=c++17 -mavx2 -mfma $files -o test.exe -I src_cpp
    if ($?) { .\test.exe }
}

Write-Host "Create-React-App-like Init: fastCompile function ready!" -ForegroundColor Green
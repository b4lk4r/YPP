='Stop'
 = Get-Content -Raw 'laba_2/sample.ipynb' | ConvertFrom-Json
 = 0
foreach( in .cells){
  if(.cell_type -eq 'code'){
    Write-Output "\n# --- code cell  ---\n"
    (.source -join '') | Write-Output
  }
  ++
}

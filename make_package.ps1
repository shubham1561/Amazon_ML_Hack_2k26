# Builds <team_name>_submission.zip in the required structure:
#   output/{matching_results.tsv,candidate_pairs.tsv}
#   code/business_entity_resolution/{src/,README.md,requirements.txt}
#   Documentation_template.md
# Usage:  powershell -File make_package.ps1 -Team "MyTeam" [-OutputDir output_v2]
param([string]$Team = "team", [string]$OutputDir = "output")
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$stage = Join-Path $env:TEMP "er_package_$Team"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force "$stage\output", "$stage\code\business_entity_resolution\src" | Out-Null
Copy-Item "$root\$OutputDir\matching_results.tsv", "$root\$OutputDir\candidate_pairs.tsv" "$stage\output\"
Copy-Item "$root\code\business_entity_resolution\README.md", "$root\code\business_entity_resolution\requirements.txt" "$stage\code\business_entity_resolution\"
Get-ChildItem "$root\code\business_entity_resolution\src" -Filter *.py | Copy-Item -Destination "$stage\code\business_entity_resolution\src\"
Copy-Item "$root\Documentation_template.md" "$stage\"
$zip = Join-Path $root "$($Team)_submission.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path "$stage\*" -DestinationPath $zip -CompressionLevel Optimal
Get-Item $zip | Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,1)}}

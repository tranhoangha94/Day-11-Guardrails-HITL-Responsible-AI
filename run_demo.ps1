# VinBank Defense Pipeline — Chatbot Demo (Streamlit)
# Usage: .\run_demo.ps1

$env:GOOGLE_GENAI_USE_VERTEXAI = "0"

if (-not $env:GOOGLE_API_KEY) {
    $envFile = Join-Path $PSScriptRoot "src\.env"
    if (Test-Path $envFile) {
        Get-Content $envFile | ForEach-Object {
            if ($_ -match '^\s*GOOGLE_API_KEY\s*=\s*(.+)\s*$') {
                $env:GOOGLE_API_KEY = $Matches[1].Trim().Trim('"').Trim("'")
            }
        }
    }
    if (-not $env:GOOGLE_API_KEY) {
        $env:GOOGLE_API_KEY = Read-Host "Enter GOOGLE_API_KEY"
    }
}

Set-Location "$PSScriptRoot\src"
streamlit run ui/streamlit_app.py

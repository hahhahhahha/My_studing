$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = 'C:\Users\Administrator\qla_env_3135\Scripts\python.exe'
$registry = Join-Path $root 'protocol\E1_TEST_REGISTRY.json'
$review = Join-Path $root 'protocol\INDEPENDENT_ENTRY_REVIEW.json'
if (-not (Test-Path -LiteralPath $review)) {
    throw '独立入口复审收据缺失；不得启动任何正式 cell'
}
$reg = Get-Content -LiteralPath $registry -Raw | ConvertFrom-Json
if ($reg.formal_entry_approved -ne $true) {
    throw '正式入口尚未获独立复审批准；保留目录并停止'
}
$config = Join-Path $root 'protocol\FORMAL_CONFIG.json'
$manifest = Join-Path $root 'TEST_IMAGES.json'
$bundles = Join-Path $root 'protocol\bundles'
$cells = @($reg.grid.path | ForEach-Object { (Get-Content -LiteralPath $_ -Raw | ConvertFrom-Json).cells }) |
    Sort-Object id
foreach ($cell in $cells) {
    $schedule = Join-Path $root ('schedules\test_seed_' + $cell.seed + '.json')
    $bundle = if ($cell.method -eq 'qla') { Join-Path $bundles 'qla_actor_bundle.json' } else { Join-Path $bundles ($cell.method + '.json') }
    $out = Join-Path $root ('results\cell_' + $cell.id)
    if (Test-Path -LiteralPath $out) { throw "已有 cell 输出，禁止覆盖：$out" }
    & $py -m systemlab run-local --config $config --manifest $manifest --schedule $schedule `
        --method $cell.method --output $out --actor-bundle $bundle `
        --entry-review $review --test-registry $registry --cell-id $cell.id
    if ($LASTEXITCODE -ne 0) { throw "cell 失败，保留原目录并停止：$($cell.id)" }
    & $py -m systemlab analyze --client-dir (Join-Path $out 'client') `
        --controller-dir (Join-Path $out 'controller') --output (Join-Path $out 'ANALYSIS.json')
    if ($LASTEXITCODE -ne 0) { throw "分析失败，保留原目录并停止：$($cell.id)" }
}

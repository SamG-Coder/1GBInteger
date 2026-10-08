param(
    [string]$Executable = '.\temporal_parallel.exe',
    [string]$Output = 'temporal-results.csv',
    [int]$Steps = 1000,
    [int]$Repeats = 3,
    [int[]]$Atoms = @(1000,10000,100000,1000000),
    [int[]]$Threads = @(1,2,4,8,16)
)
$ErrorActionPreference = 'Stop'
$rows = @()
$expected = @{}
foreach ($count in $Atoms) {
    foreach ($workers in $Threads) {
        for ($repeat = 0; $repeat -lt $Repeats; $repeat++) {
            $p = New-Object System.Diagnostics.Process
            $p.StartInfo.FileName = (Resolve-Path $Executable).Path
            $p.StartInfo.Arguments = "--gb 1 --atoms $count --steps $Steps --interval 10 --threads $workers"
            $p.StartInfo.UseShellExecute = $false
            $p.StartInfo.CreateNoWindow = $true
            $p.StartInfo.RedirectStandardOutput = $true
            $watch = [Diagnostics.Stopwatch]::StartNew()
            [void]$p.Start()
            $peak = 0L
            while (!$p.HasExited) {
                $p.Refresh()
                $peak = [Math]::Max($peak, $p.PeakWorkingSet64)
                Start-Sleep -Milliseconds 5
            }
            $stdout = $p.StandardOutput.ReadToEnd()
            $p.WaitForExit()
            $watch.Stop()
            if ($p.ExitCode -ne 0) { throw "Benchmark failed: $stdout" }
            $values = @{}
            foreach ($match in [regex]::Matches($stdout, '(\w+)=(\S+)')) { $values[$match.Groups[1].Value] = $match.Groups[2].Value }
            $signature = "$($values.observations)/$($values.ones)/$($values.checksum)"
            if ($expected.ContainsKey($count) -and $expected[$count] -ne $signature) { throw 'Non-deterministic result' }
            $expected[$count] = $signature
            $cpu = $p.TotalProcessorTime.TotalSeconds
            $rows += [pscustomobject]@{
                atoms=$count; threads=$workers; steps=$Steps; repeat=$repeat
                seconds=[double]$values.seconds; observations_per_second=[double]$values.observations_per_second
                process_wall_seconds=$watch.Elapsed.TotalSeconds; cpu_seconds=$cpu
                cpu_percent_machine=100*$cpu/$watch.Elapsed.TotalSeconds/[Environment]::ProcessorCount
                sampled_peak_working_set_bytes=$peak; checksum=$values.checksum; ones=$values.ones
            }
            $p.Dispose()
        }
        Write-Host "atoms=$count threads=$workers complete"
        $rows | Export-Csv -NoTypeInformation $Output
    }
}

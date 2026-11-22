# Post-PiPLTool fix for BloomPiPL.rc.
#
# PiPLTool writes Pascal-style strings as "\xNN<text>", where NN is the
# length byte. rc.exe reads hex escapes greedily, so when <text> starts with
# a hex digit (e.g. "\x0Dags_utilities") the first letter is swallowed into
# the escape (\x0Da), the length byte is wrong and every PiPL property after
# it is misaligned -> AE: "Couldn't find main entry point" (48 :: 72).
#
# Rewrite every \xNN escape as a 3-digit octal escape (\ooo). Octal escapes
# stop after 3 digits, so the following characters are never consumed.
param([Parameter(Mandatory = $true)][string]$Path)

$text = [System.IO.File]::ReadAllText($Path)
$fixed = [regex]::Replace($text, '\\x([0-9A-Fa-f]{2})', {
    param($m)
    '\' + [Convert]::ToString([Convert]::ToInt32($m.Groups[1].Value, 16), 8).PadLeft(3, '0')
})
[System.IO.File]::WriteAllText($Path, $fixed, [System.Text.Encoding]::ASCII)

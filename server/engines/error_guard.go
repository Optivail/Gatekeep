// Mirrors backend/engines/error_guard.py: dual-state error protection that
// keeps diagnostics (tracebacks, failed-command output) and crushes noise.
package engines

import (
	"strings"

	"github.com/Optivail/Gatekeep/server/pyre"
)

// StacktraceCeiling is the default character budget for preserved diagnostics.
const StacktraceCeiling = 3000

// FileReadLineCap is kept for parity with the Python module constant.
const FileReadLineCap = 400

var (
	reExitNonzero = pyre.MustCompile(
		`(?im)(?:` +
			`exit(?:\s+|-)?code\s*[=:]?\s*(?!0\b)\d+` +
			`|exited with code (?!(?:0\b))\d+` +
			`|command failed(?: with exit code (?!(?:0\b))\d+)?` +
			`|returned non-zero exit status` +
			`)`)
	reStderrMarkers = pyre.MustCompile(`(?im)^stderr:\s*\S`)
	reErrorLine     = pyre.MustCompile(
		`(?im)(?:` +
			`traceback \(most recent call last\)` +
			`|^panic:` +
			`|unhandledpromiserejection` +
			`|syntaxerror:` +
			`|typeerror:` +
			`|referenceerror:` +
			`|error ts\d+` +
			`|error:\s` +
			`|fatal error:` +
			`|compilation failed` +
			`|build failed` +
			`|npm err!` +
			`|pytest\.|\bFAILED\b` +
			`)`)
	reTracebackStart = pyre.MustCompile(`Traceback \(most recent call last\):`, pyre.IgnoreCase)
	reBashError      = pyre.MustCompile(`(?im)^error:`)
)

// IsErrorPayload mirrors is_error_payload. toolName "" stands for None.
func IsErrorPayload(text string, toolName string, isErrorFlag bool) bool {
	if isErrorFlag {
		return true
	}
	if text == "" || pyStrip(text) == "" {
		return false
	}
	if reExitNonzero.MatchString(text) {
		return true
	}
	if reStderrMarkers.MatchString(text) && pyLen(pyStrip(text)) > 20 {
		return true
	}
	if reErrorLine.MatchString(text) {
		return true
	}
	if strings.Contains(pyLower(text), "traceback (most recent call last)") {
		return true
	}
	if toolName != "" {
		switch pyLower(toolName) {
		case "bash", "shell", "bashtool":
			if reBashError.MatchString(text) && pyLen(text) < 8000 {
				return true
			}
		}
	}
	return false
}

// traceBlockEnd mirrors _traceback_block_end: the exclusive end index of a
// traceback block that starts at lines[start].
func traceBlockEnd(lines []string, start int) int {
	end := start + 1
	for end < len(lines) {
		line := lines[end]
		if pyStrip(line) == "" {
			if end+1 < len(lines) && (strings.HasPrefix(lines[end+1], " ") ||
				strings.HasPrefix(lines[end+1], "\t") || reTracebackStart.MatchString(lines[end+1])) {
				end++
				continue
			}
			break
		}
		stripped := pyLStrip(line)
		if line[0] == ' ' || line[0] == '\t' || strings.HasPrefix(stripped, `File "`) {
			end++
			continue
		}
		if reTracebackStart.MatchString(line) {
			end++
			continue
		}
		// Exception summary line (ValueError: ..., panic: ..., etc.)
		if end > start+1 && reErrorLine.MatchString(line) {
			end++
			continue
		}
		if end > start+1 {
			break
		}
		end++
	}
	return end
}

// ExtractStacktraceBlocks mirrors extract_stacktrace_blocks: [start, end)
// line spans of each traceback block.
func ExtractStacktraceBlocks(text string) [][2]int {
	lines := strings.Split(text, "\n")
	var spans [][2]int
	for i := 0; i < len(lines); {
		if reTracebackStart.MatchString(lines[i]) {
			end := traceBlockEnd(lines, i)
			spans = append(spans, [2]int{i, end})
			i = end
		} else {
			i++
		}
	}
	return spans
}

// PreserveStacktraceBlocks mirrors preserve_stacktrace_blocks.
func PreserveStacktraceBlocks(text string, ceiling int) string {
	if text == "" {
		return text
	}
	spans := ExtractStacktraceBlocks(text)
	if len(spans) == 0 {
		if IsErrorPayload(text, "", false) && pyLen(text) > ceiling {
			return headTail(text, ceiling)
		}
		return text
	}
	lines := strings.Split(text, "\n")
	keep := make([]bool, len(lines))
	for _, sp := range spans {
		for i := sp[0]; i < sp[1]; i++ {
			keep[i] = true
		}
	}
	var out []string
	for i := 0; i < len(lines); {
		if keep[i] {
			out = append(out, lines[i])
			i++
			continue
		}
		start := i
		for i < len(lines) && !keep[i] {
			i++
		}
		if skipped := i - start; skipped > 0 {
			out = append(out, "[…"+itoa(skipped)+" non-trace lines omitted…]")
		}
	}
	preserved := strings.Join(out, "\n")
	if pyLen(preserved) > ceiling {
		return headTail(preserved, ceiling)
	}
	return preserved
}

// PreserveDiagnosticText mirrors preserve_diagnostic_text: full preservation
// for errors, bypassing the aggressive character caps.
func PreserveDiagnosticText(text string, ceiling int) string {
	if text == "" {
		return text
	}
	if reTracebackStart.MatchString(text) {
		return PreserveStacktraceBlocks(text, ceiling)
	}
	if IsErrorPayload(text, "", false) {
		if pyLen(text) <= ceiling {
			return text
		}
		return headTail(text, ceiling)
	}
	return text
}

// CrushLogLinesSafe mirrors crush_log_lines_safe. The crusher is only run on
// payloads that are not error diagnostics.
func CrushLogLinesSafe(text string, crusher func(string) string) string {
	if IsErrorPayload(text, "", false) {
		return PreserveDiagnosticText(text, StacktraceCeiling)
	}
	return crusher(text)
}

// headTail mirrors _head_tail. Python's // and negative-slice rules still
// apply when a caller passes a small or negative max_chars.
func headTail(text string, maxChars int) string {
	if pyLen(text) <= maxChars {
		return text
	}
	head := pyFloorDiv(maxChars*2, 3)
	tail := maxChars - head - 40
	omitted := pyLen(text) - head - tail
	return pyPrefix(text, head) + "\n…[" + itoa(omitted) + " chars omitted]…\n" + pySuffix(text, tail)
}

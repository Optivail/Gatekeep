package engines

import (
	"fmt"
	"math"
	"math/big"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// exprKind enumerates the expression nodes of Python's ast module that can
// appear in the parts of a module that are kept by the distiller (defaults,
// annotations, bases, decorators are dropped, bodies are stubbed).
type pyastExprKind uint8

const (
	pyastEName pyastExprKind = iota
	pyastEConst
	pyastEJoined
	pyastEList
	pyastETuple
	pyastESet
	pyastEDict
	pyastEListComp
	pyastEGenExp
	pyastESetComp
	pyastEDictComp
	pyastEIfExp
	pyastEUnary
	pyastEBinOp
	pyastECompare
	pyastEBoolOp
	pyastEAttr
	pyastECall
	pyastESubscript
	pyastEStarred
	pyastESlice
	pyastENamedExpr
	pyastEAwait
	pyastEYield
	pyastEYieldFrom
	pyastELambda
)

type pyastConstKind uint8

const (
	pyastCNone pyastConstKind = iota
	pyastCTrue
	pyastCFalse
	pyastCEllipsis
	pyastCInt
	pyastCFloat
	pyastCComplex
	pyastCStr
	pyastCBytes
)

type pyastConstant struct {
	kind    pyastConstKind
	digits  string // cInt: decimal representation, never negative
	f       float64
	s       []rune // cStr
	b       []byte // cBytes
	uprefix bool   // cStr: first token had a u prefix (Constant.kind == 'u')
}

// jsPart is one piece of an f-string: either literal text or a formatted
// value. Pieces are merged the way CPython's FstringParser merges them.
type pyastJsPart struct {
	isConst bool
	s       []rune
	fv      *pyastFormatted
}

type pyastFormatted struct {
	value   *pyastExpr
	conv    int // -1 when absent, otherwise the conversion character
	spec    []pyastJsPart
	hasSpec bool
}

type pyastExpr struct {
	kind  pyastExprKind
	name  string // Name.id, Attribute.attr
	cnst  *pyastConstant
	parts []pyastJsPart // JoinedStr values
	elts  []*pyastExpr  // List/Tuple/Set elements, Call args, BoolOp values, Compare comparators, Dict values
	keys  []*pyastExpr  // Dict keys (nil entry is a ** unpacking)
	comps []*pyastComprehension
	kws   []*pyastKeyword // Call keywords
	args  *pyastArguments // Lambda
	op    string          // UnaryOp, BinOp, BoolOp operator
	ops   []string        // Compare operators
	a     *pyastExpr      // main child: operand, left, value, elt, func, body, lower, target ...
	b     *pyastExpr      // second child: right, slice, value, key, test, upper ...
	c     *pyastExpr      // third child: orelse, step
}

type pyastComprehension struct {
	target  *pyastExpr
	iter    *pyastExpr
	ifs     []*pyastExpr
	isAsync bool
}

type pyastKeyword struct {
	hasArg  bool // false for **value
	argName string
	value   *pyastExpr
}

type pyastArg struct {
	name string
	ann  *pyastExpr
}

type pyastArguments struct {
	posonly    []*pyastArg
	args       []*pyastArg
	vararg     *pyastArg
	kwonly     []*pyastArg
	kwDefaults []*pyastExpr // parallel to kwonly, nil entries when absent
	kwarg      *pyastArg
	defaults   []*pyastExpr
}

type pyastTparamKind uint8

const (
	pyastTpVar pyastTparamKind = iota
	pyastTpTuple
	pyastTpSpec
)

type pyastTypeParam struct {
	kind  pyastTparamKind
	name  string
	bound *pyastExpr
}

type pyastStmtKind uint8

const (
	pyastSOther pyastStmtKind = iota
	pyastSFunc
	pyastSClass
)

// stmtInfo keeps only what the distiller needs from a statement: the header
// of a function or class and, for a class, its direct function children.
type pyastStmtInfo struct {
	kind     pyastStmtKind
	name     string
	isAsync  bool
	tparams  []pyastTypeParam
	args     *pyastArguments
	returns  *pyastExpr
	bases    []*pyastExpr
	keywords []*pyastKeyword
	methods  []*pyastStmtInfo
}

// tkKind is the token class handed from the lexer to the parser. Keywords are
// split out so that a NAME token can never be a reserved word.
type pyastTkKind uint8

const (
	pyastTkEOF pyastTkKind = iota
	pyastTkNewline
	pyastTkIndent
	pyastTkDedent
	pyastTkName
	pyastTkKeyword
	pyastTkNumber
	pyastTkString
	pyastTkFStart
	pyastTkFMid
	pyastTkFEnd
	pyastTkOp
)

type pyastToken struct {
	kind  pyastTkKind
	text  string // NAME/KEYWORD/NUMBER/OP text, raw STRING text, FSTART prefix, FMID raw text
	quote string // FSTART/FEND delimiter such as the quote or three quotes
	start int    // byte offsets into the translated source
	end   int
}

// pyastDeclined is the panic payload for an input this port cannot answer
// (a broken embedded table, or a nesting depth beyond the proven limit). It is
// turned into ErrUnsupported, which sends the request to the Python backend.
type pyastDeclined struct{}

// syntaxErr is the panic payload for every rejection. Any other panic is a bug
// and is re-raised by the recover sites.
type pyastSyntaxErr struct{}

func pyastBail() { panic(pyastSyntaxErr{}) }

var pyastHardKeywords = map[string]bool{
	"False": true, "None": true, "True": true, "and": true, "as": true,
	"assert": true, "async": true, "await": true, "break": true, "class": true,
	"continue": true, "def": true, "del": true, "elif": true, "else": true,
	"except": true, "finally": true, "for": true, "from": true, "global": true,
	"if": true, "import": true, "in": true, "is": true, "lambda": true,
	"nonlocal": true, "not": true, "or": true, "pass": true, "raise": true,
	"return": true, "try": true, "while": true, "with": true, "yield": true,
}

// operators, longest first so that greedy matching works.
var pyastOperators = []string{
	"**=", "//=", ">>=", "<<=", "...",
	"!=", "%=", "&=", "**", "*=", "+=", "-=", "->", "//", "/=", ":=", "<<", "<=",
	"==", ">=", ">>", "@=", "^=", "|=",
	"%", "&", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">",
	"@", "[", "]", "^", "{", "|", "}", "~",
}

const (
	pyastFmLit  = iota // f-string literal fragment
	pyastFmSpec        // f-string format spec fragment
	pyastFmExpr        // replacement field expression (ordinary code)
)

type pyastFmode struct {
	kind    int
	quote   string // delimiter of the enclosing f-string
	raw     bool
	brDepth int // fmExpr: len(brackets) before the field's '{' was pushed
	// specDepth counts format-spec nesting inside this f-string only; a
	// nested f-string in an expression starts again at zero.
	specDepth int
	// codeMode is set on a format spec once a single-quoted spec has seen a
	// newline: CPython then reads the rest of the spec as code.
	codeMode bool
}

type pyastLexer struct {
	src      string
	pos      int
	toks     []pyastToken
	indents  []int
	alts     []int // tab-size-1 indentation, for TabError detection
	brackets []byte
	modes    []pyastFmode
	bol      bool // at the start of a logical line, indentation pending
	hasTok   bool // a token was emitted since the last NEWLINE
}

// tokenize splits translated source into tokens. Any tokenizer error panics
// with syntaxErr, which rejects the whole module (CPython does the same, the
// parser needs every token).
func pyastTokenize(src string) []pyastToken {
	lx := &pyastLexer{src: src, indents: []int{0}, alts: []int{0}, bol: true}
	lx.run()
	return lx.toks
}

func (lx *pyastLexer) emit(k pyastTkKind, text, quote string, start, end int) {
	lx.toks = append(lx.toks, pyastToken{kind: k, text: text, quote: quote, start: start, end: end})
	if k != pyastTkNewline && k != pyastTkIndent && k != pyastTkDedent {
		lx.hasTok = true
	}
}

func (lx *pyastLexer) run() {
	for {
		if n := len(lx.modes); n > 0 && lx.modes[n-1].kind != pyastFmExpr {
			lx.fstringText()
			continue
		}
		if lx.bol {
			lx.indentation()
		}
		if lx.pos >= len(lx.src) {
			lx.finish()
			return
		}
		c := lx.src[lx.pos]
		switch {
		case c == ' ' || c == '\t' || c == '\f':
			lx.pos++
		case c == '#':
			for lx.pos < len(lx.src) && lx.src[lx.pos] != '\n' {
				lx.pos++
			}
		case c == '\n':
			lx.pos++
			if len(lx.brackets) > 0 {
				continue
			}
			if lx.hasTok {
				lx.emit(pyastTkNewline, "", "", lx.pos-1, lx.pos)
				lx.hasTok = false
			}
			lx.bol = true
		case c == '\\':
			if lx.pos+1 < len(lx.src) && lx.src[lx.pos+1] == '\n' {
				lx.pos += 2
				if lx.pos >= len(lx.src) {
					pyastBail() // unexpected EOF while parsing
				}
			} else {
				pyastBail()
			}
		case pyastIsASCIIIdentStart(c) || c >= 0x80:
			lx.name()
		case pyastIsDigitByte(c) || (c == '.' && lx.pos+1 < len(lx.src) && pyastIsDigitByte(lx.src[lx.pos+1])):
			lx.number()
		case c == '\'' || c == '"':
			lx.stringLit(lx.pos, "")
		default:
			lx.operator()
		}
	}
}

// indentation handles the first token of a logical line the way CPython's
// tokenizer does: blank and comment-only lines are skipped, and the column
// and tab-insensitive column are compared against the indentation stack.
func (lx *pyastLexer) indentation() {
	s := lx.src
	col, altcol := 0, 0
	// CPython lets backslash continuations sit inside the indentation: the
	// column keeps accumulating across them, but the first non-zero column
	// seen at a backslash wins (cont_line_col).
	contCol := 0
	p := lx.pos
	for p < len(s) {
		c := s[p]
		if c == ' ' {
			col++
			altcol++
		} else if c == '\t' {
			col = (col/8 + 1) * 8
			altcol++
		} else if c == '\f' {
			col, altcol = 0, 0
		} else if c == '\\' {
			if contCol == 0 {
				contCol = col
			}
			if p+1 >= len(s) || s[p+1] != '\n' {
				pyastBail() // unexpected character after line continuation character
			}
			p += 2
			if p >= len(s) {
				pyastBail() // unexpected EOF while parsing
			}
			continue
		} else {
			break
		}
		p++
	}
	lx.pos = p
	if p >= len(s) || s[p] == '#' || s[p] == '\n' {
		return
	}
	if contCol != 0 {
		col, altcol = contCol, contCol
	}
	lx.bol = false
	top := len(lx.indents) - 1
	switch {
	case col == lx.indents[top]:
		if altcol != lx.alts[top] {
			pyastBail() // TabError
		}
	case col > lx.indents[top]:
		if len(lx.indents) >= 100 {
			pyastBail() // too many levels of indentation
		}
		if altcol <= lx.alts[top] {
			pyastBail() // TabError
		}
		lx.indents = append(lx.indents, col)
		lx.alts = append(lx.alts, altcol)
		lx.emit(pyastTkIndent, "", "", p, p)
	default:
		for len(lx.indents) > 1 && col < lx.indents[len(lx.indents)-1] {
			lx.indents = lx.indents[:len(lx.indents)-1]
			lx.alts = lx.alts[:len(lx.alts)-1]
			lx.emit(pyastTkDedent, "", "", p, p)
		}
		last := len(lx.indents) - 1
		if col != lx.indents[last] {
			pyastBail() // unindent does not match any outer indentation level
		}
		if altcol != lx.alts[last] {
			pyastBail() // TabError
		}
	}
}

func (lx *pyastLexer) finish() {
	if len(lx.brackets) > 0 || len(lx.modes) > 0 {
		pyastBail() // unclosed bracket or f-string at EOF
	}
	if lx.hasTok {
		lx.emit(pyastTkNewline, "", "", lx.pos, lx.pos)
		lx.hasTok = false
	}
	for len(lx.indents) > 1 {
		lx.indents = lx.indents[:len(lx.indents)-1]
		lx.alts = lx.alts[:len(lx.alts)-1]
		lx.emit(pyastTkDedent, "", "", lx.pos, lx.pos)
	}
	lx.emit(pyastTkEOF, "", "", lx.pos, lx.pos)
}

func pyastIsASCIIIdentStart(c byte) bool {
	return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

func pyastIsASCIIIdentChar(c byte) bool {
	return pyastIsASCIIIdentStart(c) || pyastIsDigitByte(c)
}

func pyastIsDigitByte(c byte) bool { return c >= '0' && c <= '9' }

func pyastIsHexByte(c byte) bool {
	return pyastIsDigitByte(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')
}

// isStringPrefix reports whether word (any case) is a valid string prefix.
func pyastIsStringPrefix(word string) bool {
	switch strings.ToLower(word) {
	case "r", "u", "b", "br", "rb", "f", "fr", "rf":
		return true
	}
	return false
}

func (lx *pyastLexer) name() {
	s := lx.src
	start := lx.pos
	for lx.pos < len(s) && (pyastIsASCIIIdentChar(s[lx.pos]) || s[lx.pos] >= 0x80) {
		lx.pos++
	}
	word := s[start:lx.pos]
	if lx.pos < len(s) && (s[lx.pos] == '\'' || s[lx.pos] == '"') && pyastIsStringPrefix(word) {
		lx.stringLit(start, strings.ToLower(word))
		return
	}
	if !pyastValidIdentifier(word) {
		pyastBail()
	}
	if pyastHardKeywords[word] {
		lx.emit(pyastTkKeyword, word, "", start, lx.pos)
	} else {
		lx.emit(pyastTkName, word, "", start, lx.pos)
	}
}

// pyastIdentifier is the identifier CPython stores in the AST: NAME tokens are
// NFKC-normalised (PEP 3131). ASCII names are unchanged. Keyword and soft
// keyword checks use the raw token text, as CPython does, so they do not come
// through here.
func pyastIdentifier(raw string) string {
	for i := 0; i < len(raw); i++ {
		if raw[i] >= 0x80 {
			return norm.NFKC.String(raw)
		}
	}
	return raw
}

// validIdentifier approximates PEP 3131 (XID_Start / XID_Continue) with the
// Go unicode tables. It does not apply NFKC normalisation (see report).
func pyastValidIdentifier(w string) bool {
	if w == "" {
		return false
	}
	for i, r := range w {
		if i == 0 {
			if r != '_' && !pyastIsXIDStart(r) {
				return false
			}
			continue
		}
		if !pyastIsXIDContinue(r) {
			return false
		}
	}
	return true
}

// isXIDStart is XID_Start as CPython 3.12 sees it (Unicode 15.0), built from
// the Go tables with the xidStartExtra corrections applied.
func pyastIsXIDStart(r rune) bool {
	if r < 0x80 {
		return pyastIsASCIIIdentStart(byte(r))
	}
	return pyastIsXIDStartBase(r) && !pyastInRangePairs(pyastXidStartExtra[:], r)
}

// isXIDStartBase is the uncorrected start predicate. XID_Continue contains
// all of ID_Start, so the continue check must use this, not the corrected one.
func pyastIsXIDStartBase(r rune) bool {
	if unicode.In(r, unicode.Pattern_Syntax, unicode.Pattern_White_Space) {
		return false
	}
	return unicode.IsLetter(r) || unicode.Is(unicode.Nl, r) || unicode.Is(unicode.Other_ID_Start, r)
}

// isXIDContinue is XID_Continue as CPython 3.12 sees it (Unicode 15.0).
func pyastIsXIDContinue(r rune) bool {
	if r < 0x80 {
		return pyastIsASCIIIdentChar(byte(r))
	}
	if pyastIsXIDStartBase(r) {
		return !pyastInRangePairs(pyastXidContinueExtra[:], r)
	}
	if unicode.In(r, unicode.Pattern_Syntax, unicode.Pattern_White_Space) {
		return false
	}
	ok := unicode.In(r, unicode.Mn, unicode.Mc, unicode.Nd, unicode.Pc, unicode.Other_ID_Continue)
	return ok && !pyastInRangePairs(pyastXidContinueExtra[:], r)
}

// stringLit lexes a string whose opening quote is at lx.pos. start is the
// offset of the prefix (equal to lx.pos when there is none) and prefix is the
// lower-cased prefix.
func (lx *pyastLexer) stringLit(start int, prefix string) {
	s := lx.src
	q := s[lx.pos]
	qs := string([]byte{q})
	triple := lx.pos+2 < len(s) && s[lx.pos+1] == q && s[lx.pos+2] == q
	if triple {
		qs = strings.Repeat(string([]byte{q}), 3)
	}
	lx.pos += len(qs)
	if strings.ContainsRune(prefix, 'f') {
		lx.emit(pyastTkFStart, prefix, qs, start, lx.pos)
		lx.modes = append(lx.modes, pyastFmode{kind: pyastFmLit, quote: qs, raw: strings.ContainsRune(prefix, 'r')})
		return
	}
	for {
		if lx.pos >= len(s) {
			pyastBail()
		}
		c := s[lx.pos]
		switch {
		case c == '\\':
			if lx.pos+1 >= len(s) {
				pyastBail()
			}
			lx.pos += 2
		case c == '\n' && !triple:
			pyastBail()
		case c == q && (!triple || pyastHasPrefixAt(s, lx.pos, qs)):
			lx.pos += len(qs)
			lx.emit(pyastTkString, s[start:lx.pos], "", start, lx.pos)
			return
		default:
			lx.pos++
		}
	}
}

// specCodeText skips the part of a single-quoted format spec that follows a
// newline. CPython reads it as code there: blanks, comments and line
// continuations are skipped, and the next token must be '{' or '}'.
func (lx *pyastLexer) specCodeText() {
	s := lx.src
	for {
		if lx.pos >= len(s) {
			pyastBail()
		}
		c := s[lx.pos]
		switch {
		case c == ' ' || c == '\t' || c == '\f' || c == '\n':
			lx.pos++
		case c == '#':
			for lx.pos < len(s) && s[lx.pos] != '\n' {
				lx.pos++
			}
		case c == '\\' && lx.pos+1 < len(s) && s[lx.pos+1] == '\n':
			lx.pos += 2
		case c == '{' || c == '}':
			return
		default:
			pyastBail()
		}
	}
}

func pyastHasPrefixAt(s string, p int, prefix string) bool {
	return p >= 0 && p+len(prefix) <= len(s) && s[p:p+len(prefix)] == prefix
}

// fstringText scans the literal or format-spec part of an f-string, emitting
// FMID and the closing tokens. It mirrors the PEP 701 tokenizer modes.
func (lx *pyastLexer) fstringText() {
	s := lx.src
	m := lx.modes[len(lx.modes)-1]
	triple := len(m.quote) == 3
	if m.kind == pyastFmSpec && m.codeMode {
		lx.specCodeText()
	}
	start := lx.pos
	newline := false
	for {
		if lx.pos >= len(s) {
			pyastBail()
		}
		c := s[lx.pos]
		if m.kind == pyastFmLit && c == m.quote[0] && pyastHasPrefixAt(s, lx.pos, m.quote) {
			break
		}
		if m.kind == pyastFmSpec && c == m.quote[0] && pyastHasPrefixAt(s, lx.pos, m.quote) {
			pyastBail() // f-string closed before its replacement field
		}
		if c == '\n' && !triple {
			if m.kind != pyastFmSpec {
				pyastBail()
			}
			newline = true
			break
		}
		if c == '\\' {
			if !m.raw && m.kind == pyastFmLit && lx.pos+2 < len(s) && s[lx.pos+1] == 'N' && s[lx.pos+2] == '{' {
				end := strings.IndexByte(s[lx.pos:], '}')
				if end < 0 {
					pyastBail()
				}
				lx.pos += end + 1
				continue
			}
			if lx.pos+1 >= len(s) {
				pyastBail()
			}
			if n := s[lx.pos+1]; n == '{' || n == '}' {
				// the backslash stays literal and the brace is processed next
				lx.pos++
				continue
			}
			lx.pos += 2
			continue
		}
		if c == '{' {
			if m.kind == pyastFmLit && lx.pos+1 < len(s) && s[lx.pos+1] == '{' {
				lx.pos += 2
				continue
			}
			break
		}
		if c == '}' {
			if m.kind == pyastFmLit {
				if lx.pos+1 < len(s) && s[lx.pos+1] == '}' {
					lx.pos += 2
					continue
				}
				pyastBail() // single '}' is not allowed
			}
			break
		}
		lx.pos++
	}
	if lx.pos > start {
		lx.emit(pyastTkFMid, s[start:lx.pos], "", start, lx.pos)
	}
	if newline {
		lx.pos++
		lx.modes[len(lx.modes)-1].codeMode = true
		lx.specCodeText()
	}
	c := s[lx.pos]
	switch {
	case m.kind == pyastFmLit && c == m.quote[0] && pyastHasPrefixAt(s, lx.pos, m.quote):
		end := lx.pos + len(m.quote)
		lx.emit(pyastTkFEnd, "", m.quote, lx.pos, end)
		lx.pos = end
		lx.modes = lx.modes[:len(lx.modes)-1]
	case c == '{':
		if m.specDepth >= 3 {
			pyastBail() // f-string expressions nested too deeply
		}
		lx.emit(pyastTkOp, "{", "", lx.pos, lx.pos+1)
		lx.pushBracket('{')
		lx.modes = append(lx.modes, pyastFmode{kind: pyastFmExpr, quote: m.quote, raw: m.raw, brDepth: len(lx.brackets) - 1, specDepth: m.specDepth})
		lx.pos++
	case c == '}' && m.kind == pyastFmSpec:
		// end of a format spec: close the spec and the field that owns it
		lx.emit(pyastTkOp, "}", "", lx.pos, lx.pos+1)
		lx.pos++
		lx.modes = lx.modes[:len(lx.modes)-1]
		lx.closeFieldFrame()
	default:
		pyastBail()
	}
}

func (lx *pyastLexer) pushBracket(c byte) {
	if len(lx.brackets) >= 200 {
		pyastBail() // too many nested parentheses
	}
	lx.brackets = append(lx.brackets, c)
}

// closeFieldFrame pops the expression frame of a replacement field together
// with the '{' it pushed. It is used after the field's '}' has been emitted.
func (lx *pyastLexer) closeFieldFrame() {
	lx.brackets = lx.brackets[:len(lx.brackets)-1]
	lx.modes = lx.modes[:len(lx.modes)-1]
}

// atFieldDepth reports whether the lexer is directly inside a replacement
// field expression, not inside a bracket nested in it.
func (lx *pyastLexer) atFieldDepth() bool {
	n := len(lx.modes)
	return n > 0 && lx.modes[n-1].kind == pyastFmExpr && len(lx.brackets) == lx.modes[n-1].brDepth+1
}

func (lx *pyastLexer) operator() {
	s := lx.src
	p := lx.pos
	if lx.atFieldDepth() {
		m := lx.modes[len(lx.modes)-1]
		switch s[p] {
		case '}':
			lx.emit(pyastTkOp, "}", "", p, p+1)
			lx.pos++
			lx.closeFieldFrame()
			return
		case ':':
			lx.emit(pyastTkOp, ":", "", p, p+1)
			lx.pos++
			lx.modes = append(lx.modes, pyastFmode{kind: pyastFmSpec, quote: m.quote, raw: m.raw, specDepth: m.specDepth + 1})
			return
		case '!':
			if p+1 >= len(s) || s[p+1] != '=' {
				lx.emit(pyastTkOp, "!", "", p, p+1)
				lx.pos++
				return
			}
		}
	}
	for _, op := range pyastOperators {
		if !pyastHasPrefixAt(s, p, op) {
			continue
		}
		switch op {
		case "(", "[", "{":
			lx.pushBracket(op[0])
		case ")", "]", "}":
			n := len(lx.brackets)
			if n == 0 {
				pyastBail()
			}
			want := map[string]byte{")": '(', "]": '[', "}": '{'}[op]
			if lx.brackets[n-1] != want {
				pyastBail()
			}
			lx.brackets = lx.brackets[:n-1]
		}
		lx.emit(pyastTkOp, op, "", p, p+len(op))
		lx.pos += len(op)
		return
	}
	pyastBail()
}

// number lexes an int, float or imaginary literal following CPython's
// tokenizer, including its verify_end_of_number rules.
func (lx *pyastLexer) number() {
	s := lx.src
	start := lx.pos
	end := -1
	if s[start] == '0' && start+1 < len(s) {
		switch s[start+1] {
		case 'x', 'X':
			end = pyastRadixLiteral(s, start+2, pyastIsHexByte, "hexadecimal", false)
		case 'o', 'O':
			end = pyastRadixLiteral(s, start+2, func(c byte) bool { return c >= '0' && c <= '7' }, "octal", true)
		case 'b', 'B':
			end = pyastRadixLiteral(s, start+2, func(c byte) bool { return c == '0' || c == '1' }, "binary", true)
		}
	}
	if end < 0 {
		end = pyastDecimalLiteral(s, start)
	}
	lx.emit(pyastTkNumber, s[start:end], "", start, end)
	lx.pos = end
}

func pyastAt(s string, p int) byte {
	if p < len(s) {
		return s[p]
	}
	return 0
}

func pyastRadixLiteral(s string, p int, valid func(byte) bool, kind string, octBin bool) int {
	for {
		if p < len(s) && s[p] == '_' {
			p++
		}
		if p >= len(s) || !valid(s[p]) {
			pyastBail()
		}
		for p < len(s) && valid(s[p]) {
			p++
		}
		if p < len(s) && s[p] == '_' {
			continue
		}
		break
	}
	if octBin && pyastIsDigitByte(pyastAt(s, p)) {
		pyastBail()
	}
	pyastVerifyEndOfNumber(s, p)
	return p
}

// decimalTail consumes decimal digits with single underscores between them.
func pyastDecimalTail(s string, p int) int {
	for {
		for p < len(s) && pyastIsDigitByte(s[p]) {
			p++
		}
		if p >= len(s) || s[p] != '_' {
			return p
		}
		p++
		if p >= len(s) || !pyastIsDigitByte(s[p]) {
			pyastBail()
		}
	}
}

// verifyEndOfNumber rejects a number directly followed by an identifier
// character, except for the keywords CPython allows there (with a warning).
func pyastVerifyEndOfNumber(s string, p int) {
	if p >= len(s) {
		return
	}
	if pyastKwFollows(s, p) {
		return
	}
	c := s[p]
	if pyastIsASCIIIdentChar(c) || c >= 0x80 {
		pyastBail()
	}
}

func pyastKwFollows(s string, p int) bool {
	rest := s[p:]
	switch s[p] {
	case 'a':
		return strings.HasPrefix(rest, "and")
	case 'e':
		return strings.HasPrefix(rest, "else")
	case 'f':
		return strings.HasPrefix(rest, "for")
	case 'i':
		return len(rest) > 1 && (rest[1] == 'f' || rest[1] == 'n' || rest[1] == 's')
	case 'o':
		return strings.HasPrefix(rest, "or")
	case 'n':
		return strings.HasPrefix(rest, "not")
	}
	return false
}

// decimalLiteral mirrors the decimal branch of CPython's tokenizer and returns
// the end offset of the literal starting at start.
func pyastDecimalLiteral(s string, start int) int {
	p := start
	if s[p] == '.' {
		return pyastFractionPart(s, p+1)
	}
	if s[p] == '0' {
		p++
		for {
			if p < len(s) && s[p] == '_' {
				p++
				if !pyastIsDigitByte(pyastAt(s, p)) {
					pyastBail()
				}
			}
			if p < len(s) && s[p] == '0' {
				p++
				continue
			}
			break
		}
		nonzero := false
		if pyastIsDigitByte(pyastAt(s, p)) {
			nonzero = true
			p = pyastDecimalTail(s, p)
		}
		switch pyastAt(s, p) {
		case '.':
			return pyastFractionPart(s, p+1)
		case 'e', 'E':
			return pyastExponentPart(s, p)
		case 'j', 'J':
			return pyastImaginaryPart(s, p)
		}
		if nonzero {
			pyastBail() // leading zeros in decimal integer literals
		}
		pyastVerifyEndOfNumber(s, p)
		pyastCheckIntDigits(s[start:p])
		return p
	}
	p = pyastDecimalTail(s, p)
	switch pyastAt(s, p) {
	case '.':
		return pyastFractionPart(s, p+1)
	case 'e', 'E':
		return pyastExponentPart(s, p)
	case 'j', 'J':
		return pyastImaginaryPart(s, p)
	}
	pyastVerifyEndOfNumber(s, p)
	pyastCheckIntDigits(s[start:p])
	return p
}

// checkIntDigits enforces CPython's 4300-digit limit for decimal integers.
// Leading zeros do not count, since the value is converted without them.
func pyastCheckIntDigits(lit string) {
	n := 0
	significant := false
	for i := 0; i < len(lit); i++ {
		if pyastIsDigitByte(lit[i]) && (significant || lit[i] != '0') {
			significant = true
			n++
		}
	}
	if n > 4300 {
		pyastBail()
	}
}

func pyastFractionPart(s string, p int) int {
	if pyastIsDigitByte(pyastAt(s, p)) {
		p = pyastDecimalTail(s, p)
	}
	switch pyastAt(s, p) {
	case 'e', 'E':
		return pyastExponentPart(s, p)
	case 'j', 'J':
		return pyastImaginaryPart(s, p)
	}
	pyastVerifyEndOfNumber(s, p)
	return p
}

func pyastExponentPart(s string, e int) int {
	p := e + 1
	c := pyastAt(s, p)
	switch {
	case c == '+' || c == '-':
		p++
		if !pyastIsDigitByte(pyastAt(s, p)) {
			pyastBail()
		}
		p = pyastDecimalTail(s, p)
	case !pyastIsDigitByte(c):
		// "1else": the number ends before the 'e'
		pyastVerifyEndOfNumber(s, e)
		return e
	default:
		p = pyastDecimalTail(s, p)
	}
	switch pyastAt(s, p) {
	case 'j', 'J':
		return pyastImaginaryPart(s, p)
	}
	pyastVerifyEndOfNumber(s, p)
	return p
}

func pyastImaginaryPart(s string, j int) int {
	p := j + 1
	pyastVerifyEndOfNumber(s, p)
	return p
}

// parser is a recursive-descent parser that accepts exactly the language of
// CPython 3.12's PEG grammar (python.gram) without the invalid_* rules, which
// only ever add error messages for inputs the valid rules already reject.
// Action-level errors (bad literals, bytes/str mixing, target validity,
// complex-pattern checks) are raised with bail() as well.
type pyastParser struct {
	src  string
	toks []pyastToken
	pos  int
}

func (p *pyastParser) peek() pyastToken { return p.toks[p.pos] }

func (p *pyastParser) at(k int) pyastToken {
	i := p.pos + k
	if i >= len(p.toks) {
		i = len(p.toks) - 1
	}
	return p.toks[i]
}

func (p *pyastParser) advance() pyastToken {
	t := p.toks[p.pos]
	if p.pos < len(p.toks)-1 {
		p.pos++
	}
	return t
}

func (p *pyastParser) isOp(s string) bool {
	t := p.toks[p.pos]
	return t.kind == pyastTkOp && t.text == s
}

func (p *pyastParser) isKw(s string) bool {
	t := p.toks[p.pos]
	return t.kind == pyastTkKeyword && t.text == s
}

func (p *pyastParser) acceptOp(s string) bool {
	if p.isOp(s) {
		p.advance()
		return true
	}
	return false
}

func (p *pyastParser) acceptKw(s string) bool {
	if p.isKw(s) {
		p.advance()
		return true
	}
	return false
}

func (p *pyastParser) expectOp(s string) {
	if !p.acceptOp(s) {
		pyastBail()
	}
}

func (p *pyastParser) expectKw(s string) {
	if !p.acceptKw(s) {
		pyastBail()
	}
}

func (p *pyastParser) expectName() string {
	t := p.peek()
	if t.kind != pyastTkName {
		pyastBail()
	}
	p.advance()
	return pyastIdentifier(t.text)
}

func (p *pyastParser) expectNewline() {
	if p.peek().kind != pyastTkNewline {
		pyastBail()
	}
	p.advance()
}

// try runs f and rewinds the token position if it bails.
func (p *pyastParser) try(f func()) (ok bool) {
	save := p.pos
	defer func() {
		if r := recover(); r != nil {
			if _, is := r.(pyastSyntaxErr); !is {
				panic(r)
			}
			p.pos = save
			ok = false
		}
	}()
	f()
	return true
}

func pyastParseModule(src string) (stmts []*pyastStmtInfo, ok bool) {
	defer func() {
		if r := recover(); r != nil {
			if _, is := r.(pyastSyntaxErr); is {
				stmts, ok = nil, false
				return
			}
			panic(r)
		}
	}()
	p := &pyastParser{src: src, toks: pyastTokenize(src)}
	for p.peek().kind != pyastTkEOF {
		stmts = append(stmts, p.statement())
	}
	if pyastChainWeight(p.toks) > pyastChainLimit {
		panic(pyastDeclined{})
	}
	return stmts, true
}

// pyastChainLimit bounds how deep an expression can nest. CPython's ast.unparse
// fails on deep expressions (the first failure measured was a 164-level dict
// display, and binary-operator chains failed between 300 and 500 operators).
// That failure is caught, so the function returns the original code, which this
// port does not model. Inputs above the limit are declined instead of guessed.
// The limit is well below the smallest measured failure.
const pyastChainLimit = 80

// pyastChainWeight returns the largest nesting weight of one logical line. The
// weight of a path through the expression is the number of open brackets on it
// plus the operator tokens since the last sibling separator at each open level.
// Siblings (after ',', ':', '=') reset their level, so a long flat literal
// weighs little, while deep nesting and long operator chains weigh a lot. The
// weight bounds the expression depth, so it bounds the recursion.
func pyastChainWeight(toks []pyastToken) int {
	type level struct{ ops int }
	var stack []level
	topOps := 0
	weight := 0 // sum over open levels of (1 + ops), plus topOps
	best := 0
	for _, t := range toks {
		switch t.kind {
		case pyastTkNewline:
			stack, topOps, weight = stack[:0], 0, 0
		case pyastTkFStart:
			stack = append(stack, level{})
			weight++
		case pyastTkFEnd:
			if n := len(stack); n > 0 {
				weight -= 1 + stack[n-1].ops
				stack = stack[:n-1]
			}
		case pyastTkOp:
			switch {
			case t.text == "(" || t.text == "[" || t.text == "{":
				stack = append(stack, level{})
				weight++
			case t.text == ")" || t.text == "]" || t.text == "}":
				if n := len(stack); n > 0 {
					weight -= 1 + stack[n-1].ops
					stack = stack[:n-1]
				}
			case t.text == "," || t.text == ":" || t.text == "=":
				if n := len(stack); n > 0 {
					weight -= stack[n-1].ops
					stack[n-1].ops = 0
				} else {
					weight -= topOps
					topOps = 0
				}
			case pyastWeightOps[t.text]:
				if n := len(stack); n > 0 {
					stack[n-1].ops++
				} else {
					topOps++
				}
				weight++
			}
		case pyastTkKeyword:
			if pyastWeightKeywords[t.text] {
				if n := len(stack); n > 0 {
					stack[n-1].ops++
				} else {
					topOps++
				}
				weight++
			}
		}
		if weight > best {
			best = weight
		}
	}
	return best
}

// pyastWeightOps are the operator tokens that build expression nodes. The dot
// is an attribute access, and the rest are unary, binary and comparison
// operators.
var pyastWeightOps = map[string]bool{
	"+": true, "-": true, "*": true, "/": true, "//": true, "%": true, "@": true,
	"**": true, "<<": true, ">>": true, "&": true, "|": true, "^": true, "~": true,
	"<": true, ">": true, "==": true, "!=": true, "<=": true, ">=": true,
	".": true, ":=": true,
}

// pyastWeightKeywords are the keywords that build expression nodes.
var pyastWeightKeywords = map[string]bool{
	"not": true, "and": true, "or": true, "if": true, "else": true, "lambda": true,
	"await": true, "yield": true, "in": true, "is": true, "for": true,
}

var pyastOther = &pyastStmtInfo{kind: pyastSOther}

// ---------------------------------------------------------------- statements

func (p *pyastParser) statement() *pyastStmtInfo {
	t := p.peek()
	if t.kind == pyastTkOp && t.text == "@" {
		return p.decorated()
	}
	if t.kind == pyastTkKeyword {
		switch t.text {
		case "def":
			return p.funcDef(false)
		case "class":
			return p.classDef()
		case "if":
			p.ifStmt()
			return pyastOther
		case "while":
			p.whileStmt()
			return pyastOther
		case "for":
			p.forStmt()
			return pyastOther
		case "with":
			p.withStmt()
			return pyastOther
		case "try":
			p.tryStmt()
			return pyastOther
		case "async":
			return p.asyncStmt()
		}
	}
	if t.kind == pyastTkName && t.text == "match" {
		if p.try(func() { p.matchStmt() }) {
			return pyastOther
		}
	}
	p.simpleStmts()
	return pyastOther
}

func (p *pyastParser) decorated() *pyastStmtInfo {
	for p.acceptOp("@") {
		p.namedExpression()
		p.expectNewline()
	}
	switch {
	case p.isKw("def"):
		return p.funcDef(false)
	case p.isKw("async"):
		return p.funcDef(true)
	case p.isKw("class"):
		return p.classDef()
	}
	pyastBail()
	return nil
}

func (p *pyastParser) asyncStmt() *pyastStmtInfo {
	n := p.at(1)
	if n.kind != pyastTkKeyword {
		pyastBail()
	}
	switch n.text {
	case "def":
		return p.funcDef(true)
	case "for":
		p.advance()
		p.forStmt()
		return pyastOther
	case "with":
		p.advance()
		p.withStmt()
		return pyastOther
	}
	pyastBail()
	return nil
}

func (p *pyastParser) funcDef(isAsync bool) *pyastStmtInfo {
	if isAsync {
		p.expectKw("async")
	}
	p.expectKw("def")
	name := p.expectName()
	var tps []pyastTypeParam
	if p.isOp("[") {
		tps = p.typeParams()
	}
	p.expectOp("(")
	args := p.params(false)
	p.expectOp(")")
	var ret *pyastExpr
	if p.acceptOp("->") {
		ret = p.expression()
	}
	p.expectOp(":")
	p.block()
	return &pyastStmtInfo{kind: pyastSFunc, name: name, isAsync: isAsync, tparams: tps, args: args, returns: ret}
}

func (p *pyastParser) classDef() *pyastStmtInfo {
	p.expectKw("class")
	name := p.expectName()
	var tps []pyastTypeParam
	if p.isOp("[") {
		tps = p.typeParams()
	}
	var bases []*pyastExpr
	var kws []*pyastKeyword
	if p.acceptOp("(") {
		bases, kws = p.callArgs(false)
	}
	p.expectOp(":")
	body := p.block()
	info := &pyastStmtInfo{kind: pyastSClass, name: name, tparams: tps, bases: bases, keywords: kws}
	for _, s := range body {
		if s.kind == pyastSFunc {
			info.methods = append(info.methods, s)
		}
	}
	return info
}

// block parses "NEWLINE INDENT statements DEDENT" or a simple_stmts line and
// returns the statements of an indented block.
func (p *pyastParser) block() []*pyastStmtInfo {
	if p.peek().kind != pyastTkNewline {
		p.simpleStmts()
		return nil
	}
	p.advance()
	if p.peek().kind != pyastTkIndent {
		pyastBail()
	}
	p.advance()
	var out []*pyastStmtInfo
	for p.peek().kind != pyastTkDedent {
		if p.peek().kind == pyastTkEOF {
			pyastBail()
		}
		out = append(out, p.statement())
	}
	p.advance()
	return out
}

func (p *pyastParser) ifStmt() {
	p.advance() // if or elif
	p.namedExpression()
	p.expectOp(":")
	p.block()
	if p.isKw("elif") {
		p.ifStmt()
		return
	}
	if p.acceptKw("else") {
		p.expectOp(":")
		p.block()
	}
}

func (p *pyastParser) whileStmt() {
	p.expectKw("while")
	p.namedExpression()
	p.expectOp(":")
	p.block()
	if p.acceptKw("else") {
		p.expectOp(":")
		p.block()
	}
}

func (p *pyastParser) forStmt() {
	p.expectKw("for")
	p.starTargets()
	p.expectKw("in")
	p.starExpressions()
	p.expectOp(":")
	p.block()
	if p.acceptKw("else") {
		p.expectOp(":")
		p.block()
	}
}

func (p *pyastParser) withStmt() {
	p.expectKw("with")
	if p.isOp("(") {
		ok := p.try(func() {
			p.advance()
			p.withItemList(true)
			p.expectOp(")")
			p.expectOp(":")
		})
		if ok {
			p.block()
			return
		}
	}
	p.withItemList(false)
	p.expectOp(":")
	p.block()
}

func (p *pyastParser) withItemList(paren bool) {
	p.withItem()
	for p.acceptOp(",") {
		if paren && p.isOp(")") {
			return
		}
		p.withItem()
	}
}

func (p *pyastParser) withItem() {
	p.expression()
	if p.acceptKw("as") {
		t := p.starTarget()
		if !pyastValidTarget(t, true) {
			pyastBail()
		}
		if !p.isOp(",") && !p.isOp(")") && !p.isOp(":") {
			pyastBail()
		}
	}
}

func (p *pyastParser) tryStmt() {
	p.expectKw("try")
	p.expectOp(":")
	p.block()
	if p.isKw("finally") {
		p.advance()
		p.expectOp(":")
		p.block()
		return
	}
	if !p.isKw("except") {
		pyastBail()
	}
	star := p.at(1).kind == pyastTkOp && p.at(1).text == "*"
	for p.isKw("except") {
		isStar := p.at(1).kind == pyastTkOp && p.at(1).text == "*"
		if isStar != star {
			pyastBail() // except and except* cannot be mixed
		}
		p.advance()
		if isStar {
			p.advance()
			p.expression()
		} else if !p.isOp(":") {
			p.expression()
		}
		if p.acceptKw("as") {
			p.expectName()
		}
		p.expectOp(":")
		p.block()
	}
	if p.acceptKw("else") {
		p.expectOp(":")
		p.block()
	}
	if p.acceptKw("finally") {
		p.expectOp(":")
		p.block()
	}
}

func (p *pyastParser) typeParams() []pyastTypeParam {
	p.expectOp("[")
	var out []pyastTypeParam
	for {
		switch {
		case p.acceptOp("*"):
			out = append(out, pyastTypeParam{kind: pyastTpTuple, name: p.expectName()})
		case p.acceptOp("**"):
			out = append(out, pyastTypeParam{kind: pyastTpSpec, name: p.expectName()})
		default:
			tp := pyastTypeParam{kind: pyastTpVar, name: p.expectName()}
			if p.acceptOp(":") {
				tp.bound = p.expression()
			}
			out = append(out, tp)
		}
		if !p.acceptOp(",") {
			break
		}
		if p.isOp("]") {
			break
		}
	}
	p.expectOp("]")
	return out
}

// simpleStmts parses simple_stmt (';' simple_stmt)* [';'] NEWLINE.
func (p *pyastParser) simpleStmts() {
	for {
		p.simpleStmt()
		if p.acceptOp(";") {
			if p.peek().kind == pyastTkNewline {
				p.advance()
				return
			}
			continue
		}
		p.expectNewline()
		return
	}
}

func (p *pyastParser) simpleStmt() {
	t := p.peek()
	if t.kind == pyastTkName && t.text == "type" {
		n1, n2 := p.at(1), p.at(2)
		if n1.kind == pyastTkName && n2.kind == pyastTkOp && (n2.text == "=" || n2.text == "[") {
			p.advance()
			p.expectName()
			if p.isOp("[") {
				p.typeParams()
			}
			p.expectOp("=")
			p.expression()
			return
		}
	}
	if t.kind == pyastTkKeyword {
		switch t.text {
		case "return":
			p.advance()
			if p.canStartExpr(true) {
				p.starExpressions()
			}
		case "import":
			p.importName()
		case "from":
			p.importFrom()
		case "raise":
			p.advance()
			if p.canStartExpr(false) {
				p.expression()
				if p.acceptKw("from") {
					p.expression()
				}
			}
		case "pass", "break", "continue":
			p.advance()
		case "del":
			p.advance()
			p.delTargets()
		case "yield":
			p.yieldExpr()
		case "assert":
			p.advance()
			p.expression()
			if p.acceptOp(",") {
				p.expression()
			}
		case "global", "nonlocal":
			p.advance()
			p.expectName()
			for p.acceptOp(",") {
				p.expectName()
			}
		default:
			p.exprStmt()
		}
		return
	}
	p.exprStmt()
}

func (p *pyastParser) delTargets() {
	var elts []*pyastExpr
	for {
		elts = append(elts, p.bitOr())
		if !p.acceptOp(",") {
			break
		}
		if !p.canStartExpr(false) {
			break
		}
	}
	for _, e := range elts {
		if !pyastValidDel(e) {
			pyastBail()
		}
	}
	if k := p.peek().kind; k != pyastTkNewline && !p.isOp(";") {
		pyastBail()
	}
}

func (p *pyastParser) importName() {
	p.expectKw("import")
	for {
		p.dottedName()
		if p.acceptKw("as") {
			p.expectName()
		}
		if !p.acceptOp(",") {
			return
		}
	}
}

func (p *pyastParser) dottedName() {
	p.expectName()
	for p.acceptOp(".") {
		p.expectName()
	}
}

func (p *pyastParser) importFrom() {
	p.expectKw("from")
	dots := 0
	for {
		if p.acceptOp(".") {
			dots++
		} else if p.acceptOp("...") {
			dots += 3
		} else {
			break
		}
	}
	if p.peek().kind == pyastTkName {
		p.dottedName()
	} else if dots == 0 {
		pyastBail()
	}
	p.expectKw("import")
	if p.acceptOp("*") {
		return
	}
	paren := p.acceptOp("(")
	for {
		p.expectName()
		if p.acceptKw("as") {
			p.expectName()
		}
		if !p.acceptOp(",") {
			break
		}
		if paren && p.isOp(")") {
			break
		}
	}
	if paren {
		p.expectOp(")")
	}
}

// exprStmt parses an expression statement, an annotated assignment, an
// augmented assignment or a (chained) assignment.
func (p *pyastParser) exprStmt() {
	if p.isKw("yield") {
		p.yieldExpr()
		return
	}
	e := p.starExpressions()
	if p.acceptOp(":") {
		if !pyastValidSingleTarget(e) {
			pyastBail()
		}
		p.expression()
		if p.acceptOp("=") {
			p.annotatedRHS()
		}
		return
	}
	if t := p.peek(); t.kind == pyastTkOp && pyastAugOps[t.text] {
		p.advance()
		if !pyastValidSingleTarget(e) {
			pyastBail()
		}
		p.annotatedRHS()
		return
	}
	var rhs []*pyastExpr
	for p.acceptOp("=") {
		rhs = append(rhs, p.annotatedRHS())
	}
	if len(rhs) > 0 {
		targets := append([]*pyastExpr{e}, rhs[:len(rhs)-1]...)
		for _, t := range targets {
			if !pyastValidTarget(t, true) {
				pyastBail()
			}
		}
	}
}

var pyastAugOps = map[string]bool{
	"+=": true, "-=": true, "*=": true, "@=": true, "/=": true, "%=": true,
	"&=": true, "|=": true, "^=": true, "<<=": true, ">>=": true, "**=": true, "//=": true,
}

// annotatedRHS is yield_expr | star_expressions.
func (p *pyastParser) annotatedRHS() *pyastExpr {
	if p.isKw("yield") {
		return p.yieldExpr()
	}
	return p.starExpressions()
}

// ---------------------------------------------------------------- match

func (p *pyastParser) matchStmt() {
	p.advance() // match
	first := p.starNamedExpression()
	if p.isOp(",") {
		for p.acceptOp(",") {
			if p.isOp(":") {
				break
			}
			p.starNamedExpression()
		}
	} else if first.kind == pyastEStarred {
		pyastBail()
	}
	p.expectOp(":")
	p.expectNewline()
	if p.peek().kind != pyastTkIndent {
		pyastBail()
	}
	p.advance()
	n := 0
	for p.isName("case") {
		p.caseBlock()
		n++
	}
	if n == 0 {
		pyastBail()
	}
	if p.peek().kind != pyastTkDedent {
		pyastBail()
	}
	p.advance()
}

func (p *pyastParser) isName(s string) bool {
	t := p.peek()
	return t.kind == pyastTkName && t.text == s
}

func (p *pyastParser) caseBlock() {
	p.advance() // case
	p.patterns()
	if p.acceptKw("if") {
		p.namedExpression()
	}
	p.expectOp(":")
	p.block()
}

func (p *pyastParser) patterns() {
	isStar := p.maybeStarPattern()
	if !isStar && !p.isOp(",") {
		return
	}
	p.expectOp(",")
	if p.canStartPattern() {
		p.seqItems()
	}
}

// seqItems parses one or more maybe_star_patterns separated by commas, with
// an optional trailing comma.
func (p *pyastParser) seqItems() {
	p.maybeStarPattern()
	for p.acceptOp(",") {
		if !p.canStartPattern() {
			return
		}
		p.maybeStarPattern()
	}
}

func (p *pyastParser) canStartPattern() bool {
	t := p.peek()
	switch t.kind {
	case pyastTkName, pyastTkNumber, pyastTkString, pyastTkFStart:
		return true
	case pyastTkKeyword:
		return t.text == "None" || t.text == "True" || t.text == "False"
	case pyastTkOp:
		switch t.text {
		case "-", "(", "[", "{", "*":
			return true
		}
	}
	return false
}

// maybeStarPattern parses "*name" / "*_" (returns true) or a pattern.
func (p *pyastParser) maybeStarPattern() bool {
	if p.acceptOp("*") {
		p.expectName()
		return true
	}
	p.pattern()
	return false
}

func (p *pyastParser) pattern() {
	p.orPattern()
	if p.acceptKw("as") {
		if p.peek().kind != pyastTkName || p.peek().text == "_" {
			pyastBail()
		}
		p.advance()
	}
}

func (p *pyastParser) orPattern() {
	p.closedPattern()
	for p.acceptOp("|") {
		p.closedPattern()
	}
}

func (p *pyastParser) closedPattern() {
	t := p.peek()
	switch t.kind {
	case pyastTkNumber, pyastTkString, pyastTkFStart:
		p.literalPattern()
		return
	case pyastTkKeyword:
		if t.text == "None" || t.text == "True" || t.text == "False" {
			p.advance()
			return
		}
		pyastBail()
	case pyastTkName:
		// "_" is the wildcard and cannot start a dotted or class name
		if t.text == "_" && p.at(1).kind == pyastTkOp && (p.at(1).text == "." || p.at(1).text == "(") {
			pyastBail()
		}
		p.advance()
		for p.acceptOp(".") {
			p.expectName()
		}
		if p.isOp("(") {
			p.classPatternArgs()
		}
		return
	case pyastTkOp:
		switch t.text {
		case "-":
			p.literalPattern()
			return
		case "(":
			p.advance()
			if p.acceptOp(")") {
				return
			}
			isStar := p.maybeStarPattern()
			if p.isOp(",") {
				p.advance()
				if p.canStartPattern() {
					p.seqItems()
				}
				p.expectOp(")")
				return
			}
			if isStar {
				pyastBail()
			}
			p.expectOp(")")
			return
		case "[":
			p.advance()
			if p.acceptOp("]") {
				return
			}
			p.seqItems()
			p.expectOp("]")
			return
		case "{":
			p.advance()
			p.mappingBody()
			return
		}
	}
	pyastBail()
}

// literalPattern parses a signed number, a complex number (real part must be
// real, imaginary part must be imaginary) or a string literal.
func (p *pyastParser) literalPattern() {
	if p.peek().kind == pyastTkString || p.peek().kind == pyastTkFStart {
		p.strings()
		return
	}
	p.acceptOp("-")
	if p.peek().kind != pyastTkNumber {
		pyastBail()
	}
	imag := pyastIsImagLiteral(p.advance().text)
	if p.isOp("+") || p.isOp("-") {
		if imag {
			pyastBail()
		}
		p.advance()
		if p.peek().kind != pyastTkNumber || !pyastIsImagLiteral(p.advance().text) {
			pyastBail()
		}
	}
}

func pyastIsImagLiteral(text string) bool {
	last := text[len(text)-1]
	return last == 'j' || last == 'J'
}

func (p *pyastParser) mappingBody() {
	if p.acceptOp("}") {
		return
	}
	for {
		if p.acceptOp("**") {
			if p.peek().kind != pyastTkName || p.peek().text == "_" {
				pyastBail()
			}
			p.advance()
			p.acceptOp(",")
			p.expectOp("}")
			return
		}
		t := p.peek()
		switch {
		case t.kind == pyastTkName:
			if t.text == "_" {
				pyastBail()
			}
			p.advance()
			dotted := false
			for p.acceptOp(".") {
				p.expectName()
				dotted = true
			}
			if !dotted {
				pyastBail()
			}
		case t.kind == pyastTkNumber, t.kind == pyastTkString, t.kind == pyastTkFStart, t.kind == pyastTkOp && t.text == "-":
			p.literalPattern()
		case t.kind == pyastTkKeyword && (t.text == "None" || t.text == "True" || t.text == "False"):
			p.advance()
		default:
			pyastBail()
		}
		p.expectOp(":")
		p.pattern()
		if !p.acceptOp(",") {
			break
		}
		if p.isOp("}") {
			break
		}
	}
	p.expectOp("}")
}

func (p *pyastParser) classPatternArgs() {
	p.expectOp("(")
	if p.acceptOp(")") {
		return
	}
	sawKw := false
	for {
		if p.peek().kind == pyastTkName && p.at(1).kind == pyastTkOp && p.at(1).text == "=" {
			p.advance()
			p.advance()
			p.pattern()
			sawKw = true
		} else {
			if sawKw {
				pyastBail()
			}
			p.pattern()
		}
		if !p.acceptOp(",") {
			break
		}
		if p.isOp(")") {
			break
		}
	}
	p.expectOp(")")
}

// ---------------------------------------------------------------- parameters

// params parses a def (lam=false, ends before ')') or lambda parameter list
// (lam=true, ends before ':'), following the PEG parameters rules.
func (p *pyastParser) params(lam bool) *pyastArguments {
	a := &pyastArguments{}
	closer := func() bool {
		if lam {
			return p.isOp(":")
		}
		return p.isOp(")")
	}
	seenDefault, seenStar, seenSlash, seenKw, needKw := false, false, false, false, false
	count := 0
	for !closer() {
		if seenKw {
			pyastBail()
		}
		if needKw && p.peek().kind != pyastTkName {
			pyastBail()
		}
		switch {
		case p.acceptOp("**"):
			a.kwarg = p.param(lam, true)
			seenKw = true
		case p.acceptOp("*"):
			if seenStar {
				pyastBail()
			}
			seenStar = true
			if p.isOp(",") {
				needKw = true
			} else {
				a.vararg = p.starParam(lam)
			}
		case p.isOp("/"):
			if seenSlash || seenStar || count == 0 {
				pyastBail()
			}
			p.advance()
			seenSlash = true
			a.posonly = a.args
			a.args = nil
		default:
			needKw = false
			ar := p.param(lam, true)
			var def *pyastExpr
			if p.acceptOp("=") {
				def = p.expression()
			}
			if seenStar {
				a.kwonly = append(a.kwonly, ar)
				a.kwDefaults = append(a.kwDefaults, def)
			} else {
				if def != nil {
					seenDefault = true
					a.defaults = append(a.defaults, def)
				} else if seenDefault {
					pyastBail() // non-default argument follows default argument
				}
				a.args = append(a.args, ar)
			}
			count++
		}
		if !p.acceptOp(",") {
			if !closer() {
				pyastBail()
			}
			break
		}
	}
	if needKw {
		pyastBail()
	}
	return a
}

// param parses NAME [':' expression]; annotations are not allowed in lambdas.
func (p *pyastParser) param(lam, allowAnn bool) *pyastArg {
	ar := &pyastArg{name: p.expectName()}
	if !lam && allowAnn && p.acceptOp(":") {
		ar.ann = p.expression()
	}
	return ar
}

// starParam parses the name of *args, whose annotation may be a star_expression.
func (p *pyastParser) starParam(lam bool) *pyastArg {
	ar := &pyastArg{name: p.expectName()}
	if !lam && p.acceptOp(":") {
		if p.acceptOp("*") {
			ar.ann = &pyastExpr{kind: pyastEStarred, a: p.bitOr()}
		} else {
			ar.ann = p.expression()
		}
	}
	return ar
}

// ---------------------------------------------------------------- targets

func (p *pyastParser) canStartTarget() bool {
	t := p.peek()
	if t.kind == pyastTkName {
		return true
	}
	return t.kind == pyastTkOp && (t.text == "(" || t.text == "[" || t.text == "*")
}

func (p *pyastParser) starTarget() *pyastExpr {
	if p.acceptOp("*") {
		return &pyastExpr{kind: pyastEStarred, a: p.bitOr()}
	}
	return p.bitOr()
}

// starTargets parses star_targets (the 'for' and comprehension form) and
// validates it.
func (p *pyastParser) starTargets() *pyastExpr {
	var elts []*pyastExpr
	trailing := false
	for {
		elts = append(elts, p.starTarget())
		if !p.acceptOp(",") {
			break
		}
		if !p.canStartTarget() {
			trailing = true
			break
		}
	}
	var t *pyastExpr
	if len(elts) == 1 && !trailing {
		t = elts[0]
	} else {
		t = &pyastExpr{kind: pyastETuple, elts: elts}
	}
	if !pyastValidTarget(t, true) {
		pyastBail()
	}
	return t
}

// validTarget mirrors star_targets: names, attributes, subscripts, starred
// (not nested), and tuples or lists of targets.
func pyastValidTarget(e *pyastExpr, allowStar bool) bool {
	switch e.kind {
	case pyastEName, pyastEAttr, pyastESubscript:
		return true
	case pyastEStarred:
		return allowStar && pyastValidTarget(e.a, false)
	case pyastETuple, pyastEList:
		for _, x := range e.elts {
			if !pyastValidTarget(x, true) {
				return false
			}
		}
		return true
	}
	return false
}

// validDel mirrors del_targets (no starred).
func pyastValidDel(e *pyastExpr) bool {
	switch e.kind {
	case pyastEName, pyastEAttr, pyastESubscript:
		return true
	case pyastETuple, pyastEList:
		for _, x := range e.elts {
			if !pyastValidDel(x) {
				return false
			}
		}
		return true
	}
	return false
}

// validSingleTarget mirrors single_target used by augmented and annotated
// assignments: a name, attribute or subscript.
func pyastValidSingleTarget(e *pyastExpr) bool {
	switch e.kind {
	case pyastEName, pyastEAttr, pyastESubscript:
		return true
	}
	return false
}

// ---------------------------------------------------------------- expressions

// canStartExpr reports whether the next token can begin an expression
// (star=true also admits '*', as star_expressions does).
func (p *pyastParser) canStartExpr(star bool) bool {
	t := p.peek()
	switch t.kind {
	case pyastTkName, pyastTkNumber, pyastTkString, pyastTkFStart:
		return true
	case pyastTkKeyword:
		switch t.text {
		case "not", "lambda", "await", "None", "True", "False":
			return true
		}
	case pyastTkOp:
		switch t.text {
		case "(", "[", "{", "-", "+", "~", "...":
			return true
		case "*":
			return star
		}
	}
	return false
}

func (p *pyastParser) yieldExpr() *pyastExpr {
	p.expectKw("yield")
	if p.acceptKw("from") {
		return &pyastExpr{kind: pyastEYieldFrom, a: p.expression()}
	}
	if p.canStartExpr(true) {
		return &pyastExpr{kind: pyastEYield, a: p.starExpressions()}
	}
	return &pyastExpr{kind: pyastEYield}
}

func (p *pyastParser) starExpressions() *pyastExpr {
	e := p.starExpression()
	if !p.isOp(",") {
		return e
	}
	elts := []*pyastExpr{e}
	for p.acceptOp(",") {
		if !p.canStartExpr(true) {
			break
		}
		elts = append(elts, p.starExpression())
	}
	return &pyastExpr{kind: pyastETuple, elts: elts}
}

func (p *pyastParser) starExpression() *pyastExpr {
	if p.acceptOp("*") {
		return &pyastExpr{kind: pyastEStarred, a: p.bitOr()}
	}
	return p.expression()
}

func (p *pyastParser) starNamedExpression() *pyastExpr {
	if p.acceptOp("*") {
		return &pyastExpr{kind: pyastEStarred, a: p.bitOr()}
	}
	return p.namedExpression()
}

// namedExpression is assignment_expression | expression !':='.
func (p *pyastParser) namedExpression() *pyastExpr {
	if p.peek().kind == pyastTkName && p.at(1).kind == pyastTkOp && p.at(1).text == ":=" {
		name := p.expectName()
		p.advance()
		return &pyastExpr{kind: pyastENamedExpr, a: &pyastExpr{kind: pyastEName, name: name}, b: p.expression()}
	}
	e := p.expression()
	if p.isOp(":=") {
		pyastBail()
	}
	return e
}

func (p *pyastParser) expression() *pyastExpr {
	if p.isKw("lambda") {
		return p.lambdef()
	}
	e := p.disjunction()
	if p.acceptKw("if") {
		test := p.disjunction()
		p.expectKw("else")
		orelse := p.expression()
		return &pyastExpr{kind: pyastEIfExp, a: e, b: test, c: orelse}
	}
	return e
}

func (p *pyastParser) lambdef() *pyastExpr {
	p.expectKw("lambda")
	args := p.params(true)
	p.expectOp(":")
	return &pyastExpr{kind: pyastELambda, args: args, a: p.expression()}
}

func (p *pyastParser) disjunction() *pyastExpr {
	e := p.conjunction()
	if !p.isKw("or") {
		return e
	}
	vals := []*pyastExpr{e}
	for p.acceptKw("or") {
		vals = append(vals, p.conjunction())
	}
	return &pyastExpr{kind: pyastEBoolOp, op: "or", elts: vals}
}

func (p *pyastParser) conjunction() *pyastExpr {
	e := p.inversion()
	if !p.isKw("and") {
		return e
	}
	vals := []*pyastExpr{e}
	for p.acceptKw("and") {
		vals = append(vals, p.inversion())
	}
	return &pyastExpr{kind: pyastEBoolOp, op: "and", elts: vals}
}

func (p *pyastParser) inversion() *pyastExpr {
	if p.acceptKw("not") {
		return &pyastExpr{kind: pyastEUnary, op: "not", a: p.inversion()}
	}
	return p.comparison()
}

func (p *pyastParser) comparison() *pyastExpr {
	left := p.bitOr()
	var ops []string
	var comps []*pyastExpr
	for {
		op := p.compOp()
		if op == "" {
			break
		}
		ops = append(ops, op)
		comps = append(comps, p.bitOr())
	}
	if len(ops) == 0 {
		return left
	}
	return &pyastExpr{kind: pyastECompare, a: left, ops: ops, elts: comps}
}

func (p *pyastParser) compOp() string {
	t := p.peek()
	if t.kind == pyastTkOp {
		switch t.text {
		case "==", "!=", "<=", "<", ">=", ">":
			p.advance()
			return t.text
		}
		return ""
	}
	if t.kind != pyastTkKeyword {
		return ""
	}
	switch t.text {
	case "in":
		p.advance()
		return "in"
	case "is":
		p.advance()
		if p.acceptKw("not") {
			return "is not"
		}
		return "is"
	case "not":
		if n := p.at(1); n.kind == pyastTkKeyword && n.text == "in" {
			p.advance()
			p.advance()
			return "not in"
		}
	}
	return ""
}

var pyastBinLevels = [][]string{{"|"}, {"^"}, {"&"}, {"<<", ">>"}, {"+", "-"}, {"*", "/", "//", "%", "@"}}

func (p *pyastParser) bitOr() *pyastExpr { return p.binLevel(0) }

func (p *pyastParser) binLevel(level int) *pyastExpr {
	if level == len(pyastBinLevels) {
		return p.factor()
	}
	e := p.binLevel(level + 1)
	for {
		op := ""
		if t := p.peek(); t.kind == pyastTkOp {
			for _, cand := range pyastBinLevels[level] {
				if t.text == cand {
					op = cand
				}
			}
		}
		if op == "" {
			return e
		}
		p.advance()
		e = &pyastExpr{kind: pyastEBinOp, a: e, op: op, b: p.binLevel(level + 1)}
	}
}

func (p *pyastParser) factor() *pyastExpr {
	if t := p.peek(); t.kind == pyastTkOp && (t.text == "+" || t.text == "-" || t.text == "~") {
		p.advance()
		return &pyastExpr{kind: pyastEUnary, op: t.text, a: p.factor()}
	}
	return p.power()
}

func (p *pyastParser) power() *pyastExpr {
	base := p.awaitPrimary()
	if p.acceptOp("**") {
		return &pyastExpr{kind: pyastEBinOp, a: base, op: "**", b: p.factor()}
	}
	return base
}

func (p *pyastParser) awaitPrimary() *pyastExpr {
	if p.acceptKw("await") {
		return &pyastExpr{kind: pyastEAwait, a: p.primary()}
	}
	return p.primary()
}

func (p *pyastParser) primary() *pyastExpr {
	e := p.atom()
	for {
		switch {
		case p.acceptOp("."):
			e = &pyastExpr{kind: pyastEAttr, a: e, name: p.expectName()}
		case p.acceptOp("("):
			args, kws := p.callArgs(true)
			e = &pyastExpr{kind: pyastECall, a: e, elts: args, kws: kws}
		case p.acceptOp("["):
			s := p.slices()
			p.expectOp("]")
			e = &pyastExpr{kind: pyastESubscript, a: e, b: s}
		default:
			return e
		}
	}
}

func (p *pyastParser) atom() *pyastExpr {
	t := p.peek()
	switch t.kind {
	case pyastTkName:
		p.advance()
		return &pyastExpr{kind: pyastEName, name: pyastIdentifier(t.text)}
	case pyastTkKeyword:
		switch t.text {
		case "None":
			p.advance()
			return pyastConstExpr(&pyastConstant{kind: pyastCNone})
		case "True":
			p.advance()
			return pyastConstExpr(&pyastConstant{kind: pyastCTrue})
		case "False":
			p.advance()
			return pyastConstExpr(&pyastConstant{kind: pyastCFalse})
		}
	case pyastTkNumber:
		p.advance()
		return pyastNumberExpr(t.text)
	case pyastTkString, pyastTkFStart:
		return p.strings()
	case pyastTkOp:
		switch t.text {
		case "...":
			p.advance()
			return pyastConstExpr(&pyastConstant{kind: pyastCEllipsis})
		case "(":
			return p.parenAtom()
		case "[":
			return p.listAtom()
		case "{":
			return p.braceAtom()
		}
	}
	pyastBail()
	return nil
}

func pyastConstExpr(c *pyastConstant) *pyastExpr { return &pyastExpr{kind: pyastEConst, cnst: c} }

func (p *pyastParser) parenAtom() *pyastExpr {
	p.advance() // (
	if p.acceptOp(")") {
		return &pyastExpr{kind: pyastETuple}
	}
	if p.isKw("yield") {
		e := p.yieldExpr()
		p.expectOp(")")
		return e
	}
	first := p.starNamedExpression()
	if p.isKw("for") || p.isKw("async") {
		if first.kind == pyastEStarred {
			pyastBail()
		}
		clauses := p.forIfClauses()
		p.expectOp(")")
		return &pyastExpr{kind: pyastEGenExp, a: first, comps: clauses}
	}
	if p.acceptOp(")") {
		if first.kind == pyastEStarred {
			pyastBail()
		}
		return first
	}
	p.expectOp(",")
	elts := []*pyastExpr{first}
	for !p.isOp(")") {
		elts = append(elts, p.starNamedExpression())
		if !p.acceptOp(",") {
			break
		}
	}
	p.expectOp(")")
	return &pyastExpr{kind: pyastETuple, elts: elts}
}

func (p *pyastParser) listAtom() *pyastExpr {
	p.advance() // [
	if p.acceptOp("]") {
		return &pyastExpr{kind: pyastEList}
	}
	first := p.starNamedExpression()
	if p.isKw("for") || p.isKw("async") {
		if first.kind == pyastEStarred {
			pyastBail()
		}
		clauses := p.forIfClauses()
		p.expectOp("]")
		return &pyastExpr{kind: pyastEListComp, a: first, comps: clauses}
	}
	elts := []*pyastExpr{first}
	for p.acceptOp(",") {
		if p.isOp("]") {
			break
		}
		elts = append(elts, p.starNamedExpression())
	}
	p.expectOp("]")
	return &pyastExpr{kind: pyastEList, elts: elts}
}

func (p *pyastParser) braceAtom() *pyastExpr {
	p.advance() // {
	if p.acceptOp("}") {
		return &pyastExpr{kind: pyastEDict}
	}
	if p.acceptOp("**") {
		return p.dictRest([]*pyastExpr{nil}, []*pyastExpr{p.bitOr()})
	}
	if p.isOp("*") {
		return p.finishSet(p.starNamedExpression())
	}
	if p.peek().kind == pyastTkName && p.at(1).kind == pyastTkOp && p.at(1).text == ":=" {
		return p.finishSet(p.namedExpression())
	}
	first := p.expression()
	if p.acceptOp(":") {
		v := p.expression()
		if p.isKw("for") || p.isKw("async") {
			clauses := p.forIfClauses()
			p.expectOp("}")
			return &pyastExpr{kind: pyastEDictComp, a: first, b: v, comps: clauses}
		}
		return p.dictRest([]*pyastExpr{first}, []*pyastExpr{v})
	}
	if p.isOp(":=") {
		pyastBail()
	}
	return p.finishSet(first)
}

// finishSet handles a set display or set comprehension whose first element
// has been parsed.
func (p *pyastParser) finishSet(first *pyastExpr) *pyastExpr {
	if p.isKw("for") || p.isKw("async") {
		if first.kind == pyastEStarred {
			pyastBail()
		}
		clauses := p.forIfClauses()
		p.expectOp("}")
		return &pyastExpr{kind: pyastESetComp, a: first, comps: clauses}
	}
	elts := []*pyastExpr{first}
	for p.acceptOp(",") {
		if p.isOp("}") {
			break
		}
		elts = append(elts, p.starNamedExpression())
	}
	p.expectOp("}")
	return &pyastExpr{kind: pyastESet, elts: elts}
}

func (p *pyastParser) dictRest(keys, vals []*pyastExpr) *pyastExpr {
	for p.acceptOp(",") {
		if p.isOp("}") {
			break
		}
		if p.acceptOp("**") {
			keys = append(keys, nil)
			vals = append(vals, p.bitOr())
			continue
		}
		k := p.expression()
		p.expectOp(":")
		keys = append(keys, k)
		vals = append(vals, p.expression())
	}
	p.expectOp("}")
	return &pyastExpr{kind: pyastEDict, keys: keys, elts: vals}
}

func (p *pyastParser) forIfClauses() []*pyastComprehension {
	var out []*pyastComprehension
	for p.isKw("for") || (p.isKw("async") && p.at(1).kind == pyastTkKeyword && p.at(1).text == "for") {
		c := &pyastComprehension{}
		if p.acceptKw("async") {
			c.isAsync = true
		}
		p.expectKw("for")
		c.target = p.starTargets()
		p.expectKw("in")
		c.iter = p.disjunction()
		for p.acceptKw("if") {
			c.ifs = append(c.ifs, p.disjunction())
		}
		out = append(out, c)
	}
	if len(out) == 0 {
		pyastBail()
	}
	return out
}

// slices parses the content of a subscript (without the brackets).
func (p *pyastParser) slices() *pyastExpr {
	first, isStar := p.sliceItem()
	if !p.isOp(",") && !isStar {
		return first
	}
	elts := []*pyastExpr{first}
	for p.acceptOp(",") {
		if p.isOp("]") {
			break
		}
		item, _ := p.sliceItem()
		elts = append(elts, item)
	}
	return &pyastExpr{kind: pyastETuple, elts: elts}
}

func (p *pyastParser) sliceItem() (*pyastExpr, bool) {
	if p.acceptOp("*") {
		return &pyastExpr{kind: pyastEStarred, a: p.expression()}, true
	}
	if p.isOp(":") {
		return p.sliceRest(nil), false
	}
	if p.peek().kind == pyastTkName && p.at(1).kind == pyastTkOp && p.at(1).text == ":=" {
		return p.namedExpression(), false
	}
	e := p.expression()
	if p.isOp(":") {
		return p.sliceRest(e), false
	}
	if p.isOp(":=") {
		pyastBail()
	}
	return e, false
}

func (p *pyastParser) sliceRest(lower *pyastExpr) *pyastExpr {
	p.expectOp(":")
	s := &pyastExpr{kind: pyastESlice, a: lower}
	if p.canStartExpr(false) {
		s.b = p.expression()
	}
	if p.acceptOp(":") && p.canStartExpr(false) {
		s.c = p.expression()
	}
	return s
}

// callArgs parses call or class arguments after '(' and consumes ')'. With
// allowGen, a sole bare generator argument is accepted (only in calls).
func (p *pyastParser) callArgs(allowGen bool) ([]*pyastExpr, []*pyastKeyword) {
	var args []*pyastExpr
	var kws []*pyastKeyword
	seenKw, seenDStar := false, false
	for !p.isOp(")") {
		t := p.peek()
		switch {
		case p.acceptOp("*"):
			if seenDStar {
				pyastBail()
			}
			args = append(args, &pyastExpr{kind: pyastEStarred, a: p.expression()})
		case p.acceptOp("**"):
			kws = append(kws, &pyastKeyword{value: p.expression()})
			seenKw, seenDStar = true, true
		case t.kind == pyastTkName && p.at(1).kind == pyastTkOp && p.at(1).text == "=":
			p.advance()
			p.advance()
			kws = append(kws, &pyastKeyword{hasArg: true, argName: pyastIdentifier(t.text), value: p.expression()})
			seenKw = true
		default:
			if seenKw {
				pyastBail()
			}
			e := p.namedExpression()
			if p.isOp("=") {
				pyastBail()
			}
			if p.isKw("for") || p.isKw("async") {
				if !allowGen || len(args) > 0 {
					pyastBail()
				}
				// a bare generator argument must be the only argument
				clauses := p.forIfClauses()
				args = append(args, &pyastExpr{kind: pyastEGenExp, a: e, comps: clauses})
				p.expectOp(")")
				return args, kws
			}
			args = append(args, e)
		}
		if !p.acceptOp(",") {
			break
		}
	}
	p.expectOp(")")
	return args, kws
}

func (p *pyastParser) strings() *pyastExpr {
	var pieces []pyastStrPiece
loop:
	for {
		t := p.peek()
		switch t.kind {
		case pyastTkString:
			p.advance()
			pieces = append(pieces, pyastDecodeStringToken(t.text))
		case pyastTkFStart:
			pieces = append(pieces, pyastStrPiece{isF: true, parts: p.fstring()})
		default:
			break loop
		}
	}
	anyBytes, anyStr := false, false
	for _, pc := range pieces {
		if pc.isBytes {
			anyBytes = true
		} else {
			anyStr = true
		}
	}
	if anyBytes && anyStr {
		pyastBail() // cannot mix bytes and nonbytes literals
	}
	if anyBytes {
		var b []byte
		for _, pc := range pieces {
			b = append(b, pc.b...)
		}
		return pyastConstExpr(&pyastConstant{kind: pyastCBytes, b: b})
	}
	anyF := false
	for _, pc := range pieces {
		if pc.isF {
			anyF = true
		}
	}
	if !anyF {
		var s []rune
		for _, pc := range pieces {
			s = append(s, pc.s...)
		}
		return pyastConstExpr(&pyastConstant{kind: pyastCStr, s: s, uprefix: pieces[0].uprefix})
	}
	var parts []pyastJsPart
	add := func(r []rune) {
		if len(r) == 0 {
			return
		}
		if n := len(parts); n > 0 && parts[n-1].isConst {
			parts[n-1].s = append(append([]rune{}, parts[n-1].s...), r...)
			return
		}
		parts = append(parts, pyastJsPart{isConst: true, s: r})
	}
	for _, pc := range pieces {
		if pc.isF {
			for _, jp := range pc.parts {
				if jp.isConst {
					add(jp.s)
				} else {
					parts = append(parts, jp)
				}
			}
		} else {
			add(pc.s)
		}
	}
	return &pyastExpr{kind: pyastEJoined, parts: parts}
}

// strPiece is one string token or one whole f-string.
type pyastStrPiece struct {
	isBytes bool
	isF     bool
	uprefix bool
	s       []rune
	b       []byte
	parts   []pyastJsPart
}

func pyastDecodeStringToken(text string) pyastStrPiece {
	i := 0
	for i < len(text) && text[i] != '\'' && text[i] != '"' {
		i++
	}
	prefix := strings.ToLower(text[:i])
	q := text[i]
	ql := 1
	if len(text) >= i+3 && text[i+1] == q && text[i+2] == q {
		ql = 3
	}
	body := text[i+ql : len(text)-ql]
	raw := strings.Contains(prefix, "r")
	pc := pyastStrPiece{uprefix: strings.HasPrefix(text, "u")}
	if strings.Contains(prefix, "b") {
		pc.isBytes = true
		for _, r := range body {
			if r >= 0x80 {
				pyastBail() // bytes can only contain ASCII literal characters
			}
		}
		rs := pyastDecodeBody(body, raw, true, false)
		pc.b = make([]byte, len(rs))
		for i, r := range rs {
			pc.b[i] = byte(r)
		}
		return pc
	}
	pc.s = pyastDecodeBody(body, raw, false, false)
	return pc
}

// fstring parses an f-string whose FSTART token is the next token.
func (p *pyastParser) fstring() []pyastJsPart {
	start := p.advance()
	raw := strings.ContainsRune(start.text, 'r')
	var parts []pyastJsPart
	for {
		t := p.peek()
		switch {
		case t.kind == pyastTkFMid:
			p.advance()
			parts = append(parts, pyastJsPart{isConst: true, s: pyastDecodeBody(t.text, raw, false, true)})
		case t.kind == pyastTkOp && t.text == "{":
			parts = append(parts, p.replacementField(raw, false)...)
		case t.kind == pyastTkFEnd:
			p.advance()
			return parts
		default:
			pyastBail()
		}
	}
}

// replacementField parses '{' expr ['='] ['!' NAME] [':' spec] '}'. inSpec
// marks a field nested in a format spec.
func (p *pyastParser) replacementField(raw, inSpec bool) []pyastJsPart {
	open := p.advance()
	var value *pyastExpr
	if p.isKw("yield") {
		value = p.yieldExpr()
	} else {
		value = p.starExpressions()
	}
	var out []pyastJsPart
	debug := false
	var debugText []rune
	if p.isOp("=") {
		p.advance()
		if inSpec {
			// CPython builds a Constant without a value here and ast.parse
			// raises ValueError, which the distiller turns into the input.
			pyastBail()
		}
		debug = true
		debugText = pyastDecodeBody(p.src[open.end:p.peek().start], raw, false, false)
	}
	conv := -1
	if p.isOp("!") {
		bang := p.advance()
		t := p.peek()
		// the conversion must follow '!' directly, without blanks
		if t.kind != pyastTkName || t.start != bang.end || len(t.text) != 1 || !strings.ContainsAny(t.text, "sra") {
			pyastBail()
		}
		p.advance()
		conv = int(t.text[0])
	}
	hasSpec := false
	var spec []pyastJsPart
	if p.acceptOp(":") {
		hasSpec = true
		spec = []pyastJsPart{}
		for {
			t := p.peek()
			if t.kind == pyastTkFMid {
				p.advance()
				spec = append(spec, pyastJsPart{isConst: true, s: pyastDecodeBody(t.text, raw, false, true)})
				continue
			}
			if t.kind == pyastTkOp && t.text == "{" {
				spec = append(spec, p.replacementField(raw, true)...)
				continue
			}
			break
		}
	}
	p.expectOp("}")
	if debug {
		if conv == -1 && !hasSpec {
			conv = 'r'
		}
		out = append(out, pyastJsPart{isConst: true, s: debugText})
	}
	out = append(out, pyastJsPart{fv: &pyastFormatted{value: value, conv: conv, spec: spec, hasSpec: hasSpec}})
	return out
}

// numberExpr builds the Constant for a NUMBER token. Integers are kept as
// decimal strings, which is how repr() prints them.
func pyastNumberExpr(text string) *pyastExpr {
	t := strings.ReplaceAll(text, "_", "")
	if pyastIsImagLiteral(t) {
		return pyastConstExpr(&pyastConstant{kind: pyastCComplex, f: pyastParseFloatLit(t[:len(t)-1])})
	}
	if len(t) > 1 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X' || t[1] == 'o' || t[1] == 'O' || t[1] == 'b' || t[1] == 'B') {
		base := map[byte]int{'x': 16, 'X': 16, 'o': 8, 'O': 8, 'b': 2, 'B': 2}[t[1]]
		return pyastConstExpr(&pyastConstant{kind: pyastCInt, digits: pyastRadixDigits(t[2:], base)})
	}
	if strings.ContainsAny(t, ".eE") {
		return pyastConstExpr(&pyastConstant{kind: pyastCFloat, f: pyastParseFloatLit(t)})
	}
	d := strings.TrimLeft(t, "0")
	if d == "" {
		d = "0"
	}
	return pyastConstExpr(&pyastConstant{kind: pyastCInt, digits: d})
}

// decodeBody interprets the body of a string literal the way CPython's
// string_parser.c does. fmid marks an f-string literal fragment, where doubled
// braces collapse to a single brace.
func pyastDecodeBody(body string, raw, isBytes, fmid bool) []rune {
	rs := []rune(body)
	out := make([]rune, 0, len(rs))
	for i := 0; i < len(rs); {
		c := rs[i]
		if c != '\\' || raw {
			if fmid && (c == '{' || c == '}') && i+1 < len(rs) && rs[i+1] == c {
				out = append(out, c)
				i += 2
				continue
			}
			out = append(out, c)
			i++
			continue
		}
		if i+1 >= len(rs) {
			out = append(out, '\\')
			i++
			continue
		}
		n := rs[i+1]
		switch {
		case n == '\n':
			i += 2
		case n == '\\' || n == '\'' || n == '"':
			out = append(out, n)
			i += 2
		case n == 'a':
			out = append(out, 7)
			i += 2
		case n == 'b':
			out = append(out, 8)
			i += 2
		case n == 'f':
			out = append(out, 12)
			i += 2
		case n == 'n':
			out = append(out, 10)
			i += 2
		case n == 'r':
			out = append(out, 13)
			i += 2
		case n == 't':
			out = append(out, 9)
			i += 2
		case n == 'v':
			out = append(out, 11)
			i += 2
		case n >= '0' && n <= '7':
			v := 0
			j := i + 1
			for k := 0; k < 3 && j < len(rs) && rs[j] >= '0' && rs[j] <= '7'; k++ {
				v = v*8 + int(rs[j]-'0')
				j++
			}
			if isBytes {
				v &= 0xff // bytes keep only the low byte of an octal escape
			}
			out = append(out, rune(v))
			i = j
		case n == 'x':
			out = append(out, pyastHexEscape(rs, i+2, 2))
			i += 4
		case n == 'u' && !isBytes:
			out = append(out, pyastHexEscape(rs, i+2, 4))
			i += 6
		case n == 'U' && !isBytes:
			v := pyastHexEscape(rs, i+2, 8)
			if v > 0x10FFFF {
				pyastBail() // illegal Unicode character
			}
			out = append(out, v)
			i += 10
		case n == 'N' && !isBytes:
			// \N{name}: the name table is embedded (pyast_names.go). An
			// unknown name is a SyntaxError in CPython, so it is a bail here.
			if i+2 >= len(rs) || rs[i+2] != '{' {
				pyastBail()
			}
			j := i + 3
			for j < len(rs) && rs[j] != '}' {
				j++
			}
			if j >= len(rs) {
				pyastBail()
			}
			r, found, err := pyNameLookup(string(rs[i+3 : j]))
			if err != nil {
				panic(pyastDeclined{})
			}
			if !found {
				pyastBail()
			}
			out = append(out, r)
			i = j + 1
		default:
			// unknown escapes keep the backslash; the next character is
			// then processed as an ordinary one
			out = append(out, '\\')
			i++
		}
	}
	return out
}

func pyastHexEscape(rs []rune, start, count int) rune {
	if start+count > len(rs) {
		pyastBail()
	}
	var v rune
	for k := 0; k < count; k++ {
		c := rs[start+k]
		var d rune
		switch {
		case c >= '0' && c <= '9':
			d = c - '0'
		case c >= 'a' && c <= 'f':
			d = c - 'a' + 10
		case c >= 'A' && c <= 'F':
			d = c - 'A' + 10
		default:
			pyastBail()
		}
		v = v*16 + d
	}
	return v
}

// parseFloatLit converts a float literal (underscores already removed).
// Overflow yields +Inf, as in Python.
func pyastParseFloatLit(s string) float64 {
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		if ne, ok := err.(*strconv.NumError); ok && ne.Err == strconv.ErrRange {
			return f
		}
		pyastBail()
	}
	return f
}

// radixDigits converts a hex, octal or binary digit string to decimal.
func pyastRadixDigits(digits string, base int) string {
	n, ok := new(big.Int).SetString(digits, base)
	if !ok {
		pyastBail()
	}
	return n.String()
}

// isPrintable matches str.isprintable() for one code point: everything except
// the categories Cc, Cf, Cs, Co, Cn, Zl, Zp and Zs (ASCII space excepted).
func pyastIsPrintable(r rune) bool {
	if r == ' ' {
		return true
	}
	return unicode.IsPrint(r) && !pyastInRangePairs(pyastUnassignedPrintable[:], r)
}

// strRepr is repr() of a str value.
func pyastStrRepr(s []rune) string {
	quote := '\''
	hasSingle, hasDouble := false, false
	for _, r := range s {
		switch r {
		case '\'':
			hasSingle = true
		case '"':
			hasDouble = true
		}
	}
	if hasSingle && !hasDouble {
		quote = '"'
	}
	var b strings.Builder
	b.WriteRune(quote)
	for _, r := range s {
		switch {
		case r == quote || r == '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		case r == '\t':
			b.WriteString(`\t`)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r < 0x20 || r == 0x7f:
			fmt.Fprintf(&b, `\x%02x`, r)
		case r < 0x7f:
			b.WriteRune(r)
		case pyastIsPrintable(r):
			b.WriteRune(r)
		case r <= 0xff:
			fmt.Fprintf(&b, `\x%02x`, r)
		case r <= 0xffff:
			fmt.Fprintf(&b, `\u%04x`, r)
		default:
			fmt.Fprintf(&b, `\U%08x`, r)
		}
	}
	b.WriteRune(quote)
	return b.String()
}

// bytesRepr is repr() of a bytes value.
func pyastBytesRepr(s []byte) string {
	quote := byte('\'')
	hasSingle, hasDouble := false, false
	for _, c := range s {
		switch c {
		case '\'':
			hasSingle = true
		case '"':
			hasDouble = true
		}
	}
	if hasSingle && !hasDouble {
		quote = '"'
	}
	var b strings.Builder
	b.WriteString("b")
	b.WriteByte(quote)
	for _, c := range s {
		switch {
		case c == quote || c == '\\':
			b.WriteByte('\\')
			b.WriteByte(c)
		case c == '\t':
			b.WriteString(`\t`)
		case c == '\n':
			b.WriteString(`\n`)
		case c == '\r':
			b.WriteString(`\r`)
		case c < 0x20 || c >= 0x7f:
			fmt.Fprintf(&b, `\x%02x`, c)
		default:
			b.WriteByte(c)
		}
	}
	b.WriteByte(quote)
	return b.String()
}

// unicodeEscape is str.encode('unicode_escape') for one code point, used by
// the JoinedStr unparser for backslashes and non-printable characters.
func pyastUnicodeEscape(r rune) string {
	switch {
	case r == '\\':
		return `\\`
	case r == '\t':
		return `\t`
	case r == '\n':
		return `\n`
	case r == '\r':
		return `\r`
	case r < 0x100:
		return fmt.Sprintf(`\x%02x`, r)
	case r <= 0xffff:
		return fmt.Sprintf(`\u%04x`, r)
	}
	return fmt.Sprintf(`\U%08x`, r)
}

// floatRepr is repr() of a float. dotZero selects the float form (adds ".0"
// to integral values); complex parts use the form without it.
func pyastFloatRepr(f float64, dotZero bool) string {
	switch {
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	case math.IsNaN(f):
		return "nan"
	}
	sign := ""
	if math.Signbit(f) {
		sign = "-"
	}
	a := math.Abs(f)
	if a == 0 {
		if dotZero {
			return sign + "0.0"
		}
		return sign + "0"
	}
	s := strconv.FormatFloat(a, 'e', -1, 64)
	epos := strings.IndexByte(s, 'e')
	mant := s[:epos]
	exp, _ := strconv.Atoi(s[epos+1:])
	digits := strings.Replace(mant, ".", "", 1)
	decpt := exp + 1
	if decpt <= -4 || decpt > 16 {
		m := digits[:1]
		if len(digits) > 1 {
			m += "." + digits[1:]
		}
		esign := "+"
		if exp < 0 {
			esign = "-"
		}
		ea := exp
		if ea < 0 {
			ea = -ea
		}
		return fmt.Sprintf("%s%se%s%02d", sign, m, esign, ea)
	}
	var body string
	switch {
	case decpt <= 0:
		body = "0." + strings.Repeat("0", -decpt) + digits
	case decpt >= len(digits):
		body = digits + strings.Repeat("0", decpt-len(digits))
		if dotZero {
			body += ".0"
		}
	default:
		body = digits[:decpt] + "." + digits[decpt:]
	}
	return sign + body
}

// The tables below correct the Go unicode package, which tracks a newer
// Unicode release than CPython 3.12 (Unicode 15.0). Each list holds inclusive
// [lo, hi] pairs of code points that the Go tables accept but CPython 3.12
// rejects. They were generated by comparing against str.isprintable() and
// str.isidentifier() on CPython 3.12. Entries that are valid in older Go
// releases are harmless, since removing an unassigned code point changes
// nothing.

// unassignedPrintable lists code points that are printable in Go but unassigned in Unicode 15.0.
var pyastUnassignedPrintable = [...]rune{
	0x88F, 0x88F, 0x897, 0x897, 0xC5C, 0xC5C, 0xCDC, 0xCDC,
	0x1ACF, 0x1ADD, 0x1AE0, 0x1AEB, 0x1B4E, 0x1B4F, 0x1B7F, 0x1B7F,
	0x1C89, 0x1C8A, 0x20C1, 0x20C1, 0x2427, 0x2429, 0x2B96, 0x2B96,
	0x2FFC, 0x2FFF, 0x31E4, 0x31E5, 0x31EF, 0x31EF, 0xA7CB, 0xA7CF,
	0xA7D2, 0xA7D2, 0xA7D4, 0xA7D4, 0xA7DA, 0xA7DC, 0xA7F1, 0xA7F1,
	0xFBC3, 0xFBD2, 0xFD90, 0xFD91, 0xFDC8, 0xFDCE, 0x105C0, 0x105F3,
	0x10940, 0x10959, 0x10D40, 0x10D65, 0x10D69, 0x10D85, 0x10D8E, 0x10D8F,
	0x10EC2, 0x10EC7, 0x10ED0, 0x10ED8, 0x10EFA, 0x10EFC, 0x11380, 0x11389,
	0x1138B, 0x1138B, 0x1138E, 0x1138E, 0x11390, 0x113B5, 0x113B7, 0x113C0,
	0x113C2, 0x113C2, 0x113C5, 0x113C5, 0x113C7, 0x113CA, 0x113CC, 0x113D5,
	0x113D7, 0x113D8, 0x113E1, 0x113E2, 0x116D0, 0x116E3, 0x11B60, 0x11B67,
	0x11BC0, 0x11BE1, 0x11BF0, 0x11BF9, 0x11DB0, 0x11DDB, 0x11DE0, 0x11DE9,
	0x11F5A, 0x11F5A, 0x13460, 0x143FA, 0x16100, 0x16139, 0x16D40, 0x16D79,
	0x16EA0, 0x16EB8, 0x16EBB, 0x16ED3, 0x16FF2, 0x16FF6, 0x187F8, 0x187FF,
	0x18CFF, 0x18CFF, 0x18D09, 0x18D1E, 0x18D80, 0x18DF2, 0x1CC00, 0x1CCFC,
	0x1CD00, 0x1CEB3, 0x1CEBA, 0x1CED0, 0x1CEE0, 0x1CEF0, 0x1E5D0, 0x1E5FA,
	0x1E5FF, 0x1E5FF, 0x1E6C0, 0x1E6DE, 0x1E6E0, 0x1E6F5, 0x1E6FE, 0x1E6FF,
	0x1F6D8, 0x1F6D8, 0x1F777, 0x1F77A, 0x1F8B2, 0x1F8BB, 0x1F8C0, 0x1F8C1,
	0x1F8D0, 0x1F8D8, 0x1FA54, 0x1FA57, 0x1FA89, 0x1FA8A, 0x1FA8E, 0x1FA8F,
	0x1FABE, 0x1FABE, 0x1FAC6, 0x1FAC6, 0x1FAC8, 0x1FAC8, 0x1FACD, 0x1FACD,
	0x1FADC, 0x1FADC, 0x1FADF, 0x1FADF, 0x1FAE9, 0x1FAEA, 0x1FAEF, 0x1FAEF,
	0x1FBCB, 0x1FBEF, 0x1FBFA, 0x1FBFA, 0x2B73A, 0x2B73F, 0x2CEA2, 0x2CEAD,
	0x2EBF0, 0x2EE5D, 0x323B0, 0x33479,
}

// xidStartExtra lists code points that Go accepts as XID_Start but CPython 3.12 does not.
var pyastXidStartExtra = [...]rune{
	0x37A, 0x37A, 0x88F, 0x88F, 0xC5C, 0xC5C, 0xCDC, 0xCDC,
	0xE33, 0xE33, 0xEB3, 0xEB3, 0x1C89, 0x1C8A, 0x309B, 0x309C,
	0xA7CB, 0xA7CF, 0xA7D2, 0xA7D2, 0xA7D4, 0xA7D4, 0xA7DA, 0xA7DC,
	0xA7F1, 0xA7F1, 0xFC5E, 0xFC63, 0xFDFA, 0xFDFB, 0xFE70, 0xFE70,
	0xFE72, 0xFE72, 0xFE74, 0xFE74, 0xFE76, 0xFE76, 0xFE78, 0xFE78,
	0xFE7A, 0xFE7A, 0xFE7C, 0xFE7C, 0xFE7E, 0xFE7E, 0xFF9E, 0xFF9F,
	0x105C0, 0x105F3, 0x10940, 0x10959, 0x10D4A, 0x10D65, 0x10D6F, 0x10D85,
	0x10EC2, 0x10EC7, 0x11380, 0x11389, 0x1138B, 0x1138B, 0x1138E, 0x1138E,
	0x11390, 0x113B5, 0x113B7, 0x113B7, 0x113D1, 0x113D1, 0x113D3, 0x113D3,
	0x11BC0, 0x11BE0, 0x11DB0, 0x11DDB, 0x13460, 0x143FA, 0x16100, 0x1611D,
	0x16D40, 0x16D6C, 0x16EA0, 0x16EB8, 0x16EBB, 0x16ED3, 0x16FF2, 0x16FF6,
	0x187F8, 0x187FF, 0x18CFF, 0x18CFF, 0x18D09, 0x18D1E, 0x18D80, 0x18DF2,
	0x1E5D0, 0x1E5ED, 0x1E5F0, 0x1E5F0, 0x1E6C0, 0x1E6DE, 0x1E6E0, 0x1E6E2,
	0x1E6E4, 0x1E6E5, 0x1E6E7, 0x1E6ED, 0x1E6F0, 0x1E6F4, 0x1E6FE, 0x1E6FF,
	0x2B73A, 0x2B73F, 0x2CEA2, 0x2CEAD, 0x2EBF0, 0x2EE5D, 0x323B0, 0x33479,
}

// xidContinueExtra lists code points that Go accepts as XID_Continue but CPython 3.12 does not.
var pyastXidContinueExtra = [...]rune{
	0x37A, 0x37A, 0x88F, 0x88F, 0x897, 0x897, 0xC5C, 0xC5C,
	0xCDC, 0xCDC, 0x1ACF, 0x1ADD, 0x1AE0, 0x1AEB, 0x1C89, 0x1C8A,
	0x200C, 0x200D, 0x309B, 0x309C, 0x30FB, 0x30FB, 0xA7CB, 0xA7CF,
	0xA7D2, 0xA7D2, 0xA7D4, 0xA7D4, 0xA7DA, 0xA7DC, 0xA7F1, 0xA7F1,
	0xFC5E, 0xFC63, 0xFDFA, 0xFDFB, 0xFE70, 0xFE70, 0xFE72, 0xFE72,
	0xFE74, 0xFE74, 0xFE76, 0xFE76, 0xFE78, 0xFE78, 0xFE7A, 0xFE7A,
	0xFE7C, 0xFE7C, 0xFE7E, 0xFE7E, 0xFF65, 0xFF65, 0x105C0, 0x105F3,
	0x10940, 0x10959, 0x10D40, 0x10D65, 0x10D69, 0x10D6D, 0x10D6F, 0x10D85,
	0x10EC2, 0x10EC7, 0x10EFA, 0x10EFC, 0x11380, 0x11389, 0x1138B, 0x1138B,
	0x1138E, 0x1138E, 0x11390, 0x113B5, 0x113B7, 0x113C0, 0x113C2, 0x113C2,
	0x113C5, 0x113C5, 0x113C7, 0x113CA, 0x113CC, 0x113D3, 0x113E1, 0x113E2,
	0x116D0, 0x116E3, 0x11B60, 0x11B67, 0x11BC0, 0x11BE0, 0x11BF0, 0x11BF9,
	0x11DB0, 0x11DDB, 0x11DE0, 0x11DE9, 0x11F5A, 0x11F5A, 0x13460, 0x143FA,
	0x16100, 0x16139, 0x16D40, 0x16D6C, 0x16D70, 0x16D79, 0x16EA0, 0x16EB8,
	0x16EBB, 0x16ED3, 0x16FF2, 0x16FF6, 0x187F8, 0x187FF, 0x18CFF, 0x18CFF,
	0x18D09, 0x18D1E, 0x18D80, 0x18DF2, 0x1CCF0, 0x1CCF9, 0x1E5D0, 0x1E5FA,
	0x1E6C0, 0x1E6DE, 0x1E6E0, 0x1E6F5, 0x1E6FE, 0x1E6FF, 0x2B73A, 0x2B73F,
	0x2CEA2, 0x2CEAD, 0x2EBF0, 0x2EE5D, 0x323B0, 0x33479,
}

// inRangePairs reports whether r lies in one of the inclusive [lo, hi] pairs.
func pyastInRangePairs(table []rune, r rune) bool {
	i := sort.Search(len(table)/2, func(i int) bool { return table[2*i+1] >= r })
	return i < len(table)/2 && table[2*i] <= r
}

// prec mirrors ast._Precedence (auto() numbering, BOR is an alias of EXPR).
type pyastPrec int

const (
	pyastPrecNamedExpr pyastPrec = iota + 1
	pyastPrecTuple
	pyastPrecYield
	pyastPrecTest
	pyastPrecOr
	pyastPrecAnd
	pyastPrecNot
	pyastPrecCmp
	pyastPrecExpr
	pyastPrecBXor
	pyastPrecBAnd
	pyastPrecShift
	pyastPrecArith
	pyastPrecTerm
	pyastPrecFactor
	pyastPrecPower
	pyastPrecAwait
	pyastPrecAtom
)

// next mirrors _Precedence.next(): one level up, saturating at ATOM.
func (p pyastPrec) next() pyastPrec {
	if p == pyastPrecAtom {
		return p
	}
	return p + 1
}

// exprStr is ast._Unparser.visit for an expression whose precedence (as set by
// its parent) is given. Children without an explicit precedence get TEST.
func pyastExprStr(e *pyastExpr, given pyastPrec) string {
	switch e.kind {
	case pyastEName:
		return e.name
	case pyastEConst:
		return pyastConstantStr(e.cnst)
	case pyastEJoined:
		return pyastJoinedStr(e.parts)
	case pyastEList:
		return "[" + pyastJoinExprs(e.elts, pyastPrecTest) + "]"
	case pyastETuple:
		items := pyastItemsView(e.elts)
		if len(e.elts) == 0 || given > pyastPrecTuple {
			return "(" + items + ")"
		}
		return items
	case pyastESet:
		if len(e.elts) == 0 {
			return "{*()}"
		}
		return "{" + pyastJoinExprs(e.elts, pyastPrecTest) + "}"
	case pyastEDict:
		var parts []string
		for i, v := range e.elts {
			if e.keys[i] == nil {
				parts = append(parts, "**"+pyastExprStr(v, pyastPrecExpr))
			} else {
				parts = append(parts, pyastExprStr(e.keys[i], pyastPrecTest)+": "+pyastExprStr(v, pyastPrecTest))
			}
		}
		return "{" + strings.Join(parts, ", ") + "}"
	case pyastEListComp:
		return "[" + pyastExprStr(e.a, pyastPrecTest) + pyastCompsStr(e.comps) + "]"
	case pyastEGenExp:
		return "(" + pyastExprStr(e.a, pyastPrecTest) + pyastCompsStr(e.comps) + ")"
	case pyastESetComp:
		return "{" + pyastExprStr(e.a, pyastPrecTest) + pyastCompsStr(e.comps) + "}"
	case pyastEDictComp:
		return "{" + pyastExprStr(e.a, pyastPrecTest) + ": " + pyastExprStr(e.b, pyastPrecTest) + pyastCompsStr(e.comps) + "}"
	case pyastEIfExp:
		body := pyastExprStr(e.a, pyastPrecOr) + " if " + pyastExprStr(e.b, pyastPrecOr) + " else " + pyastExprStr(e.c, pyastPrecTest)
		return pyastRequireParens(pyastPrecTest, given, body)
	case pyastEUnary:
		opPrec := pyastPrecFactor
		opText := e.op
		if e.op == "not" {
			opPrec = pyastPrecNot
			opText = "not "
		}
		return pyastRequireParens(opPrec, given, opText+pyastExprStr(e.a, opPrec))
	case pyastEBinOp:
		opPrec := pyastBinopPrec(e.op)
		left, right := opPrec, opPrec.next()
		if e.op == "**" {
			left, right = opPrec.next(), opPrec
		}
		body := pyastExprStr(e.a, left) + " " + e.op + " " + pyastExprStr(e.b, right)
		return pyastRequireParens(opPrec, given, body)
	case pyastECompare:
		var b strings.Builder
		b.WriteString(pyastExprStr(e.a, pyastPrecCmp.next()))
		for i, op := range e.ops {
			b.WriteString(" " + op + " ")
			b.WriteString(pyastExprStr(e.elts[i], pyastPrecCmp.next()))
		}
		return pyastRequireParens(pyastPrecCmp, given, b.String())
	case pyastEBoolOp:
		opPrec := pyastPrecAnd
		if e.op == "or" {
			opPrec = pyastPrecOr
		}
		cur := opPrec
		var parts []string
		for _, v := range e.elts {
			cur = cur.next()
			parts = append(parts, pyastExprStr(v, cur))
		}
		return pyastRequireParens(opPrec, given, strings.Join(parts, " "+e.op+" "))
	case pyastEAttr:
		s := pyastExprStr(e.a, pyastPrecAtom)
		if e.a.kind == pyastEConst && (e.a.cnst.kind == pyastCInt || e.a.cnst.kind == pyastCTrue || e.a.cnst.kind == pyastCFalse) {
			s += " "
		}
		return s + "." + e.name
	case pyastECall:
		var parts []string
		for _, a := range e.elts {
			parts = append(parts, pyastExprStr(a, pyastPrecTest))
		}
		for _, k := range e.kws {
			parts = append(parts, pyastKeywordStr(k))
		}
		return pyastExprStr(e.a, pyastPrecAtom) + "(" + strings.Join(parts, ", ") + ")"
	case pyastESubscript:
		var sl string
		if e.b.kind == pyastETuple && len(e.b.elts) > 0 {
			sl = pyastItemsView(e.b.elts)
		} else {
			sl = pyastExprStr(e.b, pyastPrecTest)
		}
		return pyastExprStr(e.a, pyastPrecAtom) + "[" + sl + "]"
	case pyastEStarred:
		return "*" + pyastExprStr(e.a, pyastPrecExpr)
	case pyastESlice:
		s := ""
		if e.a != nil {
			s += pyastExprStr(e.a, pyastPrecTest)
		}
		s += ":"
		if e.b != nil {
			s += pyastExprStr(e.b, pyastPrecTest)
		}
		if e.c != nil {
			s += ":" + pyastExprStr(e.c, pyastPrecTest)
		}
		return s
	case pyastENamedExpr:
		body := pyastExprStr(e.a, pyastPrecAtom) + " := " + pyastExprStr(e.b, pyastPrecAtom)
		return pyastRequireParens(pyastPrecNamedExpr, given, body)
	case pyastEAwait:
		return pyastRequireParens(pyastPrecAwait, given, "await "+pyastExprStr(e.a, pyastPrecAtom))
	case pyastEYield:
		body := "yield"
		if e.a != nil {
			body += " " + pyastExprStr(e.a, pyastPrecAtom)
		}
		return pyastRequireParens(pyastPrecYield, given, body)
	case pyastEYieldFrom:
		return pyastRequireParens(pyastPrecYield, given, "yield from "+pyastExprStr(e.a, pyastPrecAtom))
	case pyastELambda:
		body := "lambda"
		if a := pyastArgumentsStr(e.args); a != "" {
			body += " " + a
		}
		body += ": " + pyastExprStr(e.a, pyastPrecTest)
		return pyastRequireParens(pyastPrecTest, given, body)
	}
	pyastBail()
	return ""
}

func pyastRequireParens(p pyastPrec, given pyastPrec, body string) string {
	if given > p {
		return "(" + body + ")"
	}
	return body
}

func pyastBinopPrec(op string) pyastPrec {
	switch op {
	case "+", "-":
		return pyastPrecArith
	case "*", "@", "/", "%", "//":
		return pyastPrecTerm
	case "<<", ">>":
		return pyastPrecShift
	case "|":
		return pyastPrecExpr
	case "^":
		return pyastPrecBXor
	case "&":
		return pyastPrecBAnd
	case "**":
		return pyastPrecPower
	}
	pyastBail()
	return 0
}

// joinExprs is interleave(", ", traverse, items) with default precedence.
func pyastJoinExprs(items []*pyastExpr, p pyastPrec) string {
	parts := make([]string, len(items))
	for i, x := range items {
		parts[i] = pyastExprStr(x, p)
	}
	return strings.Join(parts, ", ")
}

// itemsView is ast._Unparser.items_view: a single item gets a trailing comma.
func pyastItemsView(items []*pyastExpr) string {
	if len(items) == 1 {
		return pyastExprStr(items[0], pyastPrecTest) + ","
	}
	return pyastJoinExprs(items, pyastPrecTest)
}

func pyastCompsStr(comps []*pyastComprehension) string {
	var b strings.Builder
	for _, c := range comps {
		if c.isAsync {
			b.WriteString(" async for ")
		} else {
			b.WriteString(" for ")
		}
		b.WriteString(pyastExprStr(c.target, pyastPrecTuple))
		b.WriteString(" in ")
		b.WriteString(pyastExprStr(c.iter, pyastPrecOr))
		for _, ifc := range c.ifs {
			b.WriteString(" if ")
			b.WriteString(pyastExprStr(ifc, pyastPrecOr))
		}
	}
	return b.String()
}

func pyastKeywordStr(k *pyastKeyword) string {
	if !k.hasArg {
		return "**" + pyastExprStr(k.value, pyastPrecTest)
	}
	return k.argName + "=" + pyastExprStr(k.value, pyastPrecTest)
}

// constantStr is visit_Constant / _write_constant for values that the parser
// can produce. Floats and complex numbers replace inf and nan the way the
// unparser does, so that overflowing literals reparse.
func pyastConstantStr(c *pyastConstant) string {
	switch c.kind {
	case pyastCNone:
		return "None"
	case pyastCTrue:
		return "True"
	case pyastCFalse:
		return "False"
	case pyastCEllipsis:
		return "..."
	case pyastCInt:
		// int.__repr__ refuses more than 4300 digits, so ast.unparse raises
		// and the distiller returns the original code.
		if len(c.digits) > 4300 {
			pyastBail()
		}
		return c.digits
	case pyastCFloat:
		return pyastReplaceNonFinite(pyastFloatRepr(c.f, true))
	case pyastCComplex:
		return pyastReplaceNonFinite(pyastFloatRepr(c.f, false)) + "j"
	case pyastCStr:
		s := pyastStrRepr(c.s)
		if c.uprefix {
			return "u" + s
		}
		return s
	case pyastCBytes:
		return pyastBytesRepr(c.b)
	}
	pyastBail()
	return ""
}

// replaceNonFinite substitutes 1e309 for inf, as _INFSTR does. NaN cannot be
// produced by a literal, so it is not handled further.
func pyastReplaceNonFinite(s string) string {
	return strings.ReplaceAll(s, "inf", "1e309")
}

// argumentsStr is visit_arguments without the surrounding parentheses.
func pyastArgumentsStr(a *pyastArguments) string {
	var b []string
	first := true
	sep := func() {
		if first {
			first = false
		} else {
			b = append(b, ", ")
		}
	}
	all := append(append([]*pyastArg{}, a.posonly...), a.args...)
	defaults := make([]*pyastExpr, len(all)-len(a.defaults))
	defaults = append(defaults, a.defaults...)
	for i, ar := range all {
		sep()
		b = append(b, pyastArgStr(ar))
		if d := defaults[i]; d != nil {
			b = append(b, "="+pyastExprStr(d, pyastPrecTest))
		}
		if i+1 == len(a.posonly) {
			b = append(b, ", /")
		}
	}
	if a.vararg != nil || len(a.kwonly) > 0 {
		sep()
		b = append(b, "*")
		if a.vararg != nil {
			b = append(b, pyastArgStr(a.vararg))
		}
	}
	for i, ar := range a.kwonly {
		b = append(b, ", ")
		b = append(b, pyastArgStr(ar))
		if d := a.kwDefaults[i]; d != nil {
			b = append(b, "="+pyastExprStr(d, pyastPrecTest))
		}
	}
	if a.kwarg != nil {
		sep()
		b = append(b, "**"+pyastArgStr(a.kwarg))
	}
	return strings.Join(b, "")
}

func pyastArgStr(ar *pyastArg) string {
	if ar.ann == nil {
		return ar.name
	}
	return ar.name + ": " + pyastExprStr(ar.ann, pyastPrecTest)
}

// joinedStr is visit_JoinedStr: it picks one quote style that works for all
// literal parts, falling back to triple single quotes with repr escaping.
// Literal text stays as runes so that lone surrogates survive until repr.
func pyastJoinedStr(parts []pyastJsPart) string {
	type seg struct {
		text    []rune
		isConst bool
	}
	segs := make([]seg, 0, len(parts))
	for _, pt := range parts {
		segs = append(segs, seg{text: pyastWriteFStringInner(nil, pt, false), isConst: pt.isConst})
	}
	// _ALL_QUOTES is ('', "", """, ''') in ast.py
	quoteTypes := []string{"'", `"`, `"""`, "'''"}
	fallback := false
	var newParts []string
	for _, sg := range segs {
		value := string(sg.text)
		if sg.isConst {
			escaped, newQuotes := pyastStrLiteralHelper(sg.text, quoteTypes, true)
			if pyastDisjoint(newQuotes, quoteTypes) {
				fallback = true
				break
			}
			quoteTypes = newQuotes
			value = escaped
		} else if strings.Contains(value, "\n") {
			quoteTypes = pyastFilterQuotes(quoteTypes, true)
		}
		newParts = append(newParts, value)
	}
	if fallback {
		quoteTypes = []string{"'''"}
		newParts = nil
		for _, sg := range segs {
			value := string(sg.text)
			if sg.isConst {
				r := pyastStrRepr(append([]rune{'"'}, sg.text...))
				value = r[2 : len(r)-1]
			}
			newParts = append(newParts, value)
		}
	}
	q := quoteTypes[0]
	return "f" + q + strings.Join(newParts, "") + q
}

// writeFStringInner is _write_fstring_inner for one part, appending to out.
func pyastWriteFStringInner(out []rune, pt pyastJsPart, isSpec bool) []rune {
	if !pt.isConst {
		return pyastVisitFormattedValue(out, pt.fv)
	}
	for _, r := range pt.s {
		switch {
		case r == '{' || r == '}':
			out = append(out, r, r)
		case isSpec && r == '\\':
			out = append(out, '\\', '\\')
		case isSpec && r == '\'':
			out = append(out, '\\', '\'')
		case isSpec && r == '"':
			out = append(out, '\\', '"')
		case isSpec && r == '\n':
			out = append(out, '\\', 'n')
		default:
			out = append(out, r)
		}
	}
	return out
}

func pyastAppendStr(out []rune, s string) []rune { return append(out, []rune(s)...) }

func pyastVisitFormattedValue(out []rune, fv *pyastFormatted) []rune {
	inner := pyastExprStr(fv.value, pyastPrecOr)
	out = append(out, '{')
	if strings.HasPrefix(inner, "{") {
		out = append(out, ' ')
	}
	out = pyastAppendStr(out, inner)
	if fv.conv != -1 {
		out = append(out, '!', rune(fv.conv))
	}
	if fv.hasSpec {
		out = append(out, ':')
		for _, sp := range fv.spec {
			out = pyastWriteFStringInner(out, sp, true)
		}
	}
	return append(out, '}')
}

// strLiteralHelper is ast._Unparser._str_literal_helper with quote_types. The
// returned text is always valid UTF-8: every non-printable rune is escaped.
func pyastStrLiteralHelper(s []rune, quoteTypes []string, escapeSpecialWhitespace bool) (string, []string) {
	var eb strings.Builder
	for _, c := range s {
		switch {
		case !escapeSpecialWhitespace && (c == '\n' || c == '\t'):
			eb.WriteRune(c)
		case c == '\\' || !pyastIsPrintable(c):
			eb.WriteString(pyastUnicodeEscape(c))
		default:
			eb.WriteRune(c)
		}
	}
	escaped := eb.String()
	possible := quoteTypes
	if strings.Contains(escaped, "\n") {
		possible = pyastFilterQuotes(possible, true)
	}
	var kept []string
	for _, q := range possible {
		if !strings.Contains(escaped, q) {
			kept = append(kept, q)
		}
	}
	possible = kept
	if len(possible) == 0 {
		r := pyastStrRepr(s)
		first := r[:1]
		quote := first
		for _, q := range quoteTypes {
			if strings.Contains(q, first) {
				quote = q
				break
			}
		}
		return r[1 : len(r)-1], []string{quote}
	}
	if escaped != "" {
		last := escaped[len(escaped)-1]
		// stable sort: quotes not starting with the last character come first
		sorted := make([]string, 0, len(possible))
		for _, q := range possible {
			if q[0] != last {
				sorted = append(sorted, q)
			}
		}
		for _, q := range possible {
			if q[0] == last {
				sorted = append(sorted, q)
			}
		}
		possible = sorted
		if possible[0][0] == last {
			escaped = escaped[:len(escaped)-1] + "\\" + string(last)
		}
	}
	return escaped, possible
}

func pyastFilterQuotes(qs []string, multiOnly bool) []string {
	var out []string
	for _, q := range qs {
		if !multiOnly || len(q) == 3 {
			out = append(out, q)
		}
	}
	return out
}

func pyastDisjoint(a, b []string) bool {
	for _, x := range a {
		for _, y := range b {
			if x == y {
				return false
			}
		}
	}
	return true
}

// ---------------------------------------------------------------- statements

// unparser accumulates source the way ast._Unparser does (_source list,
// indentation of four spaces per level).
type pyastUnparser struct {
	src    []string
	indent int
}

func (u *pyastUnparser) write(s string) { u.src = append(u.src, s) }

func (u *pyastUnparser) maybeNewline() {
	if len(u.src) > 0 {
		u.write("\n")
	}
}

func (u *pyastUnparser) fill(text string) {
	u.maybeNewline()
	u.write(strings.Repeat("    ", u.indent) + text)
}

func (u *pyastUnparser) typeParams(tps []pyastTypeParam) {
	if len(tps) == 0 {
		return
	}
	u.write("[")
	for i, tp := range tps {
		if i > 0 {
			u.write(", ")
		}
		switch tp.kind {
		case pyastTpVar:
			u.write(tp.name)
			if tp.bound != nil {
				u.write(": " + pyastExprStr(tp.bound, pyastPrecTest))
			}
		case pyastTpTuple:
			u.write("*" + tp.name)
		case pyastTpSpec:
			u.write("**" + tp.name)
		}
	}
	u.write("]")
}

// function is _function_helper with every body replaced by "...".
func (u *pyastUnparser) function(s *pyastStmtInfo) {
	suffix := "def "
	if s.isAsync {
		suffix = "async def "
	}
	u.maybeNewline()
	u.fill(suffix + s.name)
	u.typeParams(s.tparams)
	u.write("(")
	u.write(pyastArgumentsStr(s.args))
	u.write(")")
	if s.returns != nil {
		u.write(" -> ")
		u.write(pyastExprStr(s.returns, pyastPrecTest))
	}
	u.write(":")
	u.indent++
	u.fill("")
	u.write("...")
	u.indent--
}

func (u *pyastUnparser) class(s *pyastStmtInfo) {
	u.maybeNewline()
	u.fill("class " + s.name)
	u.typeParams(s.tparams)
	if len(s.bases) > 0 || len(s.keywords) > 0 {
		var parts []string
		for _, b := range s.bases {
			parts = append(parts, pyastExprStr(b, pyastPrecTest))
		}
		for _, k := range s.keywords {
			parts = append(parts, pyastKeywordStr(k))
		}
		u.write("(" + strings.Join(parts, ", ") + ")")
	}
	u.write(":")
	u.indent++
	for _, m := range s.methods {
		u.function(m)
	}
	u.indent--
}

// unparseModule is ast.unparse(ast.Module(body=kept, type_ignores=[])).
func pyastUnparseModule(kept []*pyastStmtInfo) string {
	u := &pyastUnparser{}
	for _, s := range kept {
		if s.kind == pyastSFunc {
			u.function(s)
		} else {
			u.class(s)
		}
	}
	return strings.Join(u.src, "")
}

// distillPythonSignatures ports _distill_python_signatures
// (backend/engines/optimizer.py, CPython 3.12) for one code block. It keeps the
// public top-level functions and classes with their public methods, stubs every
// body with "...", and returns the ast.unparse text of what it kept. Source
// that CPython rejects, or that leaves nothing to keep, comes back unchanged.
//
// Exact against CPython 3.12 on the corpus in server/parity/engine_cases.py,
// with these exceptions, which return ErrUnsupported so the request goes to
// the Python backend instead of a guessed answer:
//   - an expression nesting weight above pyastChainLimit on one logical line.
//     CPython's ast.unparse can fail on such a line, and that failure is caught
//     and turned into "return the code", which this port does not model;
//   - a broken embedded name table (pyast_names.go).
//
// Not modelled, and documented rather than declined: CPython's MemoryError on
// extremely large parser stacks, and inputs beyond CPython's memory limits.
// Identifiers are NFKC-normalised with golang.org/x/text/unicode/norm. Its
// tables are newer than CPython 3.12's Unicode 15.0, but the only code points
// where the two differ are unassigned in Unicode 15.0, and the identifier check
// (pyastValidIdentifier) rejects them before normalisation.
func distillPythonSignatures(code string) (out string, err error) {
	defer func() {
		if r := recover(); r != nil {
			if _, is := r.(pyastDeclined); is {
				out, err = "", ErrUnsupported
				return
			}
			panic(r)
		}
	}()
	// ast.parse raises ValueError for NUL bytes and cannot represent invalid
	// UTF-8, so both fall back to the original text.
	if strings.IndexByte(code, 0) >= 0 || !utf8.ValidString(code) {
		return code, nil
	}
	stmts, ok := pyastParseModule(pyastTranslateNewlines(code))
	if !ok {
		return code, nil
	}
	var kept []*pyastStmtInfo
	for _, s := range stmts {
		switch s.kind {
		case pyastSFunc:
			if pyastIsPublicName(s.name) {
				kept = append(kept, s)
			}
		case pyastSClass:
			var methods []*pyastStmtInfo
			for _, m := range s.methods {
				if pyastIsPublicName(m.name) {
					methods = append(methods, m)
				}
			}
			if len(methods) > 0 {
				s.methods = methods
				kept = append(kept, s)
			}
		}
	}
	if len(kept) == 0 {
		return code, nil
	}
	text, ok := pyastSafeUnparse(kept)
	if !ok {
		return code, nil
	}
	return text, nil
}

// safeUnparse runs the unparser; a refusal (bail) mirrors the exception that
// ast.unparse raises, which the Python function turns into the original code.
func pyastSafeUnparse(kept []*pyastStmtInfo) (out string, ok bool) {
	defer func() {
		if r := recover(); r != nil {
			if _, is := r.(pyastSyntaxErr); is {
				out, ok = "", false
				return
			}
			panic(r)
		}
	}()
	return pyastUnparseModule(kept), true
}

// isPublicName is the is_public helper: no leading underscore, or a dunder.
func pyastIsPublicName(name string) bool {
	return !strings.HasPrefix(name, "_") || (strings.HasPrefix(name, "__") && strings.HasSuffix(name, "__"))
}

// translateNewlines converts CRLF and lone CR to LF, as CPython's tokenizer
// does for str input.
func pyastTranslateNewlines(s string) string {
	if !strings.Contains(s, "\r") {
		return s
	}
	s = strings.ReplaceAll(s, "\r\n", "\n")
	return strings.ReplaceAll(s, "\r", "\n")
}

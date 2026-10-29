// gzip helpers that mirror CPython's gzip.decompress and gzip.compress.
//
// Python's gzip reports two classes of failure, and the engines catch them
// differently. Header and CRC/length problems raise OSError (BadGzipFile),
// which the engines swallow; truncation (EOFError) and corrupt deflate data
// (zlib.error) are not OSError and propagate. gzipDecompress keeps the split
// via isGzipOSError. The Go gzip reader cannot be used directly because it
// rejects the trailing NUL padding that Python skips, so the member loop is
// written out here.
package engines

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"io"
	"time"
)

// gzipOSError is Python's BadGzipFile (an OSError subclass).
type gzipOSError struct{ msg string }

func (e *gzipOSError) Error() string { return "BadGzipFile: " + e.msg }

// isGzipOSError reports whether err is the OSError class that callers catch.
func isGzipOSError(err error) bool {
	var e *gzipOSError
	return errors.As(err, &e)
}

func badGzip(msg string) error { return &gzipOSError{msg: msg} }

// gzipDecompress mirrors gzip.decompress(data), including multi-member
// streams and the lstrip(b"\x00") that skips zero padding between members.
func gzipDecompress(data []byte) ([]byte, error) {
	var out []byte
	for len(data) > 0 {
		// _read_gzip_header: a short magic is a bad file, not EOF.
		if len(data) < 2 || data[0] != 0x1f || data[1] != 0x8b {
			return nil, badGzip("Not a gzipped file")
		}
		if len(data) < 10 {
			return nil, pyError("EOFError", "Compressed file ended before the end-of-stream marker was reached")
		}
		method, flag := data[2], data[3]
		if method != 8 {
			return nil, badGzip("Unknown compression method")
		}
		pos := 10
		if flag&0x04 != 0 { // FEXTRA
			if len(data) < pos+2 {
				return nil, pyError("EOFError", "truncated gzip header")
			}
			extra := int(binary.LittleEndian.Uint16(data[pos:]))
			pos += 2 + extra
			if pos > len(data) {
				return nil, pyError("EOFError", "truncated gzip header")
			}
		}
		for _, bit := range []byte{0x08, 0x10} { // FNAME, FCOMMENT: NUL-terminated
			if flag&bit != 0 {
				for pos < len(data) && data[pos] != 0 {
					pos++
				}
				// The terminator (if any) is consumed by the loop in Python.
				if pos < len(data) {
					pos++
				}
			}
		}
		if flag&0x02 != 0 { // FHCRC
			pos += 2
			if pos > len(data) {
				return nil, pyError("EOFError", "truncated gzip header")
			}
		}

		br := bytes.NewReader(data[pos:])
		fr := flate.NewReader(br)
		member, err := io.ReadAll(fr)
		if err != nil {
			// Truncated and corrupt deflate data both surface as non-OSError.
			return nil, pyError("EOFError", "gzip deflate stream: "+err.Error())
		}
		unused := br.Len()
		if unused < 8 {
			return nil, pyError("EOFError", "Compressed file ended before the end-of-stream marker was reached")
		}
		trailer := data[len(data)-unused:]
		crc := binary.LittleEndian.Uint32(trailer[0:4])
		length := binary.LittleEndian.Uint32(trailer[4:8])
		if crc != crc32.ChecksumIEEE(member) {
			return nil, badGzip("CRC check failed")
		}
		if length != uint32(len(member)) {
			return nil, badGzip("Incorrect length of data produced")
		}
		out = append(out, member...)
		data = bytes.TrimLeft(trailer[8:], "\x00")
	}
	return out, nil
}

// gzipCompress mirrors gzip.compress(data, compresslevel=6). The deflate
// bytes come from Go's compress/flate, which is not byte-identical to zlib,
// so the compressed form differs from Python even though it round-trips.
func gzipCompress(data []byte) []byte {
	var buf bytes.Buffer
	zw, _ := gzip.NewWriterLevel(&buf, 6)
	zw.ModTime = time.Now()
	_, _ = zw.Write(data)
	_ = zw.Close()
	return buf.Bytes()
}

package security

import "fmt"

// UploadQuotaPolicy enforces upload batch size and file-size limits.
type UploadQuotaPolicy struct {
	MaxFiles      int
	MaxTotalBytes int64
	MaxFileBytes  int64
}

// QuotaDecision represents the result of a quota check.
type QuotaDecision struct {
	Kind     string
	Allowed  bool
	Reason   string
	Observed map[string]int64
	Limits   map[string]int64
}

// NewUploadQuotaPolicy creates an upload quota policy.
func NewUploadQuotaPolicy(maxFiles, maxTotalBytes, maxFileBytes int) *UploadQuotaPolicy {
	return &UploadQuotaPolicy{
		MaxFiles:      maxFiles,
		MaxTotalBytes: int64(maxTotalBytes),
		MaxFileBytes:  int64(maxFileBytes),
	}
}

// Check validates file sizes against quota limits.
func (p *UploadQuotaPolicy) Check(fileSizes []int64) QuotaDecision {
	var totalBytes, largestFile int64
	for _, s := range fileSizes {
		totalBytes += s
		if s > largestFile {
			largestFile = s
		}
	}

	var violations []string
	if len(fileSizes) > p.MaxFiles {
		violations = append(violations, fmt.Sprintf("too many files (%d > %d)", len(fileSizes), p.MaxFiles))
	}
	if totalBytes > p.MaxTotalBytes {
		violations = append(violations, fmt.Sprintf("batch too large (%d > %d)", totalBytes, p.MaxTotalBytes))
	}
	if largestFile > p.MaxFileBytes {
		violations = append(violations, fmt.Sprintf("file too large (%d > %d)", largestFile, p.MaxFileBytes))
	}

	return QuotaDecision{
		Kind:    "upload",
		Allowed: len(violations) == 0,
		Reason:  fmt.Sprintf("%v", violations),
		Observed: map[string]int64{
			"files":              int64(len(fileSizes)),
			"total_bytes":        totalBytes,
			"largest_file_bytes": largestFile,
		},
		Limits: map[string]int64{
			"max_files":      int64(p.MaxFiles),
			"max_total_bytes": p.MaxTotalBytes,
			"max_file_bytes":  p.MaxFileBytes,
		},
	}
}

// IndexQuotaPolicy enforces document/chunk/character count limits.
type IndexQuotaPolicy struct {
	MaxDocuments int
	MaxChunks    int
	MaxChars     int
}

// NewIndexQuotaPolicy creates an index quota policy.
func NewIndexQuotaPolicy(maxDocs, maxChunks, maxChars int) *IndexQuotaPolicy {
	return &IndexQuotaPolicy{
		MaxDocuments: maxDocs,
		MaxChunks:    maxChunks,
		MaxChars:     maxChars,
	}
}

// Check validates document/chunk/char counts against quota limits.
func (p *IndexQuotaPolicy) Check(documentCount, chunkCount, totalChars int) QuotaDecision {
	var violations []string
	if documentCount > p.MaxDocuments {
		violations = append(violations, fmt.Sprintf("too many documents (%d > %d)", documentCount, p.MaxDocuments))
	}
	if chunkCount > p.MaxChunks {
		violations = append(violations, fmt.Sprintf("too many chunks (%d > %d)", chunkCount, p.MaxChunks))
	}
	if totalChars > p.MaxChars {
		violations = append(violations, fmt.Sprintf("too many characters (%d > %d)", totalChars, p.MaxChars))
	}

	return QuotaDecision{
		Kind:    "index",
		Allowed: len(violations) == 0,
		Reason:  fmt.Sprintf("%v", violations),
		Observed: map[string]int64{
			"documents": int64(documentCount),
			"chunks":    int64(chunkCount),
			"chars":     int64(totalChars),
		},
		Limits: map[string]int64{
			"max_documents": int64(p.MaxDocuments),
			"max_chunks":    int64(p.MaxChunks),
			"max_chars":     int64(p.MaxChars),
		},
	}
}

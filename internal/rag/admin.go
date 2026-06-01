package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"
	"github.com/elastic/go-elasticsearch/v8"
)

// RetrievedDoc is an alias for *schema.Document so callers in the
// server package can refer to eino documents without importing the
// eino schema package directly.
type RetrievedDoc = schema.Document

// EinoMessage is an alias for *schema.Message used for LLM calls.
type EinoMessage = schema.Message

// Retriever option helpers — thin wrappers around eino's built-in
// retriever.WithIndex / WithTopK / WithScoreThreshold. These let the
// server package avoid importing the eino retriever package directly.
func WithIndexOpt(name string) retriever.Option {
	return retriever.WithIndex(name)
}

func WithTopKOpt(k int) retriever.Option {
	return retriever.WithTopK(k)
}

func WithMinScoreOpt(score float64) retriever.Option {
	return retriever.WithScoreThreshold(score)
}

// Admin indexer operations. These call the ES client directly and are
// used for ListDocuments / DeleteDocument / ListFiles / DeleteFile
// admin endpoints. eino-ext does not provide equivalents.
type adminIndexer struct {
	client *elasticsearch.Client
}

func newAdminIndexer(client *elasticsearch.Client) *adminIndexer {
	return &adminIndexer{client: client}
}

// AdminListDocuments returns paginated documents from the given index.
func AdminListDocuments(client *elasticsearch.Client, indexName string, limit, offset int) (*DocumentList, error) {
	if client == nil {
		return nil, fmt.Errorf("AdminListDocuments: nil client")
	}
	return listDocumentsAgg(client, indexName, limit, offset, nil)
}

// AdminDeleteDocument removes all chunks with the given document_id.
func AdminDeleteDocument(client *elasticsearch.Client, indexName, documentID string) error {
	if client == nil {
		return fmt.Errorf("AdminDeleteDocument: nil client")
	}
	req := map[string]any{"query": map[string]any{"term": map[string]any{"document_id": documentID}}}
	body, _ := json.Marshal(req)
	res, err := client.DeleteByQuery([]string{indexName}, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("delete document: %w", err)
	}
	res.Body.Close()
	return nil
}

// AdminListFiles returns aggregated file information for the index.
func AdminListFiles(client *elasticsearch.Client, indexName string) ([]FileInfo, error) {
	if client == nil {
		return nil, fmt.Errorf("AdminListFiles: nil client")
	}
	return listFilesAgg(client, indexName)
}

// AdminDeleteFile removes all chunks for the given filename.
func AdminDeleteFile(client *elasticsearch.Client, indexName, filename string) error {
	if client == nil {
		return fmt.Errorf("AdminDeleteFile: nil client")
	}
	req := map[string]any{"query": map[string]any{"term": map[string]any{"filename": filename}}}
	body, _ := json.Marshal(req)
	res, err := client.DeleteByQuery([]string{indexName}, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("delete file: %w", err)
	}
	res.Body.Close()
	return nil
}

// listDocumentsAgg runs the by_doc aggregation and returns a DocumentList.
func listDocumentsAgg(client *elasticsearch.Client, indexName string, limit, offset int, filename *string) (*DocumentList, error) {
	ctx := context.Background()
	req := map[string]any{"size": 0, "aggs": map[string]any{
		"by_doc": map[string]any{
			"terms": map[string]any{"field": "document_id", "size": limit + offset},
			"aggs": map[string]any{
				"sample":      map[string]any{"top_hits": map[string]any{"size": 1, "_source": []string{"content", "document_id", "source", "filename", "file_type", "chunk_index", "processed_at", "chunk_char_count"}}},
				"chunk_count": map[string]any{"value_count": map[string]string{"field": "chunk_index"}},
			},
		},
	}}
	body, _ := json.Marshal(req)
	res, err := client.Search(client.Search.WithContext(ctx), client.Search.WithIndex(indexName), client.Search.WithBody(bytes.NewReader(body)))
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode == 404 {
		return &DocumentList{Total: 0, Documents: []DocInfo{}, Limit: limit, Offset: offset}, nil
	}
	if res.StatusCode >= 400 {
		b, _ := io.ReadAll(res.Body)
		return nil, fmt.Errorf("list documents: %s", string(b))
	}
	return parseDocList(res.Body, limit, offset)
}

func listFilesAgg(client *elasticsearch.Client, indexName string) ([]FileInfo, error) {
	ctx := context.Background()
	req := map[string]any{"size": 0, "aggs": map[string]any{
		"by_file": map[string]any{
			"terms": map[string]any{"field": "filename", "size": 200},
			"aggs": map[string]any{
				"sample":      map[string]any{"top_hits": map[string]any{"size": 1, "_source": []string{"document_id", "source", "file_type", "processed_at"}}},
				"chunk_count": map[string]any{"value_count": map[string]string{"field": "chunk_index"}},
				"total_chars": map[string]any{"sum": map[string]string{"field": "chunk_char_count"}},
			},
		},
	}}
	body, _ := json.Marshal(req)
	res, err := client.Search(client.Search.WithContext(ctx), client.Search.WithIndex(indexName), client.Search.WithBody(bytes.NewReader(body)))
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode == 404 {
		return []FileInfo{}, nil
	}
	if res.StatusCode >= 400 {
		b, _ := io.ReadAll(res.Body)
		return nil, fmt.Errorf("list files: %s", string(b))
	}
	var aggResp struct {
		Aggregations struct {
			ByFile struct {
				Buckets []struct {
					Key        string  `json:"key"`
					ChunkCount struct{ Value int } `json:"chunk_count"`
					TotalChars struct{ Value float64 } `json:"total_chars"`
					Sample     struct {
						Hits struct {
							Hits []struct {
								Source struct {
									DocumentID  string `json:"document_id"`
									Source      string `json:"source"`
									FileType    string `json:"file_type"`
									ProcessedAt string `json:"processed_at"`
								} `json:"_source"`
							} `json:"hits"`
						} `json:"hits"`
					} `json:"sample"`
				} `json:"buckets"`
			} `json:"by_file"`
		} `json:"aggregations"`
	}
	json.NewDecoder(res.Body).Decode(&aggResp)
	var files []FileInfo
	for _, b := range aggResp.Aggregations.ByFile.Buckets {
		fi := FileInfo{Filename: b.Key, ChunkCount: b.ChunkCount.Value, TotalChars: int(b.TotalChars.Value)}
		if len(b.Sample.Hits.Hits) > 0 {
			src := b.Sample.Hits.Hits[0].Source
			fi.DocumentID, fi.Source, fi.FileType, fi.ProcessedAt = src.DocumentID, src.Source, src.FileType, src.ProcessedAt
		}
		files = append(files, fi)
	}
	if files == nil {
		files = []FileInfo{}
	}
	return files, nil
}

// DocInfo is a summary of a document for listing.
type DocInfo struct {
	ID          string `json:"id"`
	Content     string `json:"content"`
	Source      string `json:"source"`
	Filename    string `json:"filename"`
	FileType    string `json:"file_type"`
	ChunkCount  int    `json:"chunk_count"`
	ProcessedAt string `json:"processed_at"`
}

// DocumentList is the response for list-documents.
type DocumentList struct {
	Total     int       `json:"total"`
	Documents []DocInfo `json:"documents"`
	Limit     int       `json:"limit"`
	Offset    int       `json:"offset"`
}

// FileInfo is the response for list-files.
type FileInfo struct {
	Filename    string `json:"filename"`
	Source      string `json:"source"`
	FileType    string `json:"file_type"`
	ChunkCount  int    `json:"chunk_count"`
	TotalChars  int    `json:"total_chars"`
	DocumentID  string `json:"document_id"`
	ProcessedAt string `json:"processed_at"`
}

func parseDocList(r io.Reader, limit, offset int) (*DocumentList, error) {
	var aggResp struct {
		Aggregations struct {
			ByDoc struct {
				Buckets []struct {
					Key string `json:"key"`
					Sample struct {
						Hits struct {
							Hits []struct {
								Source struct {
									Content        string `json:"content"`
									DocumentID     string `json:"document_id"`
									Source         string `json:"source"`
									Filename       string `json:"filename"`
									FileType       string `json:"file_type"`
									ChunkIndex     int    `json:"chunk_index"`
									ProcessedAt    string `json:"processed_at"`
									ChunkCharCount int    `json:"chunk_char_count"`
								} `json:"_source"`
							} `json:"hits"`
						} `json:"hits"`
					} `json:"sample"`
					ChunkCount struct{ Value int } `json:"chunk_count"`
				} `json:"buckets"`
			} `json:"by_doc"`
		} `json:"aggregations"`
	}
	json.NewDecoder(r).Decode(&aggResp)
	buckets := aggResp.Aggregations.ByDoc.Buckets
	total := len(buckets)
	var docs []DocInfo
	for i := offset; i < total && i < offset+limit; i++ {
		b := buckets[i]
		if len(b.Sample.Hits.Hits) > 0 {
			src := b.Sample.Hits.Hits[0].Source
			docs = append(docs, DocInfo{
				ID: src.DocumentID, Content: truncate(src.Content, 200),
				Source: src.Source, Filename: src.Filename, FileType: src.FileType,
				ChunkCount: b.ChunkCount.Value, ProcessedAt: src.ProcessedAt,
			})
		}
	}
	if docs == nil {
		docs = []DocInfo{}
	}
	return &DocumentList{Total: total, Documents: docs, Limit: limit, Offset: offset}, nil
}

func truncate(s string, maxLen int) string {
	runes := []rune(s)
	if len(runes) <= maxLen {
		return s
	}
	return string(runes[:maxLen]) + "..."
}

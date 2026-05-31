# Elasticsearch Knowledge for mcp-rag-go

## Core Concepts

```
Cluster
└── Index (≈ SQL Table, ≈ ChromaDB Collection)
    ├── Has a Mapping (schema definition)
    ├── Split into Shards for horizontal scaling
    └── Document (≈ SQL Row, the physical storage unit)
        ├── _id: unique identifier, PUT with same _id = upsert
        ├── _source: original JSON document
        └── Each field indexed according to Mapping type
```

**Our mapping**: 1 KnowledgeBase = 1 ES Index = 1 Mapping

## Go Client: Two API Styles

Uses `github.com/elastic/go-elasticsearch/v8` (v8.16.0, same as eino es8 retriever).

| Basic Client | Typed Client (typedapi) |
|---|---|
| Uses `json.RawMessage` | Compile-time type safety |
| Flexible for dynamic queries | Slightly more verbose |
| | **eino es8 uses this internally** |

**We use typedapi for consistency with eino.**

## Mapping: Defining Index Structure

```go
res, err := esClient.Indices.Create("kb_1").
    Request(&indicescreate.Request{
        Mappings: &types.TypeMapping{
            Properties: map[string]types.Property{
                // Text field → BM25 full-text search
                "content": types.NewTextProperty(),

                // Dense vector → KNN approximate search
                "content_vector": types.NewDenseVectorProperty(),

                // Keyword → exact match, filter, aggregation
                "document_id": types.NewKeywordProperty(),
                "filename":    types.NewKeywordProperty(),
                "source":      types.NewKeywordProperty(),
                "file_type":   types.NewKeywordProperty(),

                // Object → tenant filtering
                "tenant": types.NewObjectProperty(),

                // Numeric
                "chunk_index":  types.NewIntegerNumberProperty(),
                "total_chunks": types.NewIntegerNumberProperty(),

                // Date
                "processed_at": types.NewDateProperty(),

                // Dynamic key-value → term queries on leaf values
                "custom_metadata": types.NewFlattenedProperty(),
            },
        },
    }).Do(ctx)
```

### Key Mapping Types

| Type | Use Case | Query Type |
|---|---|---|
| `text` | Full-text search | `match` (BM25 scoring) |
| `keyword` | Exact match / aggregation / sorting | `term`, `terms` |
| `dense_vector` | Vector similarity | `knn` |
| `integer`/`long` | Numeric range | `range` |
| `date` | Time range | `range` |
| `boolean` | Boolean filter | `term` |
| `object` | Nested structure | Query on sub-fields |
| `flattened` | Dynamic key-value | `term` on leaf values (auto-indexed) |

## Write: Index / Bulk

### Single Document

```go
// PUT /kb_1/_doc/{chunk_id}  — same _id = automatic upsert
res, err := esClient.Index("kb_1").
    Id(chunkID).
    Request(&types.Document{
        "content":        "some text...",
        "content_vector": []float32{0.12, -0.34},
        "document_id":    "a1b2c3d4",
        "chunk_index":    0,
        "filename":       "report.pdf",
    }).
    Do(ctx)
```

### Bulk Index (batch chunks)

```go
bulk := esClient.Bulk().Index("kb_1")

for _, chunk := range chunks {
    bulk.Add(types.Operation{
        Index: &types.IndexOperation{Id_: &chunk.ID},
    }, &types.Document{
        "content":        chunk.Content,
        "content_vector": chunk.Vector,
        "document_id":    chunk.DocumentID,
        // ... more fields
    })
}

res, err := bulk.Do(ctx)
if res.Errors {
    // Partial failure — check res.Items for per-document errors
    for _, item := range res.Items {
        if item.Index.Error != nil {
            log.Printf("bulk index error: %s", item.Index.Error.Reason)
        }
    }
}
```

### Refresh After Write

```go
// Default refresh: every 1s. For strong consistency:
esClient.Indices.Refresh().Index("kb_1").Do(ctx)
```

## Search: KNN + BM25 + RRF (Hybrid Retrieval)

### Pure KNN (Vector)

```go
res, err := esClient.Search().Index("kb_1").
    Request(&search.Request{
        Knn: []types.KnnSearch{{
            Field:         "content_vector",
            QueryVector:   queryVector,
            K:             &topK,
            NumCandidates: &numCandidates,
        }},
        Size: &topK,
    }).Do(ctx)
```

### Pure BM25 (Full-Text)

```go
res, err := esClient.Search().Index("kb_1").
    Request(&search.Request{
        Query: &types.Query{
            Match: map[string]types.MatchQuery{
                "content": {Query: userQuery},
            },
        },
        Size: &topK,
    }).Do(ctx)
```

### KNN + BM25 + RRF (Hybrid — Our Core Scenario)

```go
res, err := esClient.Search().Index("kb_1").
    Request(&search.Request{
        // KNN (vector path)
        Knn: []types.KnnSearch{{
            Field:         "content_vector",
            QueryVector:   queryVector,
            K:             &topK,
            NumCandidates: &candidates,
            Filter: []types.Query{{
                Term: map[string]types.TermQuery{
                    "tenant.base_collection": {Value: "default"},
                }},
            }},
        }},
        // BM25 (text path)
        Query: &types.Query{
            Bool: &types.BoolQuery{
                Must: []types.Query{{
                    Match: map[string]types.MatchQuery{
                        "content": {Query: userQuery},
                    },
                }},
                Filter: []types.Query{{
                    Term: map[string]types.TermQuery{
                        "tenant.base_collection": {Value: "default"},
                    }},
                }},
            },
        },
        // RRF fusion of both result sets
        Rank: &types.RankContainer{
            Rrf: &types.RrfRank{
                RankConstant:   ptr(int64(60)),
                RankWindowSize: ptr(int64(100)),
            },
        },
        Size:     &topK,
        MinScore: &minScore,
    }).Do(ctx)
```

### RRF (Reciprocal Rank Fusion) Explained

```
RRF only uses rank positions, not raw scores:

Vector path ranking:    DocA#1, DocB#2, DocC#3
BM25 path ranking:      DocC#1, DocD#2, DocA#3

RRF fusion (k=60):
  DocA: 1/(60+1) + 1/(60+3) = 0.0323
  DocC: 1/(60+3) + 1/(60+1) = 0.0323
  DocB: 1/(60+2) + 0        = 0.0161
  DocD: 0        + 1/(60+2) = 0.0161

Final order: A/C → B → D
Documents appearing in both paths get higher priority.
```

## Filter: Tenant Isolation

```go
func buildTenantFilter(tenant TenantContext) []types.Query {
    filters := []types.Query{
        {Term: map[string]types.TermQuery{
            "tenant.base_collection": {Value: tenant.BaseCollection},
        }},
    }
    if tenant.UserID != nil {
        filters = append(filters, types.Query{
            Term: map[string]types.TermQuery{
                "tenant.user_id": {Value: *tenant.UserID},
            },
        })
    }
    if tenant.AgentID != nil {
        filters = append(filters, types.Query{
            Term: map[string]types.TermQuery{
                "tenant.agent_id": {Value: *tenant.AgentID},
            },
        })
    }
    return filters
}
```

## Aggregation: list_documents / list_files

```go
// Group by filename, with per-file stats
res, err := esClient.Search().Index("kb_1").
    Request(&search.Request{
        Size: ptr(0), // No document results, only aggregations
        Query: &types.Query{
            Bool: &types.BoolQuery{
                Filter: buildTenantFilter(tenant),
            },
        },
        Aggregations: map[string]types.Aggregations{
            "by_filename": types.NewAggregation().
                Terms(&types.TermsAggregation{
                    Field: ptr("filename"),
                    Size:  ptr(200),
                }).
                // Sub-aggregation: sample one document per file
                TopHits(&types.TopHitsAggregation{
                    Size: ptr(1),
                    Source_: &types.SourceFilter{
                        Includes: []string{
                            "document_id", "source", "file_type",
                            "filename", "processed_at", "chunk_char_count",
                        },
                    },
                }).
                // Sub-aggregation: chunk count per file
                ValueCount("chunk_count", &types.ValueCountAggregation{
                    Field: ptr("_id"),
                }).
                // Sub-aggregation: total character count per file
                Sum("total_chars", &types.SumAggregation{
                    Field: ptr("chunk_char_count"),
                }),
        },
    }).Do(ctx)
```

## Delete

```go
// Delete all chunks by document_id
esClient.DeleteByQuery("kb_1").
    Request(&deletebyquery.Request{
        Query: &types.Query{
            Term: map[string]types.TermQuery{
                "document_id": {Value: docID},
            },
        },
    }).Do(ctx)

// Delete all chunks by filename (with tenant filter)
esClient.DeleteByQuery("kb_1").
    Request(&deletebyquery.Request{
        Query: &types.Query{
            Bool: &types.BoolQuery{
                Filter: append(
                    buildTenantFilter(tenant),
                    types.Query{
                        Term: map[string]types.TermQuery{
                            "filename": {Value: filename},
                        },
                    },
                ),
            },
        },
    }).Do(ctx)
```

## Index Lifecycle

```go
// Create index with mapping
esClient.Indices.Create("kb_3").Request(&indicescreate.Request{
    Mappings: buildKBMapping(embeddingDims),
}).Do(ctx)

// Check existence
exists, _ := esClient.Indices.Exists("kb_3").Do(ctx)
// exists.StatusCode == 200 means index exists

// Delete index (when deleting knowledge base)
esClient.Indices.Delete("kb_3").Do(ctx)
```

## Practical Notes

1. **refresh_interval**: Default 1s — documents may not be immediately searchable after write. Suggested: `"5s"` (production) or `-1` (disable during bulk writes, manually refresh after).

2. **_id design**: Use `chunk_id` (`"{document_id}_chunk_{index}"`) — natural idempotent upsert.

3. **num_candidates**: Number of candidates per shard for KNN. Single shard ≈ total doc count. Multi shard > `topK * shard_count`.

4. **Error handling**:
   - Bulk `res.Errors == true` → partial failure, check `res.Items`
   - Non-existent index → 404 on query
   - `_delete_by_query` does not wait for refresh by default

5. **typedapi conventions**:
   - `ptr(v)` — return pointer (ES API uses pointers to distinguish "unset" from "zero value")
   - `types.NewXxxProperty()` — create mapping property
   - `types.NewAggregation().Terms(...)` — chain-build aggregations

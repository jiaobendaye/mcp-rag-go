package rag

import (
	"testing"
)

func TestClassifier_TableDriven(t *testing.T) {
	qc := NewQueryClassifier()

	tests := []struct {
		name      string
		query     string
		wantIntent QueryIntent
		wantVec   float64
		wantKw    float64
	}{
		// code_explanation (vector: 0.75, keyword: 0.25)
		{"explain-code-1", "Can you explain this code?", IntentCodeExplanation, 0.75, 0.25},
		{"explain-code-2", "What does this function do?", IntentCodeExplanation, 0.75, 0.25},

		// troubleshooting (vector: 0.4, keyword: 0.6)
		{"error-query", "I'm getting a 500 error when calling the API", IntentTroubleshooting, 0.4, 0.6},
		{"fix-bug", "How to fix the bug in this exception handler?", IntentTroubleshooting, 0.4, 0.6},

		// how_to (vector: 0.65, keyword: 0.35)
		{"how-to", "How to deploy a Go application on Kubernetes?", IntentHowTo, 0.65, 0.35},
		{"tutorial", "Step by step tutorial for building REST API", IntentHowTo, 0.65, 0.35},

		// best_practices (vector: 0.8, keyword: 0.2)
		{"best-practice", "What are the best practices for error handling in Go?", IntentBestPractices, 0.8, 0.2},
		{"recommended", "Recommended way to structure a Go project", IntentBestPractices, 0.8, 0.2},

		// comparison (vector: 0.7, keyword: 0.3)
		{"vs", "Python vs Go performance comparison", IntentComparison, 0.7, 0.3},
		{"difference", "What is the difference between channels and mutexes?", IntentComparison, 0.7, 0.3},

		// technical_docs (vector: 0.6, keyword: 0.4)
		{"api-ref", "List all API parameters for /search endpoint", IntentTechnicalDocs, 0.6, 0.4},
		{"config", "Configuration reference for the embedding model", IntentTechnicalDocs, 0.6, 0.4},

		// conceptual (vector: 0.8, keyword: 0.2)
		{"what-is", "What is a knowledge base in RAG?", IntentConceptual, 0.8, 0.2},
		{"architecture", "Architecture of an MCP server", IntentConceptual, 0.8, 0.2},

		// general_qa (vector: 0.7, keyword: 0.3) - default
		{"general", "Tell me about the weather today", IntentGeneralQA, 0.7, 0.3},
		{"general-2", "Random question without technical context", IntentGeneralQA, 0.7, 0.3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cls := qc.Classify(tt.query)
			if cls.PrimaryIntent != tt.wantIntent {
				t.Errorf("query=%q: PrimaryIntent=%s, want %s", tt.query, cls.PrimaryIntent, tt.wantIntent)
			}
			got := GetWeights(cls.PrimaryIntent)
			if got.Vector != tt.wantVec || got.Keyword != tt.wantKw {
				t.Errorf("query=%q: weights=(%.2f,%.2f), want (%.2f,%.2f)",
					tt.query, got.Vector, got.Keyword, tt.wantVec, tt.wantKw)
			}
		})
	}
}

func TestClassifier_EmptyQuery(t *testing.T) {
	qc := NewQueryClassifier()
	cls := qc.Classify("")
	if cls.PrimaryIntent != IntentGeneralQA {
		t.Errorf("empty query should default to general_qa, got %s", cls.PrimaryIntent)
	}
	if cls.Confidence < 0 {
		t.Errorf("confidence should be non-negative, got %.2f", cls.Confidence)
	}
}

func TestClassifier_TechnicalDetection(t *testing.T) {
	qc := NewQueryClassifier()
	tests := []struct {
		query string
		want  bool
	}{
		{"How to use python for numpy array operations", true},
		{"React hooks with TypeScript", true},
		{"Hello, how are you?", false},
		{"Thank you", false},
	}
	for _, tt := range tests {
		t.Run(tt.query, func(t *testing.T) {
			cls := qc.Classify(tt.query)
			if cls.IsTechnical != tt.want {
				t.Errorf("query=%q: IsTechnical=%v, want %v", tt.query, cls.IsTechnical, tt.want)
			}
		})
	}
}

func TestClassifier_Keywords(t *testing.T) {
	qc := NewQueryClassifier()
	cls := qc.Classify("How to use python and django for web development?")
	if len(cls.Keywords) == 0 {
		t.Error("expected non-empty keywords")
	}
	// Should include content words, exclude stop words
	hasStop := false
	for _, k := range cls.Keywords {
		if k == "the" || k == "to" || k == "for" {
			hasStop = true
			break
		}
	}
	if hasStop {
		t.Error("keywords should exclude stop words")
	}
}

func TestClassifier_TroubleshootingBoostsKeywordWeight(t *testing.T) {
	_ = NewQueryClassifier().Classify("error: 500 Internal Server Error when calling /search")
	troubleshootingW := GetWeights(IntentTroubleshooting)
	conceptualW := GetWeights(IntentConceptual)
	if troubleshootingW.Keyword <= troubleshootingW.Vector {
		t.Error("troubleshooting should have higher keyword weight than vector weight")
	}
	if conceptualW.Vector <= conceptualW.Keyword {
		t.Error("conceptual should have higher vector weight than keyword weight")
	}
}

func TestGetWeights_UnknownIntent(t *testing.T) {
	// Force an unknown intent by using a value outside the enum
	got := GetWeights(QueryIntent(999))
	if got.Vector != DefaultWeights.Vector || got.Keyword != DefaultWeights.Keyword {
		t.Errorf("unknown intent should fall back to DefaultWeights, got %+v", got)
	}
}

func TestQueryIntent_String(t *testing.T) {
	tests := []struct {
		intent QueryIntent
		want   string
	}{
		{IntentCodeExplanation, "code_explanation"},
		{IntentTroubleshooting, "troubleshooting"},
		{IntentHowTo, "how_to"},
		{IntentBestPractices, "best_practices"},
		{IntentComparison, "comparison"},
		{IntentTechnicalDocs, "technical_docs"},
		{IntentConceptual, "conceptual"},
		{IntentGeneralQA, "general_qa"},
		{QueryIntent(999), "general_qa"}, // unknown falls to default
	}
	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := tt.intent.String(); got != tt.want {
				t.Errorf("QueryIntent(%d).String() = %s, want %s", tt.intent, got, tt.want)
			}
		})
	}
}

// --- benchmarks ---

func BenchmarkClassify(b *testing.B) {
	qc := NewQueryClassifier()
	query := "How to debug a TypeError in Python with asyncio?"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = qc.Classify(query)
	}
}

func BenchmarkClassify_Empty(b *testing.B) {
	qc := NewQueryClassifier()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = qc.Classify("")
	}
}

func BenchmarkGetWeights(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = GetWeights(IntentTroubleshooting)
	}
}

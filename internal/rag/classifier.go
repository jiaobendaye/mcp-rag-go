package rag

import (
	"regexp"
)

// QueryIntent represents the classified intent of a search query.
type QueryIntent int

const (
	IntentCodeExplanation QueryIntent = iota
	IntentTroubleshooting
	IntentHowTo
	IntentBestPractices
	IntentComparison
	IntentTechnicalDocs
	IntentConceptual
	IntentGeneralQA
)

func (i QueryIntent) String() string {
	switch i {
	case IntentCodeExplanation:
		return "code_explanation"
	case IntentTroubleshooting:
		return "troubleshooting"
	case IntentHowTo:
		return "how_to"
	case IntentBestPractices:
		return "best_practices"
	case IntentComparison:
		return "comparison"
	case IntentTechnicalDocs:
		return "technical_docs"
	case IntentConceptual:
		return "conceptual"
	default:
		return "general_qa"
	}
}

// QueryClassification is the result of classifying a query.
type QueryClassification struct {
	PrimaryIntent QueryIntent
	Confidence    float64
	Keywords      []string
	IsTechnical   bool
	AllIntents    map[QueryIntent]float64
}

// SearchWeights holds tunable vector and keyword weights for hybrid search.
type SearchWeights struct {
	Vector  float64
	Keyword float64
}

// intentConfig holds regex patterns and keywords for one intent.
type intentConfig struct {
	patterns []*regexp.Regexp
	keywords []string
	weight   float64
}

// entityPattern holds compiled entity detection regexes.
type entityPattern struct {
	typeName string
	name     string
	pattern  *regexp.Regexp
}

// QueryClassifier classifies queries into intents for adaptive hybrid search weighting.
// Ported from Python `retrieval/query_classifier.py`.
type QueryClassifier struct {
	intents              map[QueryIntent]intentConfig
	entities             []entityPattern
	nonTechnicalPatterns []*regexp.Regexp
	stopWords            map[string]struct{}
	confidenceThreshold  float64
}

// DefaultWeights returns the default search weights (general_qa / comparison).
var DefaultWeights = SearchWeights{Vector: 0.7, Keyword: 0.3}

// intentWeights maps each intent to its adaptive vector/keyword weights.
var intentWeights = map[QueryIntent]SearchWeights{
	IntentTroubleshooting: {Vector: 0.4, Keyword: 0.6},
	IntentHowTo:           {Vector: 0.65, Keyword: 0.35},
	IntentCodeExplanation: {Vector: 0.75, Keyword: 0.25},
	IntentConceptual:      {Vector: 0.8, Keyword: 0.2},
	IntentBestPractices:   {Vector: 0.8, Keyword: 0.2},
	IntentTechnicalDocs:   {Vector: 0.6, Keyword: 0.4},
	IntentComparison:      {Vector: 0.7, Keyword: 0.3},
	IntentGeneralQA:       {Vector: 0.7, Keyword: 0.3},
}

// GetWeights returns the adaptive search weights for a given intent.
func GetWeights(intent QueryIntent) SearchWeights {
	if w, ok := intentWeights[intent]; ok {
		return w
	}
	return DefaultWeights
}

// NewQueryClassifier creates an initialized QueryClassifier with compiled patterns.
func NewQueryClassifier() *QueryClassifier {
	qc := &QueryClassifier{
		intents:             make(map[QueryIntent]intentConfig),
		stopWords:           buildStopWords(),
		confidenceThreshold: 0.3,
	}

	qc.intents[IntentCodeExplanation] = intentConfig{
		patterns: compilePatterns(
			`(?i)\bexplain\s+(this\s+)?code\b`,
			`(?i)\bwhat\s+does\s+this\s+code\b`,
			`(?i)\bwhat\s+does\s+(this\s+|the\s+)?\w*\s*(function|method|class|do)\b`,
			`(?i)\bhow\s+does\s+(this\s+|the\s+)?\w+\s+work\b`,
			`(?i)\bwalk\s+through\b`,
			`(?i)\bbreak\s+down\b`,
		),
		keywords: []string{"explain", "code", "function", "method", "class", "work"},
		weight:   1.1,
	}

	qc.intents[IntentTroubleshooting] = intentConfig{
		patterns: compilePatterns(
			`(?i)\berror\b`,
			`(?i)\bexception\b`,
			`(?i)\bfailed\b`,
			`(?i)\bnot\s+working\b`,
			`(?i)\bbug\b`,
			`(?i)\bproblem\b`,
			`(?i)\bfix\b`,
			`(?i)\bdebug\b`,
		),
		keywords: []string{"error", "exception", "failed", "bug", "fix", "debug", "problem"},
		weight:   1.3,
	}

	qc.intents[IntentHowTo] = intentConfig{
		patterns: compilePatterns(
			`(?i)\bhow\s+to\b`,
			`(?i)\bhow\s+do\s+i\b`,
			`(?i)\bhow\s+can\s+i\b`,
			`(?i)\bsteps\s+to\b`,
			`(?i)\bguide\s+to\b`,
			`(?i)\btutorial\b`,
		),
		keywords: []string{"how to", "guide", "tutorial", "steps", "build", "create", "implement"},
		weight:   1.0,
	}

	qc.intents[IntentBestPractices] = intentConfig{
		patterns: compilePatterns(
			`(?i)\bbest\s+practice`,
			`(?i)\brecommended\s+(way|approach|method|practices?)\b`,
			`(?i)\bshould\s+i\b`,
			`(?i)\bidiomatic\b`,
			`(?i)\bconvention\b`,
			`(?i)\bavoid\b`,
			`(?i)\bpattern(s)?\b`,
		),
		keywords: []string{"best practice", "recommended", "should", "idiomatic", "avoid", "pattern"},
		weight:   1.4,
	}

	qc.intents[IntentComparison] = intentConfig{
		patterns: compilePatterns(
			`(?i)\bvs\.?\b`,
			`(?i)\bversus\b`,
			`(?i)\bcompare\b`,
			`(?i)\bdifference\s+between\b`,
			`(?i)\bwhich\s+is\s+better\b`,
		),
		keywords: []string{"vs", "versus", "compare", "difference", "better"},
		weight:   1.2,
	}

	qc.intents[IntentTechnicalDocs] = intentConfig{
		patterns: compilePatterns(
			`(?i)\bapi\b`,
			`(?i)\bparameter\b`,
			`(?i)\bargument\b`,
			`(?i)\bconfiguration\b`,
			`(?i)\bsyntax\b`,
			`(?i)\breference\b`,
			`(?i)\bdocumentation\b`,
		),
		keywords: []string{"api", "parameter", "argument", "config", "reference", "docs"},
		weight:   1.0,
	}

	qc.intents[IntentConceptual] = intentConfig{
		patterns: compilePatterns(
			`(?i)\bwhat\s+is\b`,
			`(?i)\bwhat\s+are\b`,
			`(?i)\bwhy\s+(does|is|do)\b`,
			`(?i)\bconcept\b`,
			`(?i)\barchitecture\b`,
			`(?i)\bdesign\s+pattern\b`,
		),
		keywords: []string{"what is", "what are", "why", "concept", "architecture", "design"},
		weight:   1.0,
	}

	// Entity patterns
	qc.entities = append(qc.entities,
		entityPattern{"language", "python", regexp.MustCompile(`(?i)\b(python|py)\b`)},
		entityPattern{"language", "javascript", regexp.MustCompile(`(?i)\b(javascript|js|node\.?js)\b`)},
		entityPattern{"language", "typescript", regexp.MustCompile(`(?i)\b(typescript|ts)\b`)},
		entityPattern{"language", "go", regexp.MustCompile(`(?i)\b(go|golang)\b`)},
		entityPattern{"language", "rust", regexp.MustCompile(`(?i)\brust\b`)},
		entityPattern{"framework", "fastapi", regexp.MustCompile(`(?i)\bfastapi\b`)},
		entityPattern{"framework", "django", regexp.MustCompile(`(?i)\bdjango\b`)},
		entityPattern{"framework", "flask", regexp.MustCompile(`(?i)\bflask\b`)},
		entityPattern{"framework", "react", regexp.MustCompile(`(?i)\breact\b`)},
		entityPattern{"library", "numpy", regexp.MustCompile(`(?i)\bnumpy\b`)},
		entityPattern{"library", "pandas", regexp.MustCompile(`(?i)\bpandas\b`)},
		entityPattern{"library", "faiss", regexp.MustCompile(`(?i)\bfaiss\b`)},
	)

	qc.nonTechnicalPatterns = compilePatterns(
		`(?i)\bhello\b`,
		`(?i)\bhi\b`,
		`(?i)\bthank(s| you)\b`,
		`(?i)\bplease\b`,
		`(?i)\bsorry\b`,
		`(?i)\bhow\s+are\s+you\b`,
	)

	return qc
}

// Classify analyzes a query and returns its intent classification.
func (qc *QueryClassifier) Classify(query string) QueryClassification {
	if query == "" {
		return QueryClassification{
			PrimaryIntent: IntentGeneralQA,
			Confidence:    0.5,
			IsTechnical:   false,
			AllIntents:    map[QueryIntent]float64{IntentGeneralQA: 0.5},
		}
	}

	scores := qc.detectIntents(query)
	primaryIntent := IntentGeneralQA
	maxScore := 0.0
	for intent, score := range scores {
		if score > maxScore {
			maxScore = score
			primaryIntent = intent
		}
	}
	if len(scores) == 0 {
		scores[IntentGeneralQA] = 0.3
		maxScore = 0.3
	}

	keywords := qc.extractKeywords(query)

	return QueryClassification{
		PrimaryIntent: primaryIntent,
		Confidence:    maxScore,
		Keywords:      keywords,
		IsTechnical:   qc.isTechnicalQuery(query),
		AllIntents:    scores,
	}
}

func (qc *QueryClassifier) detectIntents(query string) map[QueryIntent]float64 {
	scores := make(map[QueryIntent]float64)

	for intent, cfg := range qc.intents {
		var score float64

		patternMatches := 0
		for _, p := range cfg.patterns {
			if p.MatchString(query) {
				patternMatches++
			}
		}
		if patternMatches > 0 {
			score += minFloat(float64(patternMatches)*0.3, 0.9)
		}

		keywordMatches := 0
		queryLower := toLower(query)
		for _, kw := range cfg.keywords {
			if containsWord(queryLower, kw) {
				keywordMatches++
			}
		}
		if keywordMatches > 0 {
			score += minFloat(float64(keywordMatches)*0.1, 0.4)
		}

		if score > 0 {
			score = minFloat(score*cfg.weight, 1.0)
			if score >= qc.confidenceThreshold {
				scores[intent] = score
			}
		}
	}

	return scores
}

func (qc *QueryClassifier) isTechnicalQuery(query string) bool {
	for _, p := range qc.nonTechnicalPatterns {
		if p.MatchString(query) {
			return false
		}
	}

	for _, ep := range qc.entities {
		if ep.pattern.MatchString(query) {
			return true
		}
	}
	return true
}

func (qc *QueryClassifier) extractKeywords(query string) []string {
	tokenRe := regexp.MustCompile(`\b\w+\b`)
	tokens := tokenRe.FindAllString(toLower(query), -1)
	var keywords []string
	for _, token := range tokens {
		if len(token) > 2 {
			if _, isStop := qc.stopWords[token]; !isStop {
				keywords = append(keywords, token)
			}
		}
	}
	if len(keywords) > 10 {
		keywords = keywords[:10]
	}
	return keywords
}

// --- helpers ---

func compilePatterns(patterns ...string) []*regexp.Regexp {
	compiled := make([]*regexp.Regexp, len(patterns))
	for i, p := range patterns {
		compiled[i] = regexp.MustCompile(p)
	}
	return compiled
}

func buildStopWords() map[string]struct{} {
	words := []string{"the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
		"for", "with", "is", "are", "was", "were", "this", "that", "these", "those",
		"i", "me", "my", "you"}
	m := make(map[string]struct{}, len(words))
	for _, w := range words {
		m[w] = struct{}{}
	}
	return m
}

func toLower(s string) string {
	b := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 32
		}
		b[i] = c
	}
	return string(b)
}

func containsWord(s, word string) bool {
	// Simple substring check on lowercased input
	for i := 0; i <= len(s)-len(word); i++ {
		if s[i:i+len(word)] == word {
			return true
		}
	}
	return false
}

func minFloat(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

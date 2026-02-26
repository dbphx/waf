package features

import (
	"math"
	"net/url"
	"regexp"
	"strings"
)

type Metadata struct {
	NgramRange  []int     `json:"ngram_range"`
	MaxFeatures int       `json:"max_features"`
	Vocabulary  []string  `json:"vocabulary"`
	IDF         []float64 `json:"idf"`
	Keywords    []string  `json:"keywords"`
}

func ParseHTTPRequest(payload string) map[string]string {
	row := map[string]string{
		"method":  "GET",
		"path":    "/",
		"query":   "",
		"headers": "",
		"body":    "",
	}
	if payload == "" {
		return row
	}

	// 1. Handle Method/Path pattern (e.g., 'GET /path?q=v {"body": 1}')
	prefixes := []string{"GET ", "POST ", "PUT ", "DELETE "}
	hasPrefix := false
	for _, p := range prefixes {
		if strings.HasPrefix(payload, p) {
			hasPrefix = true
			break
		}
	}

	if hasPrefix {
		parts := strings.SplitN(payload, " ", 3)
		row["method"] = parts[0]
		if len(parts) > 1 {
			urlPart := parts[1]
			u, err := url.Parse(urlPart)
			if err == nil {
				row["path"] = u.Path
				row["query"] = u.RawQuery
			} else {
				row["path"] = urlPart
			}
		}
		if len(parts) > 2 {
			row["body"] = parts[2]
		}
	} else if strings.HasPrefix(payload, "{") || strings.HasPrefix(payload, "[") {
		// 2. Handle Body pattern
		row["method"] = "POST"
		row["body"] = payload
	} else if (strings.Contains(payload, "=") || strings.Contains(payload, "&")) && !strings.Contains(payload, " ") {
		// 3. Handle Query pattern
		row["query"] = payload
	} else {
		// 4. Fallback
		row["path"] = payload
	}

	return row
}

func CleanText(text string) string {
	text = strings.ToLower(text)

	// URL decode (2 passes)
	decoded, err := url.PathUnescape(text)
	if err == nil {
		text = decoded
		decoded, err = url.PathUnescape(text)
		if err == nil {
			text = decoded
		}
	}

	// Normalize whitespace
	re := regexp.MustCompile(`\s+`)
	text = re.ReplaceAllString(text, " ")

	return strings.TrimSpace(text)
}

func ExtractText(row map[string]string) string {
	fields := []string{"path", "query", "headers", "body"}
	var vals []string
	for _, f := range fields {
		v := strings.TrimSpace(row[f])
		if v != "" && strings.ToLower(v) != "nan" {
			vals = append(vals, v)
		}
	}
	return strings.Join(vals, " ")
}

func GetNGrams(text string, minN, maxN int) map[string]int {
	ngrams := make(map[string]int)
	chars := []rune(text)
	n := len(chars)

	for i := 0; i < n; i++ {
		for length := minN; length <= maxN; length++ {
			if i+length <= n {
				gram := string(chars[i : i+length])
				ngrams[gram]++
			}
		}
	}
	return ngrams
}

func CalcEntropy(text string) float64 {
	if len(text) == 0 {
		return 0
	}
	counts := make(map[rune]int)
	for _, r := range text {
		counts[r]++
	}
	var entropy float64
	total := float64(len(text))
	for _, count := range counts {
		p := float64(count) / total
		entropy -= p * math.Log(p)
	}
	return entropy
}

func GenerateFeatureVector(text string, meta *Metadata) []float64 {
	// 1. TF-IDF Part
	cleaned := CleanText(text)
	ngrams := GetNGrams(cleaned, meta.NgramRange[0], meta.NgramRange[1])

	vector := make([]float64, len(meta.Vocabulary))
	for i, term := range meta.Vocabulary {
		count := ngrams[term]
		if count > 0 {
			// Sklearn TF-IDF formula (raw counts by default, then norm)
			vector[i] = float64(count) * meta.IDF[i]
		}
	}

	// L2 Normalization
	var sumSq float64
	for _, v := range vector {
		sumSq += v * v
	}
	if sumSq > 0 {
		norm := math.Sqrt(sumSq)
		for i := range vector {
			vector[i] /= norm
		}
	}

	// 2. Statistical Features
	// Length (scaled)
	vector = append(vector, float64(len(text))/1000.0)
	// Entropy (scaled)
	vector = append(vector, CalcEntropy(text)/10.0)
	// Keyword frequencies
	for _, kw := range meta.Keywords {
		count := strings.Count(text, kw)
		vector = append(vector, float64(count)/float64(len(text)+1))
	}

	return vector
}

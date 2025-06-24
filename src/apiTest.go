package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"

	"strings"
)

type WikiSummary struct {
	Title   string `json:"title"`
	Extract string `json:"extract"`
}

type WikiArticle struct {
	Parse struct {
		Title  string            `json:"title"`
		Text   map[string]string `json:"text"`
		PageID int               `json:"pageid"`
	} `json:"parse"`
}

type WikiSearchResult struct {
	Query struct {
		Search []struct {
			Title   string `json:"title"`
			Snippet string `json:"snippet"`
		} `json:"search"`
	} `json:"query"`
}

func FetchWikiArticleText(articleTitle string) {
	// URL encode the article title
	encodedTitle := url.QueryEscape(articleTitle)

	// Use Wikipedia's extract API to get clean text content
	apiURL := fmt.Sprintf("https://en.wikipedia.org/w/api.php?action=query&format=json&titles=%s&prop=extracts&exintro=false&explaintext=true", encodedTitle)

	resp, err := http.Get(apiURL)
	if err != nil {
		fmt.Printf("Error fetching article '%s': %v\n", articleTitle, err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		fmt.Printf("Error: Wikipedia API returned status %d for article '%s'\n", resp.StatusCode, articleTitle)
		return
	}

	// Parse the JSON response
	var result struct {
		Query struct {
			Pages map[string]struct {
				Title   string `json:"title"`
				Extract string `json:"extract"`
			} `json:"pages"`
		} `json:"query"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		fmt.Printf("Error decoding response: %v\n", err)
		return
	}

	// Get the first (and only) page from the response
	for _, page := range result.Query.Pages {
		if page.Extract == "" {
			fmt.Printf("No content found for article '%s'\n", articleTitle)
			return
		}

		fmt.Printf("=== Full Article: %s ===\n", page.Title)
		fmt.Printf("Content Length: %d characters\n", len(page.Extract))

		// Show first 2000 characters for preview
		if len(page.Extract) > 2000 {
			fmt.Printf("First 2000 characters:\n---\n%s...\n---\n", page.Extract[:2000])
		} else {
			fmt.Printf("Full content:\n---\n%s\n---\n", page.Extract)
		}
		return
	}

	fmt.Printf("Article '%s' not found\n", articleTitle)
}

// Simple HTML tag stripper function
func stripHTMLTags(html string) string {
	// This is a basic implementation - for production use, consider using a proper HTML parser
	var result strings.Builder
	inTag := false

	for _, char := range html {
		if char == '<' {
			inTag = true
		} else if char == '>' {
			inTag = false
		} else if !inTag {
			result.WriteRune(char)
		}
	}

	// Clean up extra whitespace
	text := result.String()
	text = strings.ReplaceAll(text, "\n\n\n", "\n\n")
	text = strings.TrimSpace(text)

	return text
}

// SearchWikiArticle searches for articles matching the query and returns the best match
func SearchWikiArticle(searchQuery string) (string, error) {
	// URL encode the search query
	encodedQuery := url.QueryEscape(searchQuery)

	// Use Wikipedia's correct search API endpoint
	searchURL := fmt.Sprintf("https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=%s&format=json", encodedQuery)

	resp, err := http.Get(searchURL)
	if err != nil {
		return "", fmt.Errorf("error searching for '%s': %v", searchQuery, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("wikipedia search API returned status %d for query '%s'", resp.StatusCode, searchQuery)
	}

	var searchResult struct {
		Query struct {
			Search []struct {
				Title string `json:"title"`
			} `json:"search"`
		} `json:"query"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&searchResult); err != nil {
		return "", fmt.Errorf("error decoding search results: %v", err)
	}

	if len(searchResult.Query.Search) == 0 {
		return "", fmt.Errorf("no articles found for query '%s'", searchQuery)
	}

	// Return the title of the first (best) match
	return searchResult.Query.Search[0].Title, nil
}

// SearchAndFetchWikiArticle searches for an article and then fetches its full text
func SearchAndFetchWikiArticle(searchQuery string) {
	fmt.Printf("Searching for articles matching: '%s'\n", searchQuery)

	// First, search for the article
	articleTitle, err := SearchWikiArticle(searchQuery)
	if err != nil {
		fmt.Printf("Search failed: %v\n", err)
		return
	}

	fmt.Printf("Found article: '%s'\n", articleTitle)
	fmt.Println("Fetching full article content...\n")

	// Then fetch the full article
	FetchWikiArticleText(articleTitle)
}

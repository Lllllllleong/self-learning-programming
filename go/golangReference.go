package main

import "fmt"

/*
============================================================
GOLANG REFERENCE & SCRATCHPAD
============================================================

This file is used for quick testing, debugging, and
experimenting with Go code snippets.

HOW TO USE:
1. Write your test code in the main() function below
2. Run: go run golangReference.go
3. Check output in terminal
4. Modify and rerun as needed

TIPS:
- Use fmt.Println() for output
- Use fmt.Printf() for formatted output
- Remember to keep imports at the top if you add packages
- Comment out or replace code as you test different things

============================================================
*/

func main() {
	s := "Hello, 世界"
	for i, r := range s {
		fmt.Printf("Index: %d, Rune: %c\n", i, r)
	}

	runes := []rune(s)
	for i, r := range runes {
		fmt.Printf("Index: %d, Rune: %c\n", i, r)
	}

}

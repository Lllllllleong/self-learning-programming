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
	// "Hi" is 2 bytes. "👋" is 4 bytes. Total 6 bytes.
	str := "Hi 👋"

	// ❌ BAD: Slicing bytes
	// This tries to take the first 3 bytes.
	// It gets "H", "i", and HALF of the hand emoji.
	fmt.Println(str[:4])

	// ✅ GOOD: Slicing Runes
	// Convert to runes -> Slice -> Convert back to string
	runes := []rune(str)
	safeSub := string(runes[:4])
	fmt.Println(safeSub)
	fmt.Println(string(runes[0:1]))
}

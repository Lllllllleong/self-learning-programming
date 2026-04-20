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

KEYBOARD SHORTCUTS:
- Navigate cursor backwards/forwards through prior locations:
  LALT + RALT + A/D (Left arrow/Right arrow)

============================================================
*/

// ============================================================
// MAPS REFERENCE
// ============================================================

func mapsReference() {
	// --- Declaration & Initialization ---
	m1 := map[string]int{}               // empty map literal (ready to use)
	m2 := make(map[string]int)           // same, via make
	m3 := map[string]int{"a": 1, "b": 2} // map literal with values
	var m4 map[string]int                // nil map — reads ok, writes PANIC
	_, _, _, _ = m1, m2, m3, m4

	m := map[string]int{"x": 0, "y": 10}

	// --- Basic Operations ---
	m["key"] = 42        // set / update
	val := m["key"]      // get (returns zero value if key absent, not an error)
	delete(m, "key")     // remove a key (no-op if missing)
	length := len(m)     // number of key-value pairs
	_, _ = val, length

	// --- The "ok" Idiom ---
	// Two-value form distinguishes "key absent" from "zero value stored":
	v, ok := m["x"]  // ok == true  → key exists,  v == 0
	v, ok = m["z"]   // ok == false → key missing, v == 0 (zero value)
	_ = v

	if val, ok := m["y"]; ok {
		fmt.Println("found:", val)
	} else {
		fmt.Println("key not present")
	}
	_ = ok

	// --- Iterating ---
	for k, v := range m { fmt.Println(k, v) } // order is random every run
	for k := range m    { fmt.Println(k) }    // keys only
	for _, v := range m { fmt.Println(v) }    // values only

	// --- Common Patterns ---

	// Set — map[T]struct{} uses no extra memory for the value:
	seen := map[string]struct{}{}
	seen["x"] = struct{}{}
	if _, ok := seen["x"]; ok {
		fmt.Println("exists")
	}

	// Counting occurrences — zero value of int is 0, so ++ works immediately:
	words := []string{"go", "is", "go"}
	freq := map[string]int{}
	for _, word := range words {
		freq[word]++
	}

	// Grouping (map of slices) — append to nil slice is safe:
	groups := map[string][]int{}
	groups["odd"] = append(groups["odd"], 1, 3, 5)

	// Nested maps:
	nested := map[string]map[string]int{}
	nested["outer"] = map[string]int{"inner": 99}

	_ = freq
	_ = groups
	_ = nested

	// --- Gotchas ---
	// Maps are reference types — assigning copies the reference, not the data.
	// Maps are NOT safe for concurrent use — use sync.RWMutex or sync.Map.
	// Iteration order is intentionally randomised; never rely on it.
	// You cannot take the address of a map value: &m["key"] does not compile.
}

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

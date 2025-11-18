package medium

/*
============================================================

============================================================
Time Complexity: O()
Space Complexity: O()
*/

/*
============================================================
1529. Minimum Suffix Flips
============================================================
Time Complexity: O(n)
Space Complexity: O(1)
*/
func minFlips(target string) int {
	currentRune := '0'
	output := 0
	for _, rune := range target {
		if rune != currentRune {
			output++
		}
		currentRune = rune
	}
	return output
}

/*
============================================================
3557. Find Maximum Number of Non Intersecting Substrings
============================================================
Time Complexity: O(n)
Space Complexity: O(n)
*/
func maxSubstrings(word string) int {
	runes := []rune(word)
	n := len(runes)
	numChars := 26
	charIndices := make([][]int, numChars)
	dpSlice := make([]int, n+1)
	for i := n - 1; i >= 0; i-- {
		currentCharIndices := charIndices[(runes[i] - 'a')]
		for j := len(currentCharIndices) - 1; j >= 0; j-- {
			if (currentCharIndices[j] - i) < 3 {
				continue
			}
			currentBest := dpSlice[currentCharIndices[j]+1] + 1
			if dpSlice[i] < currentBest {
				dpSlice[i] = currentBest
			}
			break
		}
		if dpSlice[i] < dpSlice[i+1] {
			dpSlice[i] = dpSlice[i+1]
		}
		charIndices[(runes[i] - 'a')] = append(charIndices[(runes[i]-'a')], i)
	}
	return dpSlice[0]
}

/*
============================================================
3597. Partition String
============================================================
Time Complexity: O(n^2)
Space Complexity: O(n)
*/
func partitionString(s string) []string {
	runes := []rune(s)
	n, i := len(runes), 0
	seenSet := make(map[string]struct{})
	output := []string{}
	for i < n {
		j := i + 1
		for j <= n {
			currentSubstring := string(runes[i:j])
			if _, ok := seenSet[currentSubstring]; ok {
				j++
			} else {
				seenSet[currentSubstring] = struct{}{}
				output = append(output, currentSubstring)
				break
			}
		}
		i = j
	}
	return output
}

/*
============================================================
3675. Minimum Operations to Transform String
============================================================
Time Complexity: O(n)
Space Complexity: O(1)
*/
func minOperations(s string) int {
	numOperations := 0
	for _, rune := range s {
		currentCost := int(26-(rune-'a')) % 26
		if currentCost > numOperations {
			numOperations = currentCost
		}
	}
	return numOperations
}

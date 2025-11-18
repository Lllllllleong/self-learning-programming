package easy

import "sort"

/*
============================================================

============================================================
Time Complexity: O()
Space Complexity: O()
*/

/*
============================================================
3541. Find Most Frequent Vowel and Consonant
============================================================
Time Complexity: O(n)
Space Complexity: O(k) where k is the number of unique characters
*/
func maxFreqSum(s string) int {
	vowelSet := map[rune]bool{
		'a': true,
		'e': true,
		'i': true,
		'o': true,
		'u': true,
	}
	var vowelCount, consonantCount int
	runeMap := make(map[rune]int)
	for _, rune := range s {
		runeMap[rune]++
		if vowelSet[rune] && runeMap[rune] > vowelCount {
			vowelCount = runeMap[rune]
		}
		if !vowelSet[rune] && runeMap[rune] > consonantCount {
			consonantCount = runeMap[rune]
		}
	}
	return vowelCount + consonantCount
}

/*
============================================================
3731. Find Missing Elements
============================================================
Time Complexity: O(n log n)
Space Complexity: O(k) where k is the number of missing elements
*/
func findMissingElements(nums []int) []int {
	sort.Ints(nums)
	output := []int{}
	counter := nums[0]
	for _, v := range nums {
		for counter < v {
			output = append(output, counter)
			counter++
		}
		counter++
	}
	return output
}

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

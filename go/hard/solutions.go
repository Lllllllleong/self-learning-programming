package hard

import (
	"cmp"
	// "iter"
	"math"

	// "iter"
	// "math"
	"slices"
	// "sync"
)

/*
============================================================

============================================================
Time Complexity: O()
Space Complexity: O()
*/

/*
============================================================
1473. Paint House III
============================================================
Time Complexity: O(m * target * n^2)
Space Complexity: O(target * n)
*/
func minCost(houses []int, cost [][]int, m int, n int, target int) int {
	baseState := make([][]int, target+1)
	baseState[0] = make([]int, n)
	for i := 1; i < len(baseState); i++ {
		baseState[i] = make([]int, n)
		for j := range baseState[i] {
			baseState[i][j] = math.MaxInt32
		}
	}
	dpState := copy2DSlice(baseState)
	for h, house := range houses {
		if house != 0 {
			currentColour := house - 1
			currentState := copy2DSlice(baseState)
			for i := target; i > 0; i-- {
				for j := 0; j < n; j++ {
					if j == currentColour {
						currentState[i][currentColour] = min(currentState[i][currentColour], dpState[i][j])
						if i == 1 {
							currentState[i][currentColour] = min(currentState[i][currentColour], dpState[i-1][j])
						}

					} else {
						currentState[i][currentColour] = min(currentState[i][currentColour], dpState[i-1][j])
					}
				}
			}
			dpState = currentState
		} else {
			for i := target; i > 0; i-- {
				for j := 0; j < n; j++ {
					if dpState[i][j] != math.MaxInt32 {
						dpState[i][j] += cost[h][j]
					}
					for k := 0; k < n; k++ {
						if j == k {
							if i == 1 && dpState[i-1][k] != math.MaxInt32 {
								dpState[i][j] = min(dpState[i][j], dpState[i-1][k]+cost[h][j])
							}
							continue
						}

						if dpState[i-1][k] != math.MaxInt32 {
							dpState[i][j] = min(dpState[i][j], dpState[i-1][k]+cost[h][j])
						}
					}
				}
			}
		}
		if h == 0 {
			for i := range dpState[0] {
				dpState[0][i] = math.MaxInt32
				baseState[0][i] = math.MaxInt32
			}
		}
	}
	output := math.MaxInt32
	for _, v := range dpState[target] {
		output = min(output, v)
	}
	if output == math.MaxInt32 {
		output = -1
	}
	return output
}

func copy2DSlice(src [][]int) [][]int {
	dst := make([][]int, len(src))
	for i := range src {
		dst[i] = make([]int, len(src[i]))
		copy(dst[i], src[i])
	}
	return dst
}

/*
============================================================
992. Subarrays with K Different Integers
============================================================
Time Complexity: O(n)
Space Complexity: O(n)
*/
func subarraysWithKDistinct(nums []int, k int) int {
	return subarraysWithAtMostKDistinct(nums, k) - subarraysWithAtMostKDistinct(nums, k-1)
}

func subarraysWithAtMostKDistinct(nums []int, k int) int {
	count := make(map[int]int)
	left := 0
	output := 0

	for right, num := range nums {
		count[num]++
		for len(count) > k {
			count[nums[left]]--
			if count[nums[left]] == 0 {
				delete(count, nums[left])
			}
			left++
		}
		output += right - left + 1
	}
	return output
}

/*
============================================================
2449. Minimum Number of Operations to Make Arrays Similar
============================================================
Time Complexity: O()
Space Complexity: O()
*/
func makeSimilar(nums []int, target []int) int64 {
	var output int64
	cache := make([][]int, 4)
	for i := range cache {
		cache[i] = []int{}
	}
	for i := range nums {
		cache[nums[i]%2] = append(cache[nums[i]%2], nums[i])
		cache[2+target[i]%2] = append(cache[2+target[i]%2], target[i])
	}
	for i := range cache {
		slices.Sort(cache[i])
	}
	for i := range 2 {
		for j := range cache[i] {
			num, tar := cache[i][j], cache[2+i][j]
			if tar > num {
				ops := (tar - num) / 2
				output += int64(ops)
			}
		}
	}
	return output
}

/*
============================================================
2366. Minimum Replacements to Sort the Array
============================================================
Time Complexity: O(n)
Space Complexity: O(1)
*/
func minimumReplacement(nums []int) int64 {
	var output int64 = 0
	n := len(nums)
	prior := nums[len(nums)-1]
	for i := n - 2; i >= 0; i-- {
		num := nums[i]
		if num > prior {
			k := (num + prior - 1) / prior
			output += int64(k - 1)
			prior = num / k
		} else {
			prior = num
		}
	}
	return output
}

/*
============================================================
3414. Maximum Score of Non-overlapping Intervals
============================================================
Time Complexity: O(n log n)
Space Complexity: O(n)
*/
const kSize = 5

type Endpoint struct {
	maxScores       [kSize]int
	intervalIndices [kSize][]int
}

func shouldUpdate(currScore, newScore int, currIndices, newIndices []int) bool {
	if newScore > currScore {
		return true
	}
	return newScore == currScore && slices.Compare(newIndices, currIndices) < 0
}

func mergeEndpoints(src, dest *Endpoint) {
	for i := 0; i < kSize; i++ {
		if shouldUpdate(dest.maxScores[i], src.maxScores[i], dest.intervalIndices[i], src.intervalIndices[i]) {
			dest.maxScores[i] = src.maxScores[i]
			dest.intervalIndices[i] = slices.Clone(src.intervalIndices[i])
		}
	}
}

func maximumWeight(intervals [][]int) []int {
	uniqueEnds := make(map[int]struct{})
	for i, interval := range intervals {
		uniqueEnds[interval[1]] = struct{}{}
		intervals[i] = append(intervals[i], i)
	}
	sortedEnds := []int{}
	for k := range uniqueEnds {
		sortedEnds = append(sortedEnds, k)
	}
	slices.Sort(sortedEnds)
	epMap := make(map[int]*Endpoint, len(sortedEnds))
	for _, ep := range sortedEnds {
		epMap[ep] = &Endpoint{}
	}
	slices.SortFunc(intervals, func(a, b []int) int {
		return cmp.Or(
			cmp.Compare(a[0], b[0]),
			cmp.Compare(a[1], b[1]),
			cmp.Compare(a[3], b[3]),
		)
	})
	currentEP := &Endpoint{}
	epIdx := 0
	for _, interval := range intervals {
		start, end, value, index := interval[0], interval[1], interval[2], interval[3]
		for epIdx < len(sortedEnds) && sortedEnds[epIdx] < start {
			mergeEndpoints(epMap[sortedEnds[epIdx]], currentEP)
			epIdx++
		}
		targetEP := epMap[end]
		for k := kSize - 1; k > 0; k-- {
			prevScore := currentEP.maxScores[k-1]
			potentialScore := prevScore + value
			if k > 1 && currentEP.intervalIndices[k-1] == nil && prevScore == 0 {
				continue
			}
			newChain := make([]int, len(currentEP.intervalIndices[k-1])+1)
			copy(newChain, currentEP.intervalIndices[k-1])
			newChain[len(newChain)-1] = index
			slices.Sort(newChain)

			if shouldUpdate(targetEP.maxScores[k], potentialScore, targetEP.intervalIndices[k], newChain) {
				targetEP.maxScores[k] = potentialScore
				targetEP.intervalIndices[k] = newChain
			}
		}
	}
	for epIdx < len(sortedEnds) {
		mergeEndpoints(epMap[sortedEnds[epIdx]], currentEP)
		epIdx++
	}
	maxScore := -1
	var output []int
	for i, v := range currentEP.maxScores {
		if shouldUpdate(maxScore, v, output, currentEP.intervalIndices[i]) {
			maxScore = v
			output = currentEP.intervalIndices[i]
		}
	}

	return output
}

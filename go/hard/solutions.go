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
2977. Minimum Cost to Convert String II
============================================================
Time Complexity: O(m³ + n×L×s) where m = number of unique strings in original+changed,

	n = len(source), L = number of unique string lengths, s = max string length

Space Complexity: O(m² + n) for cost map and DP array
*/
func minimumCost(source string, target string, original []string, changed []string, cost []int) int64 {
	costMap := make(map[string]map[string]int64)
	setCost := func(from, to string, c int64) {
		if costMap[from] == nil {
			costMap[from] = make(map[string]int64)
		}
		if existing, exists := costMap[from][to]; !exists || c < existing {
			costMap[from][to] = c
		}
	}
	for i := range original {
		setCost(original[i], changed[i], int64(cost[i]))
	}
	allStrings := []string{}
	stringSet := make(map[string]struct{})
	for _, s := range original {
		if _, exists := stringSet[s]; !exists {
			stringSet[s] = struct{}{}
			allStrings = append(allStrings, s)
		}
	}
	for _, s := range changed {
		if _, exists := stringSet[s]; !exists {
			stringSet[s] = struct{}{}
			allStrings = append(allStrings, s)
		}
	}
	stringLengthSet := make(map[int]struct{})
	for _, v := range allStrings {
		stringLengthSet[len(v)] = struct{}{}
	}
	stringLengths := make([]int, 0, len(stringLengthSet))
	for k := range stringLengthSet {
		stringLengths = append(stringLengths, k)
	}
	slices.Sort(stringLengths)
	for _, k := range allStrings {
		for _, i := range allStrings {
			if i == k || costMap[i] == nil {
				continue
			}
			costIK, hasIK := costMap[i][k]
			if !hasIK {
				continue
			}

			for _, j := range allStrings {
				if i == j || j == k || costMap[k] == nil {
					continue
				}
				costKJ, hasKJ := costMap[k][j]
				if !hasKJ {
					continue
				}

				newCost := costIK + costKJ
				existingCost, hasIJ := costMap[i][j]
				if !hasIJ || newCost < existingCost {
					setCost(i, j, newCost)
				}
			}
		}
	}
	getCost := func(from, to string) int64 {
		if from == to {
			return 0
		}
		if costMap[from] == nil {
			return math.MaxInt64
		}
		if c, exists := costMap[from][to]; exists {
			return c
		}
		return math.MaxInt64
	}
	n := len(source)
	dp := make([]int64, n+1)
	for i := range dp {
		dp[i] = math.MaxInt64
	}
	dp[0] = 0
	for i := 0; i < n; i++ {
		if dp[i] == math.MaxInt64 || dp[i] >= dp[n] {
			continue
		}
		if source[i] == target[i] {
			dp[i+1] = min(dp[i+1], dp[i])
		}
		for _, length := range stringLengths {
			if i+length >= n+1 {
				break
			}
			srcSubstr := source[i : i+length]
			tgtSubstr := target[i : i+length]
			transformCost := getCost(srcSubstr, tgtSubstr)
			if transformCost != math.MaxInt64 {
				dp[i+length] = min(dp[i+length], dp[i]+transformCost)
			}
		}
	}
	if dp[n] == math.MaxInt64 {
		return -1
	}
	return dp[n]
}

/*
============================================================
1411. Number of Ways to Paint N × 3 Grid
============================================================
Time Complexity: O(n)
Space Complexity: O(1)
*/
func numOfWays(n int) int {
	const Mod = 1_000_000_007
	aba, abc := 6, 6
	for i := 1; i < n; i++ {
		newAba := (3*aba + 2*abc) % Mod
		newAbc := (2*aba + 2*abc) % Mod
		aba, abc = newAba, newAbc
	}
	return (aba + abc) % Mod
}

/*
============================================================
84. Largest Rectangle in Histogram
============================================================
Time Complexity: O(n)
Space Complexity: O(n)
*/
func largestRectangleArea(heights []int) int {
	heights = append(heights, 0)
	stack := []int{}
	maxArea := 0
	for i, h := range heights {
		for len(stack) > 0 && heights[stack[len(stack)-1]] >= h {
			height := heights[stack[len(stack)-1]]
			stack = stack[:len(stack)-1]
			width := 0
			if len(stack) == 0 {
				width = i
			} else {
				width = i - stack[len(stack)-1] - 1
			}
			maxArea = max(maxArea, height*width)
		}
		stack = append(stack, i)
	}
	return maxArea
}

/*
============================================================
1520. Maximum Number of Non-Overlapping Substrings
============================================================
Time Complexity: O(n)
Space Complexity: O(1)
*/
func maxNumOfSubstrings(s string) []string {
	n := len(s)
	firstSeen := make([]int, 26)
	lastSeen := make([]int, 26)
	for i := range firstSeen {
		firstSeen[i] = -1
	}
	for i := 0; i < n; i++ {
		charIndex := s[i] - 'a'
		if firstSeen[charIndex] == -1 {
			firstSeen[charIndex] = i
		}
		lastSeen[charIndex] = i
	}
	intervals := [][]int{}
	for i := 0; i < 26; i++ {
		if firstSeen[i] == -1 {
			continue
		}
		leftBound, rightBound := firstSeen[i], lastSeen[i]
		isValid := true
		for j := leftBound; j <= rightBound; j++ {
			charIndex := s[j] - 'a'
			if firstSeen[charIndex] < leftBound {
				isValid = false
				break
			}
			if lastSeen[charIndex] > rightBound {
				rightBound = lastSeen[charIndex]
			}
		}
		if isValid {
			intervals = append(intervals, []int{leftBound, rightBound})
		}
	}
	slices.SortFunc(intervals, func(a, b []int) int {
		return cmp.Compare(a[1], b[1])
	})
	output := []string{}
	previousEnd := -1
	for _, interval := range intervals {
		leftBound, rightBound := interval[0], interval[1]
		if leftBound > previousEnd {
			output = append(output, s[leftBound:rightBound+1])
			previousEnd = rightBound
		}
	}
	return output
}

/*
============================================================
2045. Second Minimum Time to Reach Destination
============================================================
Time Complexity: O(V + E) where V is number of nodes, E is number of edges
Space Complexity: O(V + E)
*/
type pair struct {
	node, time int
}

func secondMinimum(n int, edges [][]int, time int, change int) int {
	adj := make([][]int, n+1)
	for _, e := range edges {
		u, v := e[0], e[1]
		adj[u] = append(adj[u], v)
		adj[v] = append(adj[v], u)
	}
	dist1 := make([]int, n+1)
	dist2 := make([]int, n+1)
	for i := range dist1 {
		dist1[i], dist2[i] = -1, -1
	}
	queue := make([]pair, 0, n*2)
	queue = append(queue, pair{1, 0})
	dist1[1] = 0
	head := 0
	for head < len(queue) {
		curr := queue[head]
		head++

		// Traffic light calculation
		nextTime := curr.time
		if (nextTime/change)%2 == 1 {
			nextTime = change*((nextTime/change)+1) + time
		} else {
			nextTime += time
		}

		for _, neighbor := range adj[curr.node] {
			if dist1[neighbor] == -1 {
				dist1[neighbor] = nextTime
				queue = append(queue, pair{neighbor, nextTime})
			} else if dist2[neighbor] == -1 && dist1[neighbor] < nextTime {
				if neighbor == n {
					return nextTime
				}
				dist2[neighbor] = nextTime
				queue = append(queue, pair{neighbor, nextTime})
			}
		}
	}
	return 0
}

/*
============================================================
1955. Count Number of Special Subsequences
============================================================
Time Complexity: O(n)
Space Complexity: O(1)
*/
func countSpecialSubsequences(nums []int) int {
	const M = 1_000_000_007
	dpSlice := make([]int, 3)
	for _, v := range nums {
		dpSlice[v] = (dpSlice[v] * 2) % M
		if v > 0 {
			dpSlice[v] += dpSlice[v-1]
			dpSlice[v] %= M
		} else {
			dpSlice[v]++
			dpSlice[v] %= M
		}
	}
	return dpSlice[2]
}

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

package hard

import (
	"cmp"
	"slices"
)

/*
============================================================

============================================================
Time Complexity: O()
Space Complexity: O()
*/

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

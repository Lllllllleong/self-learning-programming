package main

import (
	"fmt"
	"sort"

)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func main() {
	fmt.Println("=== Double-ended Queue Examples ===")

	// Example 1: Using container/list as a deque
	dequeExample()

	// Example 2: Using slice-based deque
	sliceDequeExample()

	fmt.Println("\n=== Wikipedia Search ===")
	// Test the full article text function
	SearchAndFetchWikiArticle("Haskell")

}

func twoSum(nums []int, target int) []int {
	numberMap := make(map[int]int)
	for currentIndex, currentNumber := range nums {
		requiredNumber := target - currentNumber
		if secondIndex, exists := numberMap[requiredNumber]; exists {
			return []int{secondIndex, currentIndex}
		}
		numberMap[currentNumber] = currentIndex
	}
	return nil
}

func fizzBuzz(n int) []string {
	outputSlice := []string{}
	for i := 1; i <= n; i++ {
		condition1 := (i % 3) == 0
		condition2 := (i % 5) == 0
		switch {
		case condition1 && condition2:
			outputSlice = append(outputSlice, "FizzBuzz")
		case condition1:
			outputSlice = append(outputSlice, "Fizz")
		case condition2:
			outputSlice = append(outputSlice, "Buzz")
		default:
			outputSlice = append(outputSlice, fmt.Sprintf("%d", i))
		}
	}
	return outputSlice
}

func scoreOfString(s string) int {
	output := 0
	for i := 0; i < len(s)-1; i++ {
		difference := int(s[i]) - int(s[i+1])
		if difference < 0 {
			output += -difference
		} else {
			output += difference
		}
	}
	return output
}

func minDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	left := minDepth(root.Left)
	right := minDepth(root.Right)
	return 1 + min(left, right)
}

func min(a, b int) int {
	if a == 0 {
		return b
	}
	if b == 0 {
		return a
	}
	if a > b {
		return b
	}
	return a
}

func prefixCount(words []string, pref string) int {
	output := 0
	prefixLength := len(pref)
	for _, word := range words {
		if (len(word) >= prefixLength) && (word[:prefixLength] == pref) {
			output++
		}
	}
	return output
}

func isSymmetric(root *TreeNode) bool {
	if root == nil {
		return true
	}
	return isSymmetric2(root.Left, root.Right)
}

func isSymmetric2(root1, root2 *TreeNode) bool {
	condition1 := root1 == nil
	condition2 := root2 == nil
	if condition1 && condition2 {
		return true
	}
	if (condition1 && !condition2) || (!condition1 && condition2) {
		return false
	}
	if root1.Val != root2.Val {
		return false
	}
	return (isSymmetric2(root1.Right, root2.Left)) && (isSymmetric2(root1.Left, root2.Right))
}

func validPalindrome(s string) bool {
	left, right := 0, len(s)-1
	if right == 0 {
		return true
	}
	flag := false
	for left <= right {
		if s[left] != s[right] {
			if flag {
				return false
			}
			flag = true
		}
		left++
		right--
	}
	return true
}

func minimumTime(time []int, totalTrips int) int64 {
	n := len(time)
	if n == 1 {
		return int64(time[0]) * int64(totalTrips)
	}
	sort.Ints(time)
	left, right := int64(time[0]), int64(time[0])*int64(totalTrips)
	for left < right {
		mid := left + (right-left)/2
		if findMaxTrips(mid, time) >= totalTrips {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return right
}

func findMaxTrips(timeLimit int64, tripTimes []int) int {
	output := 0
	for i := 0; i < len(tripTimes); i++ {
		if int64(tripTimes[i]) > timeLimit {
			break
		}
		output += int(timeLimit) / tripTimes[i]
	}
	return output
}

func sumEvenGrandparent(root *TreeNode) int {
	return sEG(1, 1, root)
}
func sEG(grandParent, parent int, root *TreeNode) int {
	if root == nil {
		return 0
	}
	if grandParent%2 == 0 {
		return root.Val + sEG(parent, root.Val, root.Left) + sEG(parent, root.Val, root.Right)
	} else {
		return sEG(parent, root.Val, root.Left) + sEG(parent, root.Val, root.Right)
	}
}

func canBeEqual(target []int, arr []int) bool {
	if len(target) != len(arr) {
		return false
	}
	n := len(target)
	sort.Ints(target)
	sort.Ints(arr)
	for i := 0; i < n; i++ {
		if target[i] != arr[i] {
			return false
		}
	}
	return true
}

func countPartitions(nums []int, k int) int {
	n := len(nums)
	const mod = 1_000_000_007
	dp := make([]int, n+1)
	dp[n] = 1
	for i := n - 1; i >= 0; i-- {
		currentMax := nums[i]
		currentMin := nums[i]
		for j := i; j < n; j++ {
			if nums[j] > currentMax {
				currentMax = nums[j]
			}
			if nums[j] < currentMin {
				currentMin = nums[j]
			}
			if currentMax-currentMin > k {
				break
			}

			dp[i] = (dp[i] + dp[j+1]) % mod
		}
	}
	return dp[0]
}

func matrixSum(nums [][]int) int {
	for _, num := range nums {
		sort.Ints(num)
	}
	output := 0
	for i := 0; i < len(nums[0]); i++ {
		currentMax := -1
		for _, num := range nums {
			if num[i] > currentMax {
				currentMax = num[i]
			}
		}
		output += currentMax
	}
	return output
}

func maxNumOfMarkedIndices(nums []int) int {
	sort.Ints(nums)
	n := len(nums)
	left := 0
	right := n / 2
	output := 0
	for left < n/2 && right < n {
		if nums[left]*2 <= nums[right] {
			output += 2
			left++
		}
		right++
	}
	return output
}

func findCommonResponse(responses [][]string) string {
	hashMap := make(map[string]int)
	for _, stringArray := range responses {
		stringSet := make(map[string]struct{})
		for _, s := range stringArray {
			stringSet[s] = struct{}{}
		}
		for key, _ := range stringSet {
			hashMap[key]++
		}
	}
	output := ""
	currentMax := -1
	for key, value := range hashMap {
		if value > currentMax {
			output = key
			currentMax = value
		} else if value == currentMax && key < output {
			output = key
		}
	}
	return output
}

func removeDuplicateLetters(s string) string {
	const alphabetSize = 26
	lastIndex := make([]int, alphabetSize)
	for i, char := range s {
		lastIndex[char-'a'] = i
	}
	var stack []int
	seenSlice := make([]bool, alphabetSize)
	for i, char := range s {
		charIndex := int(char - 'a')
		if seenSlice[charIndex] {
			continue
		}
		for len(stack) > 0 {
			top := stack[len(stack)-1]
			if top <= charIndex || lastIndex[top] <= i {
				break
			}
			seenSlice[top] = false
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, charIndex)
		seenSlice[charIndex] = true
	}
	output := make([]byte, len(stack))
	for i, charIndex := range stack {
		output[i] = byte('a' + charIndex)
	}
	return string(output)
}

func shortestDistanceAfterQueries(n int, queries [][]int) []int {
	output := []int{}
	graph := make([][]int, n)
	distances := make([]int, n)
	for i := 0; i < n; i++ {
		distances[i] = i
		graph[i] = append(graph[i], i+1)
	}
	for _, query := range queries {
		updateDistanceGraph(query, distances, graph)
		output = append(output, distances[n-1])
	}
	return output
}

func updateDistanceGraph(query, distances []int, graph [][]int) {
	from := query[0]
	to := query[1]
	if from+1 == to {
		return
	}
	graph[from] = append(graph[from], to)
	queue := []int{0}
	for len(queue) > 0 {
		currentNode := queue[0]
		queue = queue[1:]
		if currentNode == len(distances)-1 {
			continue
		}
		for _, nextNode := range graph[currentNode] {
			if distances[currentNode]+1 > distances[nextNode] {
				continue
			}
			distances[nextNode] = distances[currentNode] + 1
			queue = append(queue, nextNode)
		}
	}
	return
}


func assignEdgeWeights(edges [][]int) int {
	const mod = 1_000_000_007
	n := len(edges) + 1
	graph := make([][]int, n)
	for _, edge := range edges {
		u, v := edge[0]-1, edge[1]-1
		graph[u] = append(graph[u], v)
		graph[v] = append(graph[v], u)
	}
	depthSlice := make([]int, n)
	seenSlice := make([]bool, n)
	queue := []int{0}
	seenSlice[0] = true
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		for _, neighbor := range graph[node] {
			if !seenSlice[neighbor] {
				seenSlice[neighbor] = true
				depthSlice[neighbor] = depthSlice[node] + 1
				queue = append(queue, neighbor)
			}
		}
	}
	maxDepth := -1
	for _, depth := range depthSlice {
		if depth > maxDepth {
			maxDepth = depth
		}
	}
	output := 1
	for i := 0; i < maxDepth-1; i++ {
		output = (output * 2) % mod
	}
	return output
}

func maxProduct(words []string) int {
	output := 0
	n := len(words)
	bitMasks := make([]int, len(words))
	for i, word := range words {
		for _, char := range word {
			charValue := int(char - 'a')
			bitMasks[i] = bitMasks[i] | (1 << charValue)
		}
	}
	for i := 0; i < n; i++ {
		for j := i+1; j < n; j++ {
			if bitMasks[i] & bitMasks[j] == 0 && len(words[i]) * len(words[j]) > output {
				output = len(words[i]) * len(words[j])
			}
		}
	}
	return output
}


func divideArray(nums []int, k int) [][]int {
    sort.Ints(nums)
	output := [][]int{}
	for i := 0; i < len(nums); i += 3 {
		currentSlice := []int{}
		for j := 0; j < 3; j++ {
			currentSlice = append(currentSlice, nums[i + j])
		}
		if (currentSlice[2] - currentSlice[0]) > k {
			return [][]int{}
		}
		output = append(output, currentSlice)
	}
	return output
}

unc orangesRotting(grid [][]int) int {
    queue := [][]int{}
    freshCount := 0
    yMax, xMax := len(grid), len(grid[0])
    for y, xArray := range grid {
        for x, value := range xArray {
            switch value {
            case 1:
                freshCount++
            case 2:
                queue = append(queue, []int{y, x})
            }
        }
    }

    minutes := 0
    directions := [][]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}

    for len(queue) > 0 && freshCount > 0 {
        minutes++
        levelSize := len(queue)
        for i := 0; i < levelSize; i++ {
            rotten := queue[0]
            queue = queue[1:]

            for _, dir := range directions {
                y, x := rotten[0]+dir[0], rotten[1]+dir[1]

                if y >= 0 && y < yMax && x >= 0 && x < xMax && grid[y][x] == 1 {
                    grid[y][x] = 2
                    freshCount--
                    queue = append(queue, []int{y, x})
                }
            }
        }
    }

    if freshCount == 0 {
        return minutes
    }
    return -1
}
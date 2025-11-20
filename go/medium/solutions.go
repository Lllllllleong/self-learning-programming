package medium

import (
	"sort"
	"strings"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

/*
============================================================

============================================================
Time Complexity: O()
Space Complexity: O()
*/

/*
============================================================
1472. Design Browser History
============================================================
Time Complexity: O(?)
Space Complexity: O(?)
*/
type Node struct {
	url  string
	prev *Node
	next *Node
}

type BrowserHistory struct {
	current *Node
}

func Constructor(homepage string) BrowserHistory {
	return BrowserHistory{
		current: &Node{
			url: homepage,
		},
	}
}

func (this *BrowserHistory) Visit(url string) {
	newNode := &Node{
		url:  url,
		prev: this.current,
		next: nil,
	}
	this.current.next = newNode
	this.current = newNode
}

func (this *BrowserHistory) Back(steps int) string {
	for this.current.prev != nil && steps > 0 {
		this.current = this.current.prev
		steps--
	}
	return this.current.url
}

func (this *BrowserHistory) Forward(steps int) string {
	for this.current.next != nil && steps > 0 {
		this.current = this.current.next
		steps--
	}
	return this.current.url
}

/*
============================================================
648. Replace Words
============================================================
Time Complexity: O(?)
Space Complexity: O(?)
*/
type Trie struct {
	eow      bool
	children map[rune]*Trie
}

func NewTrie() *Trie {
	return &Trie{
		eow:      false,
		children: make(map[rune]*Trie),
	}
}

func replaceWords(dictionary []string, sentence string) string {
	trieRoot := NewTrie()
	currentNode := trieRoot
	// Fill in the Trie
	for _, rootString := range dictionary {
		runes := []rune(rootString)
		currentNode = trieRoot
		for _, rune := range runes {
			nextNode, ok := currentNode.children[rune]
			if !ok {
				nextNode = NewTrie()
				currentNode.children[rune] = nextNode
			}
			currentNode = nextNode
		}
		currentNode.eow = true
	}
	words := strings.Fields(sentence)
	for i, word := range words {
		currentNode = trieRoot
		runes := []rune(word)
		for j, rune := range runes {
			nextNode, ok := currentNode.children[rune]
			if !ok {
				break
			}
			currentNode = nextNode
			if currentNode.eow {
				words[i] = string(runes[:j+1])
				break
			}
		}
	}
	return strings.Join(words, " ")
}

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

/*
============================================================
Minimum Time to Complete Trips
============================================================
Time Complexity: O(?)
Space Complexity: O(?)
*/
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

/*
============================================================
Sum of Nodes with Even-Valued Grandparent
============================================================
Time Complexity: O(?)
Space Complexity: O(?)
*/
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

/*
============================================================
Count Subarrays with Score Less Than K
============================================================
Time Complexity: O(?)
Space Complexity: O(?)
*/
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

/*
============================================================
Sum in a Matrix
============================================================
Time Complexity: O(?)
Space Complexity: O(?)
*/
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

/*
============================================================
Find the Maximum Number of Marked Indices
============================================================
Time Complexity: O(?)
Space Complexity: O(?)
*/
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

/*
============================================================
Most Common Word
============================================================
Time Complexity: O(?)
Space Complexity: O(?)
*/
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

/*
============================================================
Remove Duplicate Letters
============================================================
Time Complexity: O(?)
Space Complexity: O(?)
*/
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

/*
============================================================
Shortest Distance After Road Addition Queries
============================================================
Time Complexity: O(?)
Space Complexity: O(?)
*/
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

/*
============================================================
Maximum Product of Word Lengths
============================================================
Time Complexity: O(?)
Space Complexity: O(?)
*/
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
		for j := i + 1; j < n; j++ {
			if bitMasks[i]&bitMasks[j] == 0 && len(words[i])*len(words[j]) > output {
				output = len(words[i]) * len(words[j])
			}
		}
	}
	return output
}

/*
============================================================
Rotting Oranges
============================================================
Time Complexity: O(?)
Space Complexity: O(?)
*/
func orangesRotting(grid [][]int) int {
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

/*
============================================================
Remove Adjacent Almost-Equal Characters
============================================================
Time Complexity: O(?)
Space Complexity: O(?)
*/
func resultingString(s string) string {
	stack := []byte{}
	for _, char := range s {
		charIndex := byte(char)
		if len(stack) > 0 {
			top := stack[len(stack)-1]
			difference := int(charIndex) - int(top)
			if difference == 1 || difference == -1 || difference == 25 || difference == -25 {
				stack = stack[:len(stack)-1]
			} else {
				stack = append(stack, charIndex)
			}
		} else {
			stack = append(stack, charIndex)
		}
	}
	return string(stack)
}

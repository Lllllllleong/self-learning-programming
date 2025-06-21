package main

import (
    "fmt"
    "math"
    "sort"
)

type TreeNode struct {
     Val int
     Left *TreeNode
     Right *TreeNode
}

func main() {
    fmt.Println("!!")



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
    for i := 0; i < len(s) - 1; i++ {
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
    if (condition1 && condition2) {return true}
    if (condition1 && !condition2) || (!condition1 && condition2) {return false}
    if (root1.Val != root2.Val) {return false}
    return (isSymmetric2(root1.Right, root2.Left)) && (isSymmetric2(root1.Left, root2.Right))
}

func validPalindrome(s string) bool {
    left, right := 0, len(s)-1
    if right == 0 {return true}
    flag := false
    for left <= right {
        if s[left] != s[right] {
            if flag {return false}
            flag = true
        }
        left++
        right--
    }
    return true
}

func minimumTime(time []int, totalTrips int) int64 {
    n := len(time)
    if n == 1 {return int64(time[0]) * int64(totalTrips)}
    sort.Ints(time)
    left, right := int64(time[0]), int64(time[0])*int64(totalTrips);
    for left < right {
        mid := left + (right - left)/2
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
        if int64(tripTimes[i]) > timeLimit { break }
        output += int(timeLimit) / tripTimes[i]
    }
    return output
}

func sumEvenGrandparent(root *TreeNode) int {
    return sEG(1, 1, root)
}
func sEG(grandParent, parent int, root *TreeNode) int {
    if root == nil {return 0}
    if (grandParent % 2 == 0) {
        return root.Val + sEG(parent, root.Val, root.Left) + sEG(parent, root.Val, root.Right)
    } else {
        return sEG(parent, root.Val, root.Left) + sEG(parent, root.Val, root.Right)
    }
}

func canBeEqual(target []int, arr []int) bool {
    if len(target) != len(arr) {return false}
    n := len(target)
    sort.Ints(target)
    sort.Ints(arr)
    for i := 0; i < n; i++ {
        if target[i] != arr[i] {return false}
    }
    return true
}
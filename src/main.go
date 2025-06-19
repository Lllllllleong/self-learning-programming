package main

import (
    "fmt"
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
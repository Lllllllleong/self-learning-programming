# Project Overview

This repository contains LeetCode solutions implemented in Go. The solutions are organized by difficulty into three packages: `easy`, `medium`, and `hard`.

# Building and Running

There are no explicit build or run commands in this project. The code is intended to be used in one of the following ways:
1.  Copied and pasted into the LeetCode online editor for a specific problem.
2.  Used with a testing framework that can execute the individual solution functions.

A possible way to run the code would be to create a `main.go` file in the root of the project and call the functions from there. For example:

```go
package main

import (
	"fmt"
	"github.com/yourusername/golangPractice/easy"
)

func main() {
	// Example of calling a function from the easy package
	nums := []int{1, 2, 4, 5, 7}
	missing := easy.findMissingElements(nums)
	fmt.Println(missing)
}
```

Then, you can run the `main.go` file using the following command:

```bash
go run main.go
```

# Development Conventions

When adding new solutions, please adhere to the following conventions:

1.  **File Organization:** Place the solution in the appropriate package (`easy`, `medium`, or `hard`) based on the problem's difficulty.
2.  **Solution Format:** Each solution should be in its own function and include the following information in a comment block above the function:
    *   Problem number and name
    *   A link to the LeetCode problem
    *   A brief description of the problem
    *   The time and space complexity of your solution

    Example:
    ```go
    /*
    ============================================================
    1: Two Sum
    ============================================================
    Time Complexity: O(n)
    Space Complexity: O(n)
    */
    func twoSum(nums []int, target int) []int {
        // Your solution here
    }
    ```

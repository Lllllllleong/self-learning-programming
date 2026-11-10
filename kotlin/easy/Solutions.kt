package easy

class TreeNode(var `val`: Int) {
    var left: TreeNode? = null
    var right: TreeNode? = null
}

class ListNode(var `val`: Int) {
    var next: ListNode? = null
}

/*
============================================================

============================================================
Time Complexity: O()
Space Complexity: O()
*/

/*
============================================================
496. Next Greater Element I
============================================================
Time Complexity: O(n + m), where n = nums1.size, m = nums2.size
Space Complexity: O(m)
*/
fun nextGreaterElement(nums1: IntArray, nums2: IntArray): IntArray {
    val dq = ArrayDeque<Int>()
    val map = HashMap<Int, Int>()
    for (i in nums2.size - 1 downTo 0) {
        val num = nums2[i]
        while (dq.firstOrNull() != null && dq.first() < num) {
            dq.removeFirst()
        }
        if (dq.firstOrNull() == null) {
            map[num] = -1
        } else {
            map[num] = dq.first()
        }
        dq.addFirst(num)
    }

    val out = IntArray(nums1.size)
    for ((index, value) in nums1.withIndex()) {
        out[index] = map.getValue(value)
    }

    return out
}


/*
============================================================
283. Move Zeroes
============================================================
Time Complexity: O(n)
Space Complexity: O(1)
*/
fun moveZeroes(nums: IntArray) {
    var j = 0
    for (num in nums) {
        if (num != 0) nums[j++] = num
    }
    nums.fill(0, fromIndex = j)
}

/*
============================================================
1281. Subtract the Product and Sum of Digits of an Integer
============================================================
Time Complexity: O(log n)
Space Complexity: O(1)
*/
fun subtractProductAndSum(n: Int): Int {
    var num = n
    var product = 1
    var sum = 0
    while (num > 0) {
        val digit = num % 10
        product *= digit
        sum += digit
        num /= 10
    }

    return product - sum
}


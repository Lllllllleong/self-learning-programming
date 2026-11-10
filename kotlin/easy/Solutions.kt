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



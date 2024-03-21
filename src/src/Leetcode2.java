import java.util.*;

public class Leetcode2 {
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }


    public int[] maxSlidingWindow(int[] nums, int k) {
        if (k == 1) return nums;
        int n = nums.length;
        int[] output = new int[n - k + 1];
        int outputIndex = 0;
        Deque<Integer> windowQueue = new ArrayDeque<>();
        windowQueue.addLast(Integer.MIN_VALUE);
        for (int i = 0; i < nums.length; i++) {
            int current = nums[i];
            int windowLB = i - k + 1;
            while (!windowQueue.isEmpty() && windowQueue.peek() < windowLB) windowQueue.pop();

            //New max
            if (windowQueue.isEmpty() || current >= nums[windowQueue.peek()]) {
                windowQueue.clear();
                windowQueue.add(i);
            } else {
                while (nums[windowQueue.peekLast()] < current) {
                    windowQueue.pollLast();
                }
                windowQueue.addLast(i);
            }
            //Begin adding to output
            if (i >= (k - 1)) {
                output[outputIndex] = nums[windowQueue.peek()];
                outputIndex++;
            }
        }
        return output;
    }

    public boolean isLeaf(TreeNode root) {
        if (root == null) return false;
        if (root.left == null && root.right == null) return true;
        return false;
    }

    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode previous = null;
        ListNode current = head;
        ListNode next = current.next;
        while (next != null) {
            current.next = previous;
            previous = current;
            current = next;
            next = next.next;
        }
        current.next = previous;
        return current;
    }


    public int findTheCity(int n, int[][] edges, int distanceThreshold) {
        int[][] dpArray = new int[n][n];
        for (int[] edge : edges) {
            int first = edge[0];
            int second = edge[1];
            int third = edge[2];
            dpArray[first][second] = third;
            dpArray[second][first] = third;
        }
        for (int i = 0; i < n; i++) {
            Deque<Integer> currentQueue = new ArrayDeque<>();
            for (int j = 0; j < n; j++) {
                if (dpArray[i][j] != 0 && dpArray[i][j] <= distanceThreshold) currentQueue.add(j);
            }
            while (!currentQueue.isEmpty()) {
                Integer nextCity = currentQueue.pop();
                Integer currentDistance = dpArray[i][nextCity];
                for (int j = 0; j < n; j++) {
                    if (j == i || dpArray[i][j] != 0) continue;
                    else {
                        Integer currentNextDistance = dpArray[nextCity][j] + currentDistance;
                        if (currentNextDistance <= distanceThreshold) {
                            currentQueue.add(j);
                            dpArray[i][j] = currentNextDistance;
                        }
                    }
                }
            }
        }
        int currentMinimum = Integer.MAX_VALUE;
        int currentCity = -1;
        for (int i = 0; i < n; i++) {
            int[] dp = dpArray[i];
            System.out.println(Arrays.toString(dp));
            int cityCount = 0;
            for (int j : dp) {
                if (j != 0 && j <= distanceThreshold) cityCount++;
            }
            if (cityCount <= currentMinimum) {
                currentCity = i;
                currentMinimum = cityCount;
            }
        }
        return currentCity;
    }


    public int maxScoreSightseeingPair(int[] values) {
        int n = values.length;
        int maxPair = 0;
        //As i increases, the value of the ith entry decreases by 1 everytime;
        //Therefore, the equivalent value of choosing the first of the pair to be
        //i will be value[i] + i
        int maxEquivalentValue = values[0] + 0;
        for (int i = 1; i < n; i++) {
            maxPair = Math.max(maxPair, maxEquivalentValue + values[i] - i);
            //Stores the current maximum equivalent value(for the first of the pair) to be
            //Used in future comparisons
            maxEquivalentValue = Math.max(maxEquivalentValue, values[i] + i);
        }
        return maxPair;
    }


}






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

    public int maxUncrossedLines(int[] nums1, int[] nums2) {
        int xBound = nums1.length;
        int yBound = nums2.length;
        int[][] dpArray = new int[yBound + 1][xBound + 1];
        for (int y = yBound-1; y >=0; y--) {
            for (int x = xBound-1; x >= 0; x--) {
                if (nums1[x] == nums2[y]) {
                    dpArray[y][x] = 1 + dpArray[y+1][x+1];
                } else {
                    dpArray[y][x] = Math.max(dpArray[y+1][x], dpArray[y][x+1]);
                }
            }
        }
        return dpArray[0][0];
    }


    public int maxResult(int[] nums, int k) {
        int output = 0;
        Deque<Integer> dQueue = new ArrayDeque<>();
        for (int i = nums.length-1; i >= 0 ; i--) {
            if (nums[i] < 0) {
                dQueue.addFirst(nums[i]);
            } else {
                if (!dQueue.isEmpty() && dQueue.size() >= k) {
                    output += queueSum(dQueue, k);
                }
                dQueue.clear();
                output += nums[i];
            }
        }
        return output;
    }
    public int queueSum(Deque<Integer> dQ, int k) {
        Deque<Integer> sumQueue = new ArrayDeque<>();
        while (sumQueue.size() != k) {
            sumQueue.addFirst(dQ.pollLast());
        }
        int output = 0;
        while (!dQ.isEmpty()) {
            int current = dQ.pollLast();
            current = queueMax(sumQueue) + current;
            sumQueue.pollLast();
            sumQueue.addFirst(current);
        }
        return queueMax(sumQueue);
    }

    public int queueMax(Deque<Integer> q) {
        if (q.isEmpty()) return -1;
        int output = Integer.MIN_VALUE;
        while (!q.isEmpty()) {
            output = Math.max(output, q.pop());
        }
        return output;
    }



    public int minSideJumps(int[] obstacles) {
        int[][] dpArray = new int[3][obstacles.length];
        int counter = 0;
        for (int i : obstacles) {
            if (i != 0) {
                i = i - 1;
                dpArray[i][counter] = 1000;
            }
            counter++;
        }
        Deque<Integer> orderQueue = new ArrayDeque<>(Arrays.asList(0,1,2));
        for (int i = obstacles.length-2; i >= 0 ; i--) {
            for (int j = 0; j < 3; j++) {
                if (dpArray[j][i] == 1000) continue;
                dpArray[j][i] = dpArray[j][i+1];
            }
            for (int j = 0; j < 3; j++) {
                dpArray[j][i] = Math.min(dpArray[j][i], dpArray[0][i] + 1);
                dpArray[j][i] = Math.min(dpArray[j][i], dpArray[1][i] + 1);
                dpArray[j][i] = Math.min(dpArray[j][i], dpArray[2][i] + 1);
            }
            System.out.println("currentRoad is " + i);
            System.out.println(dpArray[0][i]);
            System.out.println(dpArray[1][i]);
            System.out.println(dpArray[2][i]);
        }
        return dpArray[1][0];
    }


    public int rob(int[] nums) {
        //nums is a circular array
        //You can never rob the first and last at the same time
        int n = nums.length;
        if (n == 1) return nums[0];
        if (n == 2) return Math.max(nums[0], nums[1]);
        int first = rob2(Arrays.copyOfRange(nums, 1, n));
        int second = rob2(Arrays.copyOfRange(nums, 0, n-1));
        return Math.max(first,second);
    }
    public int rob2(int[] nums) {
        //Let
        //dp[0][n] be the max value if you choose to rob the nth house
        //dp[1][n] is the max value if you don't rob the nth house
        int n = nums.length;
        int[][] dp = new int[2][n+1];
        for (int i = n-1; i >= 0 ; i--) {
            int currentRob = nums[i];
            dp[0][i] = currentRob + dp[1][i+1];
            dp[1][i] = Math.max(dp[0][i+1], dp[1][i+1]);
        }
        return Math.max(dp[0][0], dp[1][0]);
    }


    public int[] reverseArray(int[] in) {
        int n = in.length;
        int[] reverse = new int[n];
        n = n-1;
        for (int i : in) {
            reverse[n] = i;
            n--;
        }
        return reverse;
    }
    public int maxSizeSlices(int[] slices) {
        int n = slices.length;
        int[] sliceReverse = reverseArray(slices);
        int[][] dpArray = new int[n+2][n+2];
        int j = 0;
        for (int x = n-1; x >= 0 ; x--) {
            for (int y = j; y >= 0; y--) {
                int one = slices[x] + dpArray[y+1][x+2];
                int two = slices[y] + dpArray[y+2][x+1];
                int three = dpArray[y+1][x+1];
                dpArray[y][x] = Math.max(one, Math.max(two, three));
            }
            j++;
        }
        for (int[] a : dpArray) {
            System.out.println(Arrays.toString(a));
        }
        return Math.max(dpArray[1][0], Math.max(dpArray[0][1], dpArray[0][0]));
    }


    public int longestZigZag(TreeNode root) {
        if (root == null) return 0;
        int a = zagLeft(root.right);
        int b = zagRight(root.left);
        int c = longestZigZag(root.left);
        int d = longestZigZag(root.right);
        int output = Math.max(a, Math.max(b, Math.max(c,d)));
        return output;
    }
    public int zagLeft(TreeNode root) {
        if (root == null) return 0;
        return (1 + zagRight(root.left));
    }
    public int zagRight(TreeNode root) {
        if (root == null) return 0;
        return (1 + zagLeft(root.right));
    }


    public int jump(int[] nums) {
        int n = nums.length;
        if (n == 1) return 0;
        nums[n-1] = 0;
        for (int i = n-2; i >= 0; i--) {
            int currentJump = nums[i];
            if (currentJump == 0) {
                nums[i] = 10000;
                continue;
            }
            int max = i + currentJump;
            if (max >= n - 1) {
                nums[i] = 1;
            } else {
                int currentMinimum = Integer.MAX_VALUE;
                for (int j = max; j > i; j--) {
                    currentMinimum = Math.min(currentMinimum, 1 + nums[j]);
                }
                nums[i] = currentMinimum;
            }
        }
        return nums[0];
    }


    public int trap(int[] height) {
        int n = height.length;
        if (n <= 2) return 0;
        int leftIndex = 0;
        int rightIndex = n-1;
        int leftMaxHeight = height[leftIndex];
        int rightMaxHeight = height[rightIndex];
        int output = 0;
        while (leftMaxHeight == 0) {
            leftIndex++;
            leftMaxHeight = height[leftIndex];
        }
        while (leftIndex != rightIndex) {
            if (leftMaxHeight <= rightMaxHeight) {
                leftIndex++;
                int currentHeight = height[leftIndex];
                if (currentHeight < leftMaxHeight) {
                    output += leftMaxHeight - currentHeight;
                } else {
                    leftMaxHeight = currentHeight;
                }
            } else {
                rightIndex--;
                int currentHeight = height[rightIndex];
                if (currentHeight < rightMaxHeight) {
                    output += rightMaxHeight - currentHeight;
                } else {
                    rightMaxHeight = currentHeight;
                }
            }
        }
        return output;
    }

    public int maxProfit(int[] prices) {
        int n = prices.length;
        if (n == 1) return 0;
        if (n == 2) {
            int p = prices[1] - prices[0];
            return (p > 0) ? p : 0;
        } else {
            int[][] dpArray = new int[n+1][n+1];
            for (int i = n-1; i >= 0; i--) {
                int currentPeak = prices[i];
                for (int j = i; j >= 0; j--) {
                    int currentPrice = prices[j];
                    dpArray[i][j] = Math.max(dpArray[i][j+1], currentPeak - currentPrice);
                    currentPeak = Math.max(currentPeak, currentPrice);
                }
            }
            for (int[] a : dpArray) {
                System.out.println(Arrays.toString(a));
            }
            int output = dpArray[n-1][0];
            for (int i = 1; i < n; i++) {
                int c = dpArray[n-1][i] + dpArray[i-1][0];
                output = Math.max(output, c);
            }
            return output;
        }
    }



    public boolean isSubsequence(String s, String t) {
        int y = s.length();
        int x = t.length();
        if (y == 0 || y == 0 && x == 0) return true;
        if (x == 0) return false;
        boolean[][] dpArray = new boolean[y+1][x+1];
        //Corner solution initialise
        Arrays.fill(dpArray[y], true);
        for (int i = y-1; i >= 0; i--) {
            for (int j = x-1; j >= 0; j--) {
                if (s.charAt(i) == t.charAt(j)) {
                    dpArray[i][j] = dpArray[i+1][j+1];
                } else {
                    dpArray[i][j] = dpArray[i][j+1];
                }
            }
        }
        return dpArray[0][0];
    }


    public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        if (n == 1) return 1;
        int[] dpArray = new int[n];
        //Initialise
        Arrays.fill(dpArray, 1);
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dpArray[i] = Math.max(dpArray[i], dpArray[j] + 1);
                }
            }
        }
        int output = 0;
        for (int i : dpArray) {
            output = Math.max(output, i);
        }
        return output;
    }



    public int findLengthOfLCIS(int[] nums) {
        int n = nums.length;
        if (n == 1) return 1;
        int output = 1;
        int counter = 1;
        for (int i = 1; i < n; i++) {
            if (nums[i] > nums[i-1]) {
                counter++;
                output = Math.max(output, counter);
            } else {
                output = Math.max(output, counter);
                counter = 1;
            }
        }
        return output;
    }

    public int lenLongestFibSubseq(int[] arr) {
        int output = 0;
        int currentCount = 2;
        for (int i = 0; i < arr.length-2; i++) {
            int middle = arr[i+1];
            int target = arr[i] + middle;
            for (int j = i+2; j < arr.length; j++) {
                int current = arr[j];
                if (current == target) {
                    currentCount++;
                    target = target + middle;
                    middle = current;
                }
            }
            output = Math.max(output, currentCount);
            currentCount = 2;
        }
        return output;
    }

    public int longestMountain(int[] arr) {
        int n = arr.length;
        int output = 0;
        if (n <= 2) return 0;
        for (int i = 1; i < n; i++) {
            int prev = arr[i-1];
            int curr = arr[i];
            if (prev < curr) {
                arr[i-1] = 1;
            } else if (prev == curr) {
                arr[i-1] = 0;
            } else {
                arr[i-1] = -1;
            }
        }
        int previousPrefix = -1;
        int currentMax = 0;
        for (int i = 0; i < n-1; i++) {

            int currentPrefix = arr[i];
            System.out.println("i is " + i);
            System.out.println("currentPrevfix is " + currentPrefix);
            System.out.println("count is " + currentMax);

            if (previousPrefix == -1 && currentPrefix == 1) {
                output = Math.max(output, currentMax);
                currentMax = 2;
            } else if (previousPrefix == -1 && currentPrefix == -1 && currentMax != 0) {
                currentMax++;
                if (i == n-2) output = Math.max(output, currentMax);
            } else if (previousPrefix == 1 && currentPrefix == -1) {
                currentMax++;
                output = Math.max(output, currentMax);
            } else if (previousPrefix == 1 && currentPrefix == 1) {
                currentMax++;
            } else if (previousPrefix == 0 && currentPrefix == 1) {
                currentMax = 2;
            } else {
                currentMax = 0;
            }
            previousPrefix = currentPrefix;
        }
        return output;
    }


    public int maxIncreaseKeepingSkyline(int[][] grid) {
        int xLength = grid[0].length;
        int yLength = grid.length;
        int[] xMax = new int[xLength];
        int[] yMax = new int[yLength];
        for (int y = 0; y < yLength; y++) {
            int[] currentGrid = grid[y];
            yMax[y] = arrMaxValue(currentGrid);
            for (int x = 0; x < xLength; x++) {
                xMax[x] = Math.max(xMax[x], currentGrid[x]);
            }
        }
        int output = 0;
        for (int y = 0; y < yLength; y++) {
            for (int x = 0; x < xLength; x++) {
                int currentHeight = grid[y][x];
                int currentMax = Math.min(xMax[x], yMax[y]);
                output += (currentMax - currentHeight);
            }
        }
        return output;
    }
    public int arrMaxValue(int[] in) {
        int output = Integer.MIN_VALUE;
        for (int i : in) {
            output = Math.max(output, i);
        }
        return output;
    }

}






import java.security.spec.RSAOtherPrimeInfo;
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
        for (int y = yBound - 1; y >= 0; y--) {
            for (int x = xBound - 1; x >= 0; x--) {
                if (nums1[x] == nums2[y]) {
                    dpArray[y][x] = 1 + dpArray[y + 1][x + 1];
                } else {
                    dpArray[y][x] = Math.max(dpArray[y + 1][x], dpArray[y][x + 1]);
                }
            }
        }
        return dpArray[0][0];
    }


    public int maxResult(int[] nums, int k) {
        int output = 0;
        Deque<Integer> dQueue = new ArrayDeque<>();
        for (int i = nums.length - 1; i >= 0; i--) {
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
        Deque<Integer> orderQueue = new ArrayDeque<>(Arrays.asList(0, 1, 2));
        for (int i = obstacles.length - 2; i >= 0; i--) {
            for (int j = 0; j < 3; j++) {
                if (dpArray[j][i] == 1000) continue;
                dpArray[j][i] = dpArray[j][i + 1];
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
        int second = rob2(Arrays.copyOfRange(nums, 0, n - 1));
        return Math.max(first, second);
    }

    public int rob2(int[] nums) {
        //Let
        //dp[0][n] be the max value if you choose to rob the nth house
        //dp[1][n] is the max value if you don't rob the nth house
        int n = nums.length;
        int[][] dp = new int[2][n + 1];
        for (int i = n - 1; i >= 0; i--) {
            int currentRob = nums[i];
            dp[0][i] = currentRob + dp[1][i + 1];
            dp[1][i] = Math.max(dp[0][i + 1], dp[1][i + 1]);
        }
        return Math.max(dp[0][0], dp[1][0]);
    }


    public int maxSizeSlices(int[] slices) {
        int n = slices.length;
//        int[] sliceReverse = reverseArray(slices);
        int[][] dpArray = new int[n + 2][n + 2];
        int j = 0;
        for (int x = n - 1; x >= 0; x--) {
            for (int y = j; y >= 0; y--) {
                int one = slices[x] + dpArray[y + 1][x + 2];
                int two = slices[y] + dpArray[y + 2][x + 1];
                int three = dpArray[y + 1][x + 1];
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
        int output = Math.max(a, Math.max(b, Math.max(c, d)));
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
        nums[n - 1] = 0;
        for (int i = n - 2; i >= 0; i--) {
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
        int rightIndex = n - 1;
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
            int[][] dpArray = new int[n + 1][n + 1];
            for (int i = n - 1; i >= 0; i--) {
                int currentPeak = prices[i];
                for (int j = i; j >= 0; j--) {
                    int currentPrice = prices[j];
                    dpArray[i][j] = Math.max(dpArray[i][j + 1], currentPeak - currentPrice);
                    currentPeak = Math.max(currentPeak, currentPrice);
                }
            }
            for (int[] a : dpArray) {
                System.out.println(Arrays.toString(a));
            }
            int output = dpArray[n - 1][0];
            for (int i = 1; i < n; i++) {
                int c = dpArray[n - 1][i] + dpArray[i - 1][0];
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
        boolean[][] dpArray = new boolean[y + 1][x + 1];
        //Corner solution initialise
        Arrays.fill(dpArray[y], true);
        for (int i = y - 1; i >= 0; i--) {
            for (int j = x - 1; j >= 0; j--) {
                if (s.charAt(i) == t.charAt(j)) {
                    dpArray[i][j] = dpArray[i + 1][j + 1];
                } else {
                    dpArray[i][j] = dpArray[i][j + 1];
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
            if (nums[i] > nums[i - 1]) {
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
        for (int i = 0; i < arr.length - 2; i++) {
            int middle = arr[i + 1];
            int target = arr[i] + middle;
            for (int j = i + 2; j < arr.length; j++) {
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
            int prev = arr[i - 1];
            int curr = arr[i];
            if (prev < curr) {
                arr[i - 1] = 1;
            } else if (prev == curr) {
                arr[i - 1] = 0;
            } else {
                arr[i - 1] = -1;
            }
        }
        int previousPrefix = -1;
        int currentMax = 0;
        for (int i = 0; i < n - 1; i++) {

            int currentPrefix = arr[i];
            System.out.println("i is " + i);
            System.out.println("currentPrevfix is " + currentPrefix);
            System.out.println("count is " + currentMax);

            if (previousPrefix == -1 && currentPrefix == 1) {
                output = Math.max(output, currentMax);
                currentMax = 2;
            } else if (previousPrefix == -1 && currentPrefix == -1 && currentMax != 0) {
                currentMax++;
                if (i == n - 2) output = Math.max(output, currentMax);
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

    public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> output = new ArrayList<>();
        int subLength = words[0].length();
        int sLength = subLength * words.length;
        int n = s.length();
        //Initialise map
        HashMap<String, Integer> frequencyHM = new HashMap<>();
        for (String w : words) {
            frequencyHM.merge(w, 1, Integer::sum);
        }
        for (int i = 0; i <= n - sLength; i++) {
            String current = s.substring(i, i + sLength);
            HashMap<String, Integer> currentHM = new HashMap<>(frequencyHM);
            int k = 0;
            while (k < sLength) {
                String currentSubstring = current.substring(k, k + subLength);
                if (!currentHM.containsKey(currentSubstring)) break;
                currentHM.compute(currentSubstring, (a, b) -> (b == null) ? null : b - 1);
                k += subLength;
            }
            boolean flag = true;
            for (Integer I : currentHM.values()) {
                if (I != 0) flag = false;
            }
            if (flag) output.add(i);
        }
        return output;
    }

    public int maxProduct(int[] nums) {
        int n = nums.length;
        if (n == 1) return nums[0];
        int output = 0;
        for (int i = n - 1; i >= 0; i--) {
            int current = nums[i];
            output = Math.max(output, current);
            for (int j = i + 1; j < n; j++) {
                nums[j] = nums[j] * current;
                output = Math.max(output, nums[j]);
            }
        }
        return output;
    }

    public int coinChange(int[] coins, int amount) {
        if (amount == 0) return amount;
        int[] dpArray = new int[amount + 1];
        Arrays.fill(dpArray, Integer.MAX_VALUE - 1);
        dpArray[0] = 0;
        Arrays.sort(coins);
        for (int i = 1; i <= amount; i++) {
            for (int c : coins) {
                int change = i - c;
                if (change < 0) break;
                else {
                    dpArray[i] = Math.min(dpArray[i], dpArray[change] + 1);
                }
            }
        }
        if (dpArray[amount] == Integer.MAX_VALUE - 1) return -1;
        else return dpArray[amount];
    }


    public int minNumberOperations(int[] target) {
        int n = target.length;
        if (n == 1) return target[0];
        int output = 0;
        int previous = 0;
        for (int i : target) {
            if (previous < i) {
                output += (i - previous);
            }
            previous = i;
        }
        return output;
    }


    public long maxPoints(int[][] points) {
        int yMax = points.length;
        int xMax = points[0].length;
        long[] dpArray = new long[xMax];
        for (int x = 0; x < xMax; x++) {
            dpArray[x] = (long) points[yMax - 1][x];
        }
        for (int y = yMax - 2; y >= 0; y--) {
            for (int x = 0; x < xMax; x++) {
                long currentX = 0;
                for (int z = 0; z < xMax; z++) {
                    currentX = Math.max(currentX, points[y][z] - Math.abs(x - z));
                }
                dpArray[x] += currentX;
            }
        }
        long output = 0;
        for (long l : dpArray) output = Math.max(output, l);
        return output;

    }


    public int countSpecialSubsequences(int[] nums) {
        int n = nums.length;
        double[][] dpArray = new double[3][n + 1];
        for (int i = n - 1; i >= 0; i--) {
            int current = nums[i];
            dpArray[0][i] = dpArray[0][i + 1];
            dpArray[1][i] = dpArray[1][i + 1];
            dpArray[2][i] = dpArray[2][i + 1];
            if (current == 0) {
                dpArray[0][i] = ((dpArray[0][i] * 2) + dpArray[1][i]);
            } else if (current == 1) {
                dpArray[1][i] = ((dpArray[1][i] * 2) + dpArray[2][i]);
            } else {
                dpArray[2][i] = ((dpArray[2][i] * 2) + 1);
            }
        }
        for (double[] a : dpArray) {
            System.out.println(Arrays.toString(a));
        }
        return (int) (dpArray[0][0] % (Math.pow(10, 9) + 7));
    }


    public int peopleAwareOfSecret(int n, int delay, int forget) {
        int output = 1;
        int originalDelay = delay;
        int originalForget = forget;
        while (n != 0) {
            if (delay <= 0) {
                output += peopleAwareOfSecret(n, originalDelay, originalForget);
            }
            if (forget == 0) {
                output--;
            }
            n--;
            delay--;
            forget--;
        }
        return output;
    }

    public boolean canReach(String s, int minJump, int maxJump) {
        int n = s.length();
        if (s.charAt(n - 1) == 0) return false;
        if (n == 2) {
            return (s.charAt(0) == '0' && minJump >= 1);
        }
        int[] dpArray = new int[n];
        Arrays.fill(dpArray, 1);
        dpArray[n - 1] = 0;
        for (int i = n - 2; i >= 0; i--) {
            if (s.charAt(i) == '1') {
                dpArray[i] = 1;
            } else {
                for (int j = minJump; j <= maxJump; j++) {
                    int jumpto = Math.min(n - 1, i + j);
                    if (dpArray[jumpto] == 0) {
                        dpArray[i] = 0;
                        break;
                    }
                }
            }
        }
        return dpArray[0] == 0;
    }


    public long maxTaxiEarnings(int n, int[][] rides) {
        Arrays.sort(rides, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                return (b[0] - a[0]);
            }
        });

        TreeMap<Integer, Long> tm = new TreeMap<>();
        tm.put(n, (long) 0);
        long currentMax = 0;
        for (int[] ride : rides) {
            long currentEarning = ride[1] - ride[0] + ride[2];
            currentEarning += tm.get(tm.ceilingKey(ride[1]));
            currentMax = Math.max(currentMax, currentEarning);
            tm.put(ride[0], currentMax);
        }
        return currentMax;
    }


    public int minSumOfLengths(int[] arr, int target) {
        List<Integer> l = new ArrayList<>();
        int n = arr.length;
        if (n == 1) return 0;
        for (int i = n - 1; i >= 0; i--) {
            if (arr[i] == target) l.add(1);
            if (arr[i] > target) continue;
            for (int j = i + 1; j < n; j++) {
                arr[j] += arr[i];
                if (arr[j] == target) l.add(j - i + 1);
            }
        }
        Collections.sort(l);
        if (l.size() < 2) return -1;
        else {
            return (l.get(0) + l.get(1));
        }
    }


    public int longestSquareStreak(int[] nums) {
        Set<Integer> s = new HashSet<>();
        for (int i : nums) {
            s.add(i);
        }
        int output = -1;
        List<Integer> l = new ArrayList<>(s);
        Collections.sort(l, Collections.reverseOrder());
        HashMap<Integer, Integer> hm = new HashMap<>();
        for (Integer I : l) {
            Integer II = I * I;
            if (hm.containsKey(II)) {
                Integer newValue = hm.get(II) + 1;
                output = Math.max(newValue, output);
                hm.put(I, newValue);
            } else {
                hm.put(I, 1);
            }
        }
        return output;

    }

    public long maxScore(int[] nums, int x) {
        int n = nums.length;
        long oddMax = 0, evenMax = 0;
        for (int i = n - 1; i >= 0; i--) {
            long l = nums[i];
            if (l % 2 == 0) {
                evenMax += l;
                evenMax = Math.max(evenMax, oddMax + l - x);
            } else {
                oddMax += l;
                oddMax = Math.max(oddMax, evenMax + l - x);
            }
        }
        if (nums[0] % 2 == 0) return evenMax;
        else return oddMax;
    }


    public int minimizeArrayValue(int[] nums) {
        int out = 0;
        long sum = 0;
        for (int i = 0; i < nums.length; i++) {
            int current = nums[i];
            sum = sum + current;
            int j = (int) sum / (i + 1);
            j = (sum % (i + 1) == 0) ? j : j + 1;
            out = Math.max(out, j);
        }
        return out;
    }

    public int rootCount(int[][] edges, int[][] guesses, int k) {
        int maxNode = edges.length;
        int output = 0;
        for (int i = 0; i < maxNode + 1; i++) {
            int kk = k;
            int[][] dpArray = constructEdgeTree(edges, i);
            for (int[] guess : guesses) {
                if (dpArray[guess[0]][guess[1]] == 1) {
                    kk--;
                    if (kk == 0) break;
                }
            }
            if (kk <= 0) {
                System.out.println("correct for node " + i);
                output++;
            }

        }
        return output;
    }

    public int[][] constructEdgeTree(int[][] edges, int root) {
        Deque<Integer> d = new ArrayDeque<>();
        d.add(root);
        int n = edges.length;
        int[][] out = new int[n][n];
        Set<Integer> s = new HashSet<>();
        s.add(root);
        while (!d.isEmpty()) {
            int target = d.pollFirst();
            for (int[] edge : edges) {
                if (edge[0] == target && !s.contains(edge[1])) {
                    d.addLast(edge[1]);
                    s.add(edge[1]);
                    out[target][edge[1]] = 1;
                } else if (edge[1] == target && !s.contains(edge[0])) {
                    d.addLast(edge[0]);
                    s.add(edge[0]);
                    out[target][edge[0]] = 1;
                }
            }
        }
        return out;
    }


    public int minSessions(int[] tasks, int sessionTime) {
        int n = tasks.length;
        if (n == 1) return n;
        Arrays.sort(tasks);
        boolean b = allZero(tasks);
        int out = 0;
        while (!b) {
            System.out.println(Arrays.toString(tasks));
            out++;
            int currentTime = sessionTime;
            for (int i = n - 1; i >= 0; i--) {
                if (tasks[i] > currentTime) continue;
                else {
                    if (tasks[i] != 0) {
                        currentTime -= tasks[i];
                        tasks[i] = 0;
                    }
                }
            }
            b = allZero(tasks);
        }
        return out;
    }

    public boolean allZero(int[] in) {
        for (int i : in) {
            if (i != 0) return false;
        }
        return true;
    }

    public int closestCost(int[] baseCosts, int[] toppingCosts, int target) {
        Set<Integer> combinationSums = new HashSet<>();
        generateSums(toppingCosts, 0, 0, combinationSums);
        int output = -1;
        int delta = Integer.MAX_VALUE;
        for (int i : baseCosts) {
            for (int j : combinationSums) {
                int k = i + j;
                int difference = Math.abs(target - k);
                if (difference < delta) {
                    delta = difference;
                    output = k;
                } else if (difference == delta) {
                    output = Math.min(output, k);
                }
            }
        }
        return output;
    }


    public static void generateSums(int[] nums, int index, int currentSum, Set<Integer> result) {
        // Base case: If we've considered all elements, add the current sum to the result set
        if (index == nums.length) {
            result.add(currentSum);
            return;
        }
        // Case 1: Don't include the current element
        generateSums(nums, index + 1, currentSum, result);
        // Case 2: Include the current element once
        generateSums(nums, index + 1, currentSum + nums[index], result);
        // Case 3: Include the current element twice
        generateSums(nums, index + 1, currentSum + 2 * nums[index], result);
    }


    public int maxSum(int[] nums1, int[] nums2) {
        int aIndex = 0, bIndex = 0;
        int aMax = nums1.length, bMax = nums2.length;
        long output = 0;
        long sumA = 0, sumB = 0;
        while (aIndex < aMax && bIndex < bMax) {
            long a = nums1[aIndex];
            long b = nums2[bIndex];
            if (a == b) {
                sumA += a;
                sumB += b;
                output += Math.max(sumA, sumB);
                sumA = 0;
                sumB = 0;
                aIndex++;
                bIndex++;
            } else if (a < b) {
                sumA += a;
                aIndex++;
            } else if (a > b) {
                sumB += b;
                bIndex++;
            }
        }
        //Clear the queues
        for (int i = aIndex; i < aMax; i++) {
            sumA += nums1[i];
        }
        for (int i = bIndex; i < bMax; i++) {
            sumB += nums2[i];
        }
        output += Math.max(sumA, sumB);
        return (int) (output % (Math.pow(10, 9) + 7));
    }


    public int maxAbsoluteSum(int[] nums) {
        int n = nums.length;
        int output = 0;
        if (n == 1) return (Math.abs(nums[0]));
        for (int i = n - 1; i >= 0; i--) {
            int a = nums[i];
            output = Math.max(output, Math.abs(a));
            for (int j = i + 1; j < n; j++) {
                nums[j] += a;
                output = Math.max(output, Math.abs(nums[j]));
            }
        }
        return output;
    }

    public String kthSmallestPath(int[] destination, int k) {
        int y = destination[0];
        int x = destination[1];
        String s = "";
        //Base cases
        if (x == 0 && y == 0) return s;
        if (x > 0 && y == 0) {
            s = "H";
            return s.repeat(x);
        }
        if (x == 0 && y > 0) {
            s = "V";
            return s.repeat(y);
        }
        if (x == 1 && y == 1) {
            if (k == 1) return "HV";
            else return "VH";
        }
        //Non-Base case
        //Find the number of ways to dest, if we move right
        long right = nCr((y + x - 1), y);
        //If k is less than no. ways, we move right. Otherwise, move down
        if (k <= right) {
            int[] newDest = {y, x - 1};
            s = "H" + kthSmallestPath(newDest, k);
        } else {
            int kk = (int) (k - right);
            int[] newDest = {y - 1, x};
            s = "V" + kthSmallestPath(newDest, kk);
        }
        return s;
    }

    public long nCr(int n, int r) {
        if (r > n) {
            return 0;
        }
        if (r == 0 || r == n) {
            return 1;
        }
        r = Math.min(r, n - r); // Use symmetry property nCr = nC(n-r)
        long result = 1;
        // Calculate the result of nCr using a loop to avoid integer overflow
        for (int i = 1; i <= r; i++) {
            result *= n - r + i;
            result /= i;
        }
        return result;
    }

    public ListNode mergeNodes(ListNode head) {
        if (head == null) return null;
        if (head.next == null) return null;
        ListNode current = head;
        ListNode next = current.next;
        while (next.val != 0) {
            current = current.next;
            next = current.next;
        }
        current.next = null;
        ListNode newHead = nodeSumMerge(head);
        ListNode newTail = mergeNodes(next);
        newHead.next = newTail;
        return newHead;
    }

    public ListNode nodeSumMerge(ListNode head) {
        if (head == null) return null;
        int out = 0;
        ListNode current = head;
        while (current != null) {
            out += current.val;
            current = current.next;
        }
        ListNode output = new ListNode(out);
        return output;
    }


    public int maximumTop(int[] nums, int k) {
        int n = nums.length;
        if (n == 0) return -1;
        if (k == 0) return nums[0];
        if (n == 1) {
            if (k % 2 == 0) return nums[0];
            else return -1;
        }
        if (k == 1) {
            if (n <= 1) return -1;
            else return nums[1];
        }
        if (k >= n + 1) return maxInArray(nums);
        if (k == n) {
            nums[n - 1] = nums[0];
            return maxInArray(nums);
        } else {
            int max = nums[0];
            int index = 0;
            while (k != 1) {
                max = Math.max(max, nums[index]);
                index++;
                k--;
            }
            return Math.max(max, nums[index + 1]);
        }
    }

    public int maxInArray(int[] in) {
        int out = in[0];
        for (int i : in) {
            out = Math.max(out, i);
        }
        return out;
    }


    public List<String> findAllRecipes(String[] recipes,
                                       List<List<String>> ingredients,
                                       String[] supplies) {
        List<String> out = new ArrayList<>();
        List<String> recList = new ArrayList<>(Arrays.asList(recipes));
        List<String> supList = new ArrayList<>(Arrays.asList(supplies));
        boolean suppliesAdded = true;
        while (suppliesAdded) {
            suppliesAdded = false;
            int n = recList.size();
            for (int i = 0; i < n; i++) {
                String recipe = recList.get(i);
                if (!supList.contains(recipe)) {
                    List<String> ing = ingredients.get(i);
                    boolean enoughSupplies = true;
                    for (String s : ing) {
                        if (!supList.contains(s)) {
                            enoughSupplies = false;
                            break;
                        }
                    }
                    if (enoughSupplies) {
                        supList.add(recipe);
                        out.add(recipe);
                        suppliesAdded = true;
                    }
                }
            }
        }
        return out;
    }



    public void reverseString(char[] s) {
        char c;
        int n = s.length;
        int j = n-1;
        for (int i = 0; i < n/2; i++, j--) {
            c = s[i];
            s[i] = s[j];
            s[j] = c;
        }

    }
    public ListNode doubleIt(ListNode head) {
        if (head == null) return null;
        if (head.next == null) {
            head.val = head.val * 2;
            return head;
        } else {
            ListNode current = head;
            long sum = 0;
            while (current != null) {
                sum = sum * 10;
                sum += current.val;
                current = current.next;
            }
            sum = sum * 2;
            ListNode previous = new ListNode();
            while (sum != 0) {
                ListNode tail = new ListNode((int) (sum % 10));
                tail.next = previous;
                previous = tail;
                sum = sum / 10;
            }
            return previous;
        }
    }


    public int addRungs(int[] rungs, int dist) {
        int n = rungs.length;
        if (n == 1) {
            if (rungs[0] > dist) return 1;
            else return 0;
        }
        int out = 0;
        int prev = 0;
        for (int i = 0; i < n; i++) {
            int a = rungs[i];
            out += Math.max(0,(prev - a - 1) / dist);
            prev = a;
        }
        return out;
    }


    public boolean mergeTriplets(int[][] triplets, int[] target) {
        int a = 0, b = 0, c = 0;
        int x = target[0];
        int y = target[1];
        int z = target[2];
        for (int[] t : triplets) {
            if (t[0] <= x && t[1] <= y && t[2] <= z) {
                a = Math.max(a, t[0]);
                b = Math.max(b, t[1]);
                c = Math.max(c, t[2]);
            }
        }
        return (a == x && b == y && c == z);
    }



    public int[] platesBetweenCandles(String s, int[][] queries) {
        int n = s.length();
        int m = queries.length;
        TreeMap<Integer, Integer> tm = new TreeMap<>();
        int counter = 0;
        boolean firstCandle = true;
        for (int i = 0; i < n; i++) {
            char c = s.charAt(i);
            if (c == '*') counter++;
            else if (c == '|') {
                if (firstCandle) {
                    firstCandle = false;
                    counter = 0;
                }
                tm.put(i, counter);
            }
        }
        int[] out = new int[m];
        for (int i = 0; i < m; i++) {
            int[] query = queries[i];
            if (tm.ceilingKey(query[0]) != null && tm.floorKey(query[1]) != null) {
                int left = tm.get(tm.ceilingKey(query[0]));
                int right = tm.get(tm.floorKey(query[1]));
                out[i] = Math.max(right - left, 0);
            } else {
                out[i] = 0;
            }
        }
        return out;
    }


    public boolean winnerOfGame(String colors) {
        if (colors.length() < 3) return false;
        int aMove = 0, bMove = 0;
        int consecCount = 1;
        char prior = 'Z';
        int n = colors.length();
        for (int i = 0; i < n; i++) {
            char c = colors.charAt(i);
            if (c == prior) {
                consecCount++;
                if (consecCount >= 3) {
                    if (c == 'A') aMove++;
                    else bMove++;
                }
            } else {
                consecCount = 1;
            }
            prior = c;
        }
        return aMove > bMove;
    }



    public int[] maximumBeauty(int[][] items, int[] queries) {
        int n = items.length;
        Arrays.sort(items, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                if (a[0] != b[0]) return a[0] - b[0];
                else return b[1] - a[1];
            }
        });
        TreeMap<Integer, Integer> tm = new TreeMap<>();
        tm.put(0,0);
        int maxBeauty = 0;
        for (int[] item : items) {
            int price = item[0];
            int beauty = item[1];
            if (beauty <= maxBeauty) continue;
            else {
                maxBeauty = beauty;
                tm.put(price, maxBeauty);
            }
        }
        int[] out = new int[queries.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = tm.get(tm.floorKey(queries[i]));
        }
        return out;
    }



    public int networkBecomesIdle(int[][] edges, int[] patience) {
        HashMap<Integer, Integer> distanceMap = new HashMap<>();
        distanceMap.put(0, 0);
        Deque<int[]> dQ = new ArrayDeque<>();
        for (int[] edge : edges) {
            dQ.addLast(edge);
        }
        while (!dQ.isEmpty()) {
            int[] edge = dQ.pollFirst();
            int a = edge[0];
            int b = edge[1];
            if (distanceMap.containsKey(a) && !distanceMap.containsKey(b)) {
                distanceMap.put(b, distanceMap.get(a) + 1);
            } else if (!distanceMap.containsKey(a) && distanceMap.containsKey(b)) {
                distanceMap.put(a, distanceMap.get(b) + 1);
            } else if (distanceMap.containsKey(a) && distanceMap.containsKey(b)) {
                distanceMap.put(a, Math.min(distanceMap.get(b) + 1, distanceMap.get(a)));
                distanceMap.put(b, Math.min(distanceMap.get(a) + 1, distanceMap.get(b)));
            } else {
                dQ.addLast(edge);
            }
        }
        for (Map.Entry<Integer,Integer> e : distanceMap.entrySet()) {
            System.out.println(e.toString());
        }
        int out = -1;
        for (int server : distanceMap.keySet()) {
            int d = distanceMap.get(server);
            int p = patience[server];
            System.out.println(d + " d is and p is " + p);
            if (p >= 2*d) {
                out = Math.max(out, 2*d);
            } else {
                if (p == 1) {
                    out = Math.max(out, 4*d - 1);
                } else if ((2*d) % p == 0) {
                    out = Math.max(out, 4*d - p);
                } else {
                    out = Math.max(out, 4*d - ((2*d) % p));
                }
            }
            System.out.println("out is " + out);
        }
        return out + 1;
    }


    public int wateringPlants(int[] plants, int capacity) {
        int output = 0;
        int currentCapacity = capacity;
        for (int i = 0; i < plants.length; i++) {
            System.out.println("cap start " + currentCapacity);
            output++;
            System.out.println("output " + output);
            int position = i + 1;
            int req = plants[0];
            if (currentCapacity < req) {
                output += (position * 2);
                currentCapacity = capacity;
            }
            currentCapacity -= req;
            System.out.println("cap end " + currentCapacity);
        }
        return output;
    }


    public int minimumDeletions(int[] nums) {
        int n = nums.length;
        if (n <= 2) return n;
        int a = Integer.MAX_VALUE;
        int aa = 0;
        int b = Integer.MIN_VALUE;
        int bb = 0;
        for (int i = 0; i < n; i++) {
            int current = nums[i];
            if (current > b) {
                b = current;
                bb = i;
            }
            if (current < a) {
                a = current;
                aa = i;
            }
        }
        int one = n - Math.min(aa,bb);
        int two = Math.min(aa,bb) + (n - Math.max(aa,bb));
        int three = n - Math.max(aa,bb) + 1;
        return Math.min(one, Math.min(two, three));
    }


    public List<Integer> findLonely(int[] nums) {
        int n = nums.length;
        List<Integer> out = new ArrayList<>();
        if (n == 1) {
            out.add(nums[0]);
            return out;
        } else if (n == 2) {
            if (Math.abs(nums[1] - nums[0]) > 1) {
                out.add(nums[0]);
                out.add(nums[1]);
                return out;
            } else return out;
        } else {
            Arrays.sort(nums);
            if (Math.abs(nums[1] - nums[0]) > 1) {
                out.add(nums[0]);
            }
            if (Math.abs(nums[n-1] - nums[n-2]) > 1) {
                out.add(nums[n-1]);
            }
            for (int i = 1; i < n-1; i++) {
                if (nums[i] - nums[i-1] > 1 && nums[i+1] - nums[i] > 1) {
                    out.add(nums[i]);
                }
            }
        }
        return out;
    }

    public int minimumRefill(int[] plants, int capacityA, int capacityB) {
        int n = plants.length;
        if (n == 1) {
            if (capacityA >= plants[0] || capacityB >= plants[0]) return 0;
            else return 1;
        } else {
            Deque<Integer> dq = new ArrayDeque<>();
            for (int i : plants) {
                dq.addLast(i);
            }
            int out = 0;
            int a = capacityA;
            int b = capacityB;
            while (!dq.isEmpty()) {
                if (dq.size() == 1) {
                    int i = dq.pop();
                    if (a < i && b < i) out++;
                }
                int plantA = dq.pollFirst();
                int plantB = dq.pollLast();
                if (a < plantA) {
                    out++;
                    a = capacityA;
                }
                if (b < plantB) {
                    out++;
                    b = capacityB;
                }
                a -= plantA;
                b -= plantB;
            }
            return out;
        }
    }

    public long[] getDistances(int[] arr) {
        HashMap<Integer, Set<Integer>> hmSet = new HashMap<>();
        int n = arr.length;
        for (int i = 0; i < n; i++) {
            int current = arr[i];
            if (!hmSet.containsKey(current)) {
                hmSet.put(current, new HashSet<>());
            }
            hmSet.get(current).add(i);
        }

        long[] out = new long[n];
        for (int i = 0; i < n; i++) {
            Set<Integer> indexes = hmSet.get(arr[i]);
            long sum = 0;
            for (Integer I : indexes) {
                if (I == i) continue;
                sum += Math.abs(I - i);
            }
            out[i] = sum;
        }
        return out;
    }



    public ListNode sortList(ListNode head) {
        if (head == null) return null;
        if (head.next == null) return head;
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        ListNode ln = head;
        while(ln != null) {
            pq.add(ln.val);
            ln = ln.next;
        }
        ListNode out = new ListNode();
        ListNode prev = out;

        while (!pq.isEmpty()) {
            ListNode current = new ListNode(pq.poll());
            prev.next = current;
            prev = prev.next;
        }
        return out.next;
    }

    public void wiggleSort(int[] nums) {
        int n = nums.length;
        if (n == 1) return;
        else {
            Arrays.sort(nums);
            Deque<Integer> dQ = new ArrayDeque<>(Arrays.stream(nums).boxed().toList());
            boolean pollFirst = true;
            int i = 0;
            while (!dQ.isEmpty()) {
                int j;
                if (pollFirst) {
                    j = dQ.pollFirst();
                    pollFirst = false;
                } else {
                    j = dQ.pollLast();
                    pollFirst = true;
                }
                nums[i] = j;
                i++;
            }
        }
    }

    public int[][] sortTheStudents(int[][] score, int k) {
        Arrays.sort(score, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                return (b[k] - a[k]);
            }
        });
        return score;
    }

}


//          return (int) (output % (Math.pow(10,9) + 7));
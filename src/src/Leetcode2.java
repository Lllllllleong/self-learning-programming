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
        int j = n - 1;
        for (int i = 0; i < n / 2; i++, j--) {
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
            out += Math.max(0, (prev - a - 1) / dist);
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
        tm.put(0, 0);
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
        for (Map.Entry<Integer, Integer> e : distanceMap.entrySet()) {
            System.out.println(e.toString());
        }
        int out = -1;
        for (int server : distanceMap.keySet()) {
            int d = distanceMap.get(server);
            int p = patience[server];
            System.out.println(d + " d is and p is " + p);
            if (p >= 2 * d) {
                out = Math.max(out, 2 * d);
            } else {
                if (p == 1) {
                    out = Math.max(out, 4 * d - 1);
                } else if ((2 * d) % p == 0) {
                    out = Math.max(out, 4 * d - p);
                } else {
                    out = Math.max(out, 4 * d - ((2 * d) % p));
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
        int one = n - Math.min(aa, bb);
        int two = Math.min(aa, bb) + (n - Math.max(aa, bb));
        int three = n - Math.max(aa, bb) + 1;
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
            if (Math.abs(nums[n - 1] - nums[n - 2]) > 1) {
                out.add(nums[n - 1]);
            }
            for (int i = 1; i < n - 1; i++) {
                if (nums[i] - nums[i - 1] > 1 && nums[i + 1] - nums[i] > 1) {
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
        while (ln != null) {
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


    public int maxSubArray(int[] nums) {
        int n = nums.length;
        if (n == 1) return nums[0];
        int maxSum = Integer.MIN_VALUE;
        int currentSum = 0;
        for (int i : nums) {
            currentSum += i;
            maxSum = Math.max(maxSum, currentSum);
            currentSum = (currentSum < 0) ? 0 : currentSum;
        }
        return maxSum;
    }


    public String smallestSubsequence(String s) {
        int n = s.length();
        if (n == 1) return s;
        char[] c = s.toCharArray();
        Arrays.sort(c);
        String out = String.valueOf(s.charAt(0));
        for (char cc : c) {
            if (cc != out.charAt(out.length() - 1)) out += cc;
        }
        return out;
    }


    public int[][] rangeAddQueries(int n, int[][] queries) {
        int[][] matrix = new int[n][n];
        for (int[] query : queries) {
            int y = query[0];
            int x = query[1];
            int yy = query[2];
            int xx = query[3];
            for (int i = y; i <= yy; i++) {
                for (int j = x; j <= xx; j++) {
                    matrix[i][j]++;
                }
            }
        }
        return matrix;
    }


    public int maxSumTwoNoOverlap(int[] nums, int firstLen, int secondLen) {
        int out = 0;
        int n = nums.length;
        if (n == 1) return n;
        int[] dpAFirst = maxSumDPFromLeftToRight(nums, firstLen);
        int[] dpBFirst = maxSumDPFromRightToLeft(nums, secondLen);
        for (int i = firstLen - 1; i < n - secondLen; i++) {
            out = Math.max(out, dpAFirst[i] + dpBFirst[i + 1]);
        }
        int[] dpASecond = maxSumDPFromLeftToRight(nums, secondLen);
        int[] dpBSecond = maxSumDPFromRightToLeft(nums, firstLen);
        for (int i = secondLen - 1; i < n - firstLen; i++) {
            out = Math.max(out, dpASecond[i] + dpBSecond[i + 1]);
        }
        return out;
    }

    public int[] maxSumDPFromRightToLeft(int[] nums, int length) {
        int n = nums.length;

        int[] out = new int[n];
        int max = 0;
        int prev = 0;
        for (int i = n - 1; i >= 0; i--) {
            int current = prev + nums[i];
            int minus = (i + length >= n) ? 0 : nums[i + length];
            current -= minus;
            max = Math.max(max, current);
            out[i] = max;
            prev = current;
        }
        return out;
    }

    public int[] maxSumDPFromLeftToRight(int[] nums, int length) {
        int n = nums.length;

        int[] out = new int[n];
        int max = 0;
        int prev = 0;
        for (int i = 0; i < n; i++) {
            int current = prev + nums[i];
            int minus = (i - length < 0) ? 0 : nums[i - length];
            current -= minus;
            max = Math.max(max, current);
            out[i] = max;
            prev = current;
        }
        return out;
    }


    public boolean increasingTriplet(int[] nums) {
        int n = nums.length;
        if (n < 3) return false;
        int a = Integer.MAX_VALUE;
        int b = Integer.MAX_VALUE;
        for (int i : nums) {
            if (i > b) return true;
            if (i < b) {
                if (i < a) {
                    a = i;
                } else if (i > a) {
                    b = i;
                }
            }
        }
        return false;
    }


    public int longestSubarray(int[] nums, int limit) {
        int n = nums.length;
        if (n == 1) {
            return 1;
        }
        int out = 1;
        PriorityQueue<Integer> inc = new PriorityQueue<>();
        PriorityQueue<Integer> dec = new PriorityQueue<>(Collections.reverseOrder());
        Deque<Integer> dq = new ArrayDeque<>();
        for (int i : nums) {
            dq.addLast(i);
            inc.add(i);
            dec.add(i);
            int diff = dec.peek() - inc.peek();
            while (diff > limit) {
                int removed = dq.removeFirst();
                inc.remove(removed);
                dec.remove(removed);
                diff = dec.peek() - inc.peek();
            }
            out = Math.max(out, dq.size());
        }
        return out;
    }


    public boolean possibleToStamp(int[][] grid, int stampHeight, int stampWidth) {
        int yMax = grid.length;
        int xMax = grid[0].length;
        int maxWidth = maxMatrixWidth(grid);
        int maxHeight = maxMatrixWidth(transposeMatrix(grid));
        System.out.println("max width " + maxWidth);
        System.out.println("max height " + maxHeight);
        if (maxHeight == 0 && maxWidth == 0) return true;
        return (stampHeight <= maxHeight && stampWidth <= maxWidth);
    }

    public int maxMatrixWidth(int[][] matrix) {
        int xMax = matrix[0].length;
        int maxWidth = xMax;
        boolean flag = false;
        for (int[] y : matrix) {
            int currentWidth = 0;
            for (int x = 0; x < xMax; x++) {
                if (y[x] == 0) {
                    currentWidth++;
                    flag = true;
                } else if (y[x] == 1 && currentWidth != 0) {
                    maxWidth = Math.min(maxWidth, currentWidth);
                    currentWidth = 0;
                }
            }
            if (currentWidth != 0) maxWidth = Math.min(maxWidth, currentWidth);
        }
        if (!flag) return 0;
        else return maxWidth;
    }

    public int[][] transposeMatrix(int[][] matrix) {
        int rows = matrix.length; // Number of rows of original matrix
        int cols = matrix[0].length; // Number of columns of original matrix
        // Initialize the transposed matrix with dimensions cols x rows
        int[][] transposed = new int[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Swap the elements
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }


    public int maxProfitAssignment(int[] difficulty, int[] profit, int[] worker) {
        TreeMap<Integer, Integer> jobProfitMap = new TreeMap<>();
        for (int i = 0; i < difficulty.length; i++) {
            jobProfitMap.put(difficulty[i], Math.max(jobProfitMap.getOrDefault(difficulty[i], 0), profit[i]));
        }

        int max = 0;
        for (Map.Entry<Integer, Integer> entry : jobProfitMap.entrySet()) {
            max = Math.max(max, entry.getValue());
            entry.setValue(max); // Ensure each difficulty level maps to the max profit so far
        }

        int totalProfit = 0;
        for (int capability : worker) {
            Integer floorKey = jobProfitMap.floorKey(capability);
            if (floorKey != null) {
                totalProfit += jobProfitMap.get(floorKey);
            }
        }
        return totalProfit;
    }


    public int maximumProduct(int[] nums, int k) {
        int n = nums.length;
        if (n == 1) return (nums[0] + k);
        PriorityQueue<Long> pq = new PriorityQueue<>();
        long mod = 1000000007;
        for (int num : nums) {
            pq.add((long) num);
        }
        while (k > 0) {
            Long I = pq.poll();
            I++;
            pq.add(I);
            k = k - 1;
        }
        long out = 1;
        while (!pq.isEmpty()) {
            out = (out * pq.poll()) % mod;
        }
        return (int) out;
    }


    public int tupleSameProduct(int[] nums) {
        int n = nums.length;
        if (n < 4) {
            return 0;
        }
        HashMap<Integer, Integer> productMap = new HashMap<>();
        for (int i = 0; i < nums.length - 1; i++) {
            int a = nums[i];
            for (int j = i + 1; j < nums.length; j++) {
                int b = nums[j];
                int prod = a * b;
                productMap.merge(prod, 1, Integer::sum);
            }
        }
        int output = 0;
        for (Integer I : productMap.values()) {
            if (I == 1) continue;
            else {
                output += (I * (I - 1) * 4);
            }
        }
        return output;
    }


    public String shiftingLetters(String s, int[][] shifts) {
        int n = s.length();
        //Prefix array
        int[] prefix = new int[n + 1];
        for (int[] shift : shifts) {
            int a = shift[0];
            int b = shift[1];
            int c = shift[2];
            if (c == 1) {
                prefix[a]++;
                prefix[b + 1]--;
            } else {
                prefix[a]--;
                prefix[b + 1]++;
            }
        }
        //Prefix sum
        for (int i = 1; i < n + 1; i++) {
            prefix[i] = prefix[i] + prefix[i - 1];
        }
        char[] charArray = s.toCharArray();
        //Shift the characters
        for (int i = 0; i < n; i++) {
            int charIndex = charArray[i] - 'a';
            int shift = prefix[i] % 26;
            charIndex = (charIndex + 26 + shift) % 26;
            charArray[i] = (char) ('a' + charIndex);
        }
        String out = new String(charArray);
        return out;
    }


    public List<List<Long>> splitPainting(int[][] segments) {
        List<List<Long>> out = new ArrayList<>();
        HashMap<Long, Long> prefixHM = new HashMap<>();
        for (int[] segment : segments) {
            long a = segment[0];
            long b = segment[1];
            long c = segment[2];
            prefixHM.merge(a, c, (existing, current) -> existing + current);
            prefixHM.merge(b, -c, (existing, current) -> existing - current);
        }
        System.out.println(prefixHM);
        Set<Long> keySet = prefixHM.keySet();
        Long prev = Long.valueOf(0);
        for (Long key : keySet) {
            prev = prev + prefixHM.get(key);
            prefixHM.put(key, prev);
        }
        List<Long> keyList = new ArrayList<>(keySet);
        //Get the first entry ready
        Long prevIndex = keyList.get(0);
        Long prevColour = prefixHM.get(prevIndex);
        for (int i = 1; i < keyList.size(); i++) {
            Long nextKey = keyList.get(i);
            List<Long> currentL = new ArrayList<>();
            currentL.add(prevIndex);
            currentL.add(nextKey);
            currentL.add(prevColour);
            out.add(currentL);
            prevIndex = nextKey;
            prevColour = prefixHM.get(prevIndex);
        }
        return out;
    }


    HashMap<Integer, Long> treeHM;

    public long kthLargestLevelSum(TreeNode root, int k) {
        long l = -1;
        if (root == null) {
            return l;
        }
        treeHM = new HashMap<>();
        rootToMap(root, 1);
        List<Long> list = new ArrayList<>(treeHM.values());
        Collections.sort(list, Collections.reverseOrder());
        int s = list.size();
        if (k > s) return -1;
        else return (list.get(k - 1));
    }

    public void rootToMap(TreeNode root, int level) {
        if (root == null) {
            return;
        }
        Long l = Long.valueOf(root.val);
        treeHM.merge(level, l, (a, b) -> a + b);
        int nextLevel = level + 1;
        rootToMap(root.left, nextLevel);
        rootToMap(root.right, nextLevel);
    }


    public ListNode partition(ListNode head, int x) {
        if (head.next == null) return head;
        ListNode lowerHead = new ListNode();
        ListNode lowerCurrent = lowerHead;

        ListNode upperHead = new ListNode();
        ListNode upperCurrent = upperHead;

        ListNode current = head;
        while (current != null) {
            if (current.val >= x) {
                upperCurrent.next = current;
                upperCurrent = upperCurrent.next;
            } else {
                lowerCurrent.next = current;
                lowerCurrent = lowerCurrent.next;
            }
            current = current.next;
        }
        upperCurrent.next = null;
        upperHead = upperHead.next;
        lowerCurrent.next = upperHead;
        return lowerHead.next;
    }


    public int minimumLines(int[][] stockPrices) {
        int out = 0;
        int n = stockPrices.length;
        if (n == 1) {
            return out;
        }
        Arrays.sort(stockPrices, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                return (a[0] - b[0]);
            }
        });
        for (int[] sp : stockPrices) {
            System.out.println(Arrays.toString(sp));
        }
        float priorGradient = Integer.MAX_VALUE;
        for (int i = 0; i < stockPrices.length - 1; i++) {
            int[] stockPrice1 = stockPrices[i];
            int[] stockPrice2 = stockPrices[i + 1];
            float x = stockPrice1[0];
            float y = stockPrice1[1];
            float xx = stockPrice2[0];
            float yy = stockPrice2[1];
            float currentGradient = (yy - y) / (xx - x);
            System.out.println(Math.abs(priorGradient - currentGradient));
            if (Math.abs(priorGradient - currentGradient) > 0.0001) {
                out++;
                priorGradient = currentGradient;
            }
        }
        return out;
    }


    public int findLength(int[] nums1, int[] nums2) {
        int yMax = nums1.length;
        int xMax = nums2.length;
        int[][] dpArray = new int[yMax + 1][xMax + 1];
        for (int y = 0; y < yMax; y++) {
            dpArray[y][xMax - 1] = (nums1[y] == nums2[xMax - 1]) ? 1 : 0;
        }
        for (int x = 0; x < xMax; x++) {
            dpArray[yMax - 1][x] = (nums1[yMax - 1] == nums2[x]) ? 1 : 0;
        }
        for (int y = yMax - 2; y >= 0; y--) {
            int currentY = nums1[y];
            for (int x = xMax - 2; x >= 0; x--) {
                int currentX = nums2[x];
                if (currentY == currentX) {
                    dpArray[y][x] = dpArray[y + 1][x + 1] + 1;
                } else {
                    dpArray[y][x] = Math.max(dpArray[y + 1][x], dpArray[y][x + 1]);
                }
            }
        }
        for (int[] dp : dpArray) {
            System.out.println(Arrays.toString(dp));
        }
        return dpArray[0][0];
    }


    public int wiggleMaxLength(int[] nums) {
        int n = nums.length;
        int out = 1;
        if (n == 1) {
            return out;
        }
        int priorDifference = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            int a = nums[i];
            int b = nums[i + 1];
            int currentDifference = b - a;
            if (currentDifference > 0 && priorDifference <= 0 || currentDifference < 0 && priorDifference >= 0) {
                out++;
                priorDifference = currentDifference;
            }
        }
        return out;
    }


    public int maxProfit(int[] prices, int fee) {
        int n = prices.length;
        if (n == 1) {
            return 0;
        }
        int minimumPrice = Integer.MAX_VALUE;
        int profit = 0;
        for (int price : prices) {
            minimumPrice = Math.min(minimumPrice, price);
            if (price - minimumPrice > fee) {
                profit += (price - minimumPrice - fee);
                minimumPrice = price;
            }
        }
        return profit;
    }


    int[][] pathMatrix;

    public int minFallingPathSum(int[][] matrix) {
        int out = 0;
        pathMatrix = matrix;
        int yMax = pathMatrix.length;
        int xMax = pathMatrix[0].length;
        if (yMax != 1) {
            for (int y = yMax - 2; y >= 0; y--) {
                for (int x = 0; x < xMax; x++) {
                    pathMatrix[y][x] = pathMatrix[y][x] + getMinLower(y, x);
                }
            }
        }
        for (int i : pathMatrix[0]) {
            out = Math.min(out, i);
        }
        return out;
    }

    public int getMinLower(int y, int x) {
        int yMax = pathMatrix.length;
        int xMax = pathMatrix[0].length;
        int out = Integer.MAX_VALUE;
        y = y + 1;
        //Should not happen
        if (y == yMax) return -1;
        x = x - 1;
        if (x >= 0) {
            out = Math.min(out, pathMatrix[y][x]);
        }
        x = x + 1;
        out = Math.min(out, pathMatrix[y][x]);
        x = x + 1;
        if (x < xMax) {
            out = Math.min(out, pathMatrix[y][x]);
        }
        return out;
    }


    public int minimumTimeRequired(int[] jobs, int k) {
        Deque<Integer> jobQueue = new ArrayDeque<>(Arrays.stream(jobs).boxed().toList());
        int[] bins = new int[k];
        return binPack(jobQueue, bins);
    }

    public int binPack(Deque<Integer> jobs, int[] bins) {
        // If no jobs left, find and return the max fill level among all bins.
        if (jobs.isEmpty()) {
            int maxFill = 0;
            for (int fill : bins) {
                maxFill = Math.max(maxFill, fill);
            }
            return maxFill;
        }
        // Take the next job from the deque.
        int currentJob = jobs.pop();
        int minMaxFill = Integer.MAX_VALUE;
        for (int i = 0; i < bins.length; i++) {
            // Place the current job in bin[i] if it fits (this example doesn't have a capacity limit per bin).
            bins[i] += currentJob;
            // Recurse with the remaining jobs.
            int currentFill = binPack(jobs, bins);
            minMaxFill = Math.min(minMaxFill, currentFill);
            // Backtrack: remove the current job from bin[i].
            bins[i] -= currentJob;
        }
        // Return the job to the deque for the next iteration.
        jobs.addFirst(currentJob);
        return minMaxFill;
    }


    public int numRollsToTarget(int n, int k, int target) {
        int[][] dpArray = new int[n + 2][target + 2];
        dpArray[0][0] = 1;
        for (int y = 1; y < n + 1; y++) {
            int xLim = Math.min(y * k, target);
            for (int x = y; x <= xLim; x++) {
                //With y dices, how many ways to reach a target sum of x?
                long currentDP = 0;
                for (int i = 1; i <= Math.min(k, x); i++) {
                    currentDP += dpArray[y - 1][x - i];
                }
                dpArray[y][x] = (int) (currentDP % (Math.pow(10, 9) + 7));
            }
        }
        return dpArray[n][target];
    }


    public int[] sumOfDistancesInTree(int n, int[][] edges) {
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            int distanceCounter = 0;
            Set<Integer> s = new HashSet<>();
            s.add(i);
            int multiplier = 1;
            int countAdded = 1;
            while (countAdded != 0) {
                Set<Integer> ss = new HashSet<>();
                for (int[] edge : edges) {
                    Integer a = edge[0];
                    Integer b = edge[1];
                    if (s.contains(a) && !s.contains(b) || !s.contains(a) && s.contains(b)) {
                        if (ss.contains(a)) {
                            ss.add(b);
                        } else {
                            ss.add(a);
                        }
                    }
                }
                System.out.println("s and ss are");
                System.out.println(s);
                System.out.println(ss);
                countAdded = ss.size();
                distanceCounter += countAdded * multiplier;
                multiplier++;
                s.addAll(ss);
            }
            out[i] = distanceCounter;
        }
        return out;
    }


    public int combinationSum4(int[] nums, int target) {
        Arrays.sort(nums);
        int n = target;
        if (n == 0) return 0;
        if (nums.length == 1) {
            if (target % nums[0] == 0) return 1;
            else return 0;
        }
        long[] dpArray = new long[n + 1];
        dpArray[0] = 1;
        for (int i = 1; i <= target; i++) {
            for (int coin : nums) {
                int change = i - coin;
                if (change < 0) break;
                else {
                    dpArray[i] += dpArray[change];
                }
            }
        }
        return (int) dpArray[target];
    }

    public int integerReplacement(int n) {
        if (n == 1) return 0;
        int[] dpArray = new int[n + 1];
        dpArray[1] = 0;
        dpArray[2] = 1;
        dpArray[3] = 2;
        dpArray[4] = 2;
        for (int i = 5; i <= n; i++) {
            if (i % 2 == 0) {
                dpArray[i] = dpArray[i / 2] + 1;
            } else {
                dpArray[i + 1] = dpArray[(i + 1) / 2] + 1;
                dpArray[i] = Math.min(dpArray[i - 1], dpArray[i + 1]) + 1;
                i++;
            }
        }
        return dpArray[n];
    }

    public int minDistance(String word1, String word2) {
        int yMax = word1.length();
        int xMax = word2.length();
        int[][] dpArray = new int[yMax + 1][xMax + 1];
        //Initialise values
        for (int y = 0, x = yMax; y < yMax; y++, x--) {
            dpArray[y][xMax] = x;
        }
        for (int x = 0, y = xMax; x < xMax; x++, y--) {
            dpArray[yMax][x] = y;
        }
        //DP Fill
        for (int y = yMax - 1; y >= 0; y--) {
            for (int x = xMax - 1; x >= 0; x--) {
                if (word1.charAt(y) == word2.charAt(x)) {
                    dpArray[y][x] = dpArray[y + 1][x + 1];
                } else {
                    dpArray[y][x] = Math.min(dpArray[y + 1][x], Math.min(dpArray[y][x + 1], dpArray[y + 1][x + 1])) + 1;
                }
            }
        }
        return dpArray[0][0];
    }


    public boolean isInterleave(String s1, String s2, String s3) {
        if (s1.length() + s2.length() != s3.length()) return false;

        int yMax = s1.length();
        int xMax = s2.length();
        boolean[][] dp = new boolean[yMax + 1][xMax + 1];

        // Initialize the DP table
        dp[0][0] = true;
        for (int i = 1; i <= yMax; i++) {
            dp[i][0] = dp[i - 1][0] && s1.charAt(i - 1) == s3.charAt(i - 1);
        }
        for (int j = 1; j <= xMax; j++) {
            dp[0][j] = dp[0][j - 1] && s2.charAt(j - 1) == s3.charAt(j - 1);
        }

        // Fill in the DP table
        for (int i = 1; i <= yMax; i++) {
            for (int j = 1; j <= xMax; j++) {
                int k = i + j - 1; // Index in s3
                dp[i][j] = (dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(k)) ||
                        (dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(k));
            }
        }

        // Uncomment to print the DP table for debugging
        // for (boolean[] row : dp) System.out.println(Arrays.toString(row));

        return dp[yMax][xMax];
    }


    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        if (n == 2) {
            return Math.min(cost[0], cost[1]);
        } else {
            for (int i = cost.length - 3; i >= 0; i--) {
                int a = cost[i];
                int b = cost[i + 1];
                int c = cost[i + 2];
                cost[i] = a + Math.min(b, c);
            }
            return Math.min(cost[0], cost[1]);
        }
    }

    public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
        if (maxChoosableInteger >= desiredTotal) return true;
        int checkSum = sumSeries(maxChoosableInteger);
        if (checkSum < desiredTotal) return false;
        if ((maxChoosableInteger + 1) < desiredTotal) return true;
        return false;
    }

    public int sumSeries(int x) {
        return x * (x + 1) / 2;
    }


    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Map<Integer, List<List<Integer>>> hmList = new HashMap<>();

        // Base case: add an empty list for target 0
        List<List<Integer>> baseCaseList = new ArrayList<>();
        baseCaseList.add(new ArrayList<>());
        hmList.put(0, baseCaseList);

        Arrays.sort(candidates); // Sort candidates to optimize the loop

        for (int i = 1; i <= target; i++) {
            List<List<Integer>> currentList = new ArrayList<>();

            for (int candidate : candidates) {
                int current = i - candidate;
                if (current < 0) break; // If current is negative, no need to proceed further
                else {
                    List<List<Integer>> previousLists = hmList.getOrDefault(current, new ArrayList<>());

                    for (List<Integer> list : previousLists) {
                        List<Integer> newList = new ArrayList<>(list); // Create a new list to avoid modifying the original
                        newList.add(candidate);
                        currentList.add(newList);
                    }
                }
            }

            if (!currentList.isEmpty()) {
                hmList.put(i, currentList);
            }
        }
        List<List<Integer>> out = hmList.getOrDefault(target, new ArrayList<>());
        Set<List<Integer>> outSet = new HashSet<>();
        for (var v : out) {
            Collections.sort(v);
            outSet.add(v);
        }

        return (new ArrayList<>(outSet));
    }


    public List<List<Integer>> combinationSum3(int k, int n) {
        return combSum3(n, 1, k);
    }

    public List<List<Integer>> combSum3(int target, int start, int k) {
        List<List<Integer>> out = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        if (start >= 10) return out;
        if (k == 1) {
            if (start <= target && target < 10) {
                list.add(target);
                out.add(list);
                return out;
            } else {
                return out;
            }
        } else {
            for (int i = start; i < 10; i++) {
                int newStart = i + 1;
                int newTarget = target - i;
                int newK = k - 1;
                if (newTarget < newStart) {
                    break;
                } else {
                    List<List<Integer>> doubleList = new ArrayList<>(combSum3(newTarget, newStart, newK));
                    if (doubleList.size() == 0) {
                        continue;
                    } else {
                        for (List<Integer> l : doubleList) {
                            List<Integer> current = new ArrayList<>(l);
                            current.add(0, i);
                            out.add(current);
                        }
                    }
                }
            }
            return out;
        }
    }

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int i : candidates) pq.add(i);
        return combSum2(pq, target);
    }

    public List<List<Integer>> combSum2(PriorityQueue<Integer> pq, int target) {
        System.out.println("pq and target at start");
        System.out.println(pq);
        System.out.println(target);
        List<List<Integer>> out = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        if (target == 0) {
            out.add(list);
            return out;
        }
        if (pq.size() == 0) return out;
        if (pq.size() == 1) {
            Integer i = pq.poll();
            if (i == target) {
                list.add(i);
                out.add(list);
                return out;
            } else {
                return out;
            }
        } else {
            while (!pq.isEmpty()) {
                Integer i = pq.poll();
                PriorityQueue<Integer> pqNext = new PriorityQueue<>(pq);
//                PriorityQueue<Integer> pqNext = new PriorityQueue<>();
//                for (Integer ii : pq) pqNext.add(ii);
                int newTarget = target - i;
                if (newTarget <= 0) {
                    return out;
                } else {
                    List<List<Integer>> doubleList = combSum2(pqNext, newTarget);
//                    List<List<Integer>> doubleList = new ArrayList<>(combSum2(pqNext, newTarget));
                    if (doubleList.size() == 0) {
                        continue;
                    } else {
                        for (List<Integer> l : doubleList) {
                            List<Integer> current = new ArrayList<>(l);
                            current.add(0, i);
                            if (!out.contains(current)) out.add(current);
                        }
                    }
                }
            }
            return out;
        }
    }


    public int videoStitching(int[][] clips, int time) {
        int[] dpArray = new int[time + 10];
        Arrays.fill(dpArray, 101);
        dpArray[0] = 0;
        Arrays.sort(clips, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                int a1 = a[0];
                int a2 = a[1];
                int b1 = b[0];
                int b2 = b[1];
                if (a1 != b1) return (a1 - b1);
                return (a2 - b2);
            }
        });
        for (int[] clip : clips) {
            int start = clip[0];
            if (start >= time) continue;

            int end = Math.min(clip[1], time);
            int newMinClips = dpArray[start] + 1;
            for (int i = start; i <= end; i++) {
                dpArray[i] = Math.min(dpArray[i], newMinClips);
            }
        }
        return (dpArray[time] == 101) ? -1 : dpArray[time];
    }


    public int numMatchingSubseq(String s, String[] words) {
        HashMap<String, Deque> queueMap = new HashMap<>();
        HashMap<String, Integer> frequencyMap = new HashMap<>();
        int n = words.length;
        for (int i = 0; i < n; i++) {
            String currentString = words[i];
            char[] charArray = currentString.toCharArray();
            if (frequencyMap.containsKey(currentString)) {
                frequencyMap.merge(currentString, 1, Integer::sum);
            } else {
                Deque<Character> q = new ArrayDeque<>();
                for (Character c : charArray) q.addLast(c);
                queueMap.put(currentString, q);
                frequencyMap.put(currentString, 1);
            }
        }
        int sLength = s.length();
        for (int i = 0; i < sLength; i++) {
            Character c = s.charAt(i);
            for (Deque q : queueMap.values()) {
                if (q.isEmpty()) continue;
                if (q.peek() == c) q.pollFirst();
            }
        }
        Set<String> keySet = queueMap.keySet();
        int out = 0;
        for (String key : keySet) {
            if (queueMap.get(key).isEmpty()) {
                out += frequencyMap.get(key);
            }
        }
        return out;
    }


    public int minSumOfLengths(int[] arr, int target) {
        int n = arr.length;
        int[] dpArrayRightToLeft = new int[n];
        Arrays.fill(dpArrayRightToLeft, n);
        int currentMinimumLength = n;
        for (int i = n - 1; i >= 0; i--) {
            int currentSum = 0;
            int currentLength = 1;
            int iterateBound = Math.min(n, i + currentMinimumLength);
            for (int j = i; j < iterateBound; j++, currentLength++) {
                currentSum += arr[j];
                if (currentSum == target) {
                    currentMinimumLength = Math.min(currentMinimumLength, currentLength);
                    break;
                }
            }
            dpArrayRightToLeft[i] = currentMinimumLength;
        }
        int[] dpArrayLeftToRight = new int[n];
        Arrays.fill(dpArrayLeftToRight, n);
        currentMinimumLength = n;
        for (int i = 0; i < n; i++) {
            int currentSum = 0;
            int currentLength = 1;
            int iterateBound = Math.max(-1, i - currentMinimumLength - 1);
            System.out.println("i, currentsum, currentlength, ib");
            System.out.println(i);
            System.out.println(currentSum);
            System.out.println(currentLength);
            System.out.println(iterateBound);
            for (int j = i; j > iterateBound; j--, currentLength++) {
                System.out.println(j);
                currentSum += arr[j];
                if (currentSum == target) {
                    currentMinimumLength = Math.min(currentMinimumLength, currentLength);
                    System.out.println(j);
                    System.out.println(currentSum);
                    System.out.println(currentMinimumLength);

                    break;
                }
            }
            dpArrayLeftToRight[i] = Math.min(currentMinimumLength, currentLength);
        }
        int out = n * 2;
        for (int i = 0; i < n - 1; i++) {
            int a = dpArrayLeftToRight[i];
            int b = dpArrayRightToLeft[i + 1];
            if (a == n || b == n) continue;
            int c = a + b;
            out = Math.min(out, c);
        }
        System.out.println(Arrays.toString(dpArrayLeftToRight));
        System.out.println(Arrays.toString(dpArrayRightToLeft));
        return (out == n * 2) ? -1 : out;
    }


    public int maxRepeating(String sequence, String word) {
        int n = sequence.length();
        int bound = word.length();
        int out = 0;
        int index = 0;
        for (int i = 0; i < n; i++) {
            char c = sequence.charAt(i);
            char cc = word.charAt(index);
            if (c == cc) {
                index++;
            }
            if (index == bound) {
                out++;
                index = 0;
            }
        }
        return out;
    }


    int boardMax;
    double[][] chessBoard;

    public double knightProbability(int n, int k, int row, int column) {
        boardMax = n;
        chessBoard = new double[n][n];
        for (double[] dArr : chessBoard) {
            Arrays.fill(dArr, 1);
        }
        generateProbabilityBoard(k);
        return getBoardAtPos(row, column);
    }


    public void generateProbabilityBoard(int k) {
        if (k == 0) return;
        else {
            double[][] currentBoard = new double[boardMax][boardMax];
            for (int x = 0; x < boardMax; x++) {
                for (int y = 0; y < boardMax; y++) {
                    double currentCount = 0;
                    currentCount += getBoardAtPos(y - 2, x - 1);
                    currentCount += getBoardAtPos(y - 2, x + 1);
                    currentCount += getBoardAtPos(y - 1, x + 2);
                    currentCount += getBoardAtPos(y + 1, x + 2);
                    currentCount += getBoardAtPos(y + 2, x + 1);
                    currentCount += getBoardAtPos(y + 2, x - 1);
                    currentCount += getBoardAtPos(y + 1, x - 2);
                    currentCount += getBoardAtPos(y - 1, x - 2);
                    currentBoard[y][x] = currentCount / (double) 8;

                    if (y == 1 && x == 2) {
                        System.out.println(currentCount);
                    }
                }
            }
            chessBoard = currentBoard;
            int newK = k - 1;
            generateProbabilityBoard(newK);
        }
    }

    public double getBoardAtPos(int y, int x) {
        boolean xBound = (0 <= x && x < boardMax);
        boolean yBound = (0 <= y && y < boardMax);
        if (xBound && yBound) {
            return chessBoard[y][x];
        } else {
            return 0;
        }
    }

    public int sumSubarrayMins(int[] arr) {
        int n = arr.length;
        if (n == 1) {
            return arr[0];
        } else {
            long[] dpArray = new long[n];
            long out = 0;
            for (int i = n - 1; i >= 0; i--) {
                int current = arr[i];
                dpArray[i] += current;
                for (int j = i + 1; j < n; j++) {
                    int prev = arr[j];
                    if (current >= prev) {
                        dpArray[i] += dpArray[j];
                        break;
                    } else {
                        dpArray[i] += current;
                    }
                }
                out += dpArray[i];
            }
            return (int) (out % (Math.pow(10, 9) + 7));
        }
    }


    public double new21Game(int n, int k, int maxPts) {
        if (n < k) return 0;
        if (k == 1) return (n / maxPts);
        if (n >= (k + maxPts)) return 1;
        double[] dpArray = new double[n + 1];
        dpArray[0] = 1;
        double multiplier = (double) 1 / (double) maxPts;
        for (int i = 1; i < k; i++) {
            for (int j = 1; j <= maxPts; j++) {
                int prev = i - j;
                if (prev < 0) break;
                else {
                    dpArray[i] += dpArray[prev] * multiplier;
                }
            }
        }
        dpArray[k] = 1;
        for (int i = maxPts; i > 0; i--) {
            int prev = n - i;
            if (prev >= k) {
                break;
            } else {
                dpArray[n] += dpArray[prev] * multiplier;
            }
        }
        System.out.println(Arrays.toString(dpArray));
        return dpArray[n];
    }


    class Solution {
        public int maxSubarraySumCircular(int[] nums) {
            int n = nums.length;
            if (n == 1) {
                return nums[0];
            }
            int maxSum = nums[0];
            int currentMaximumSum = 0;
            int minSum = nums[0];
            int currentMinimumSum = 0;
            int totalSum = 0;
            for (int i = 0; i < n; i++) {
                currentMaximumSum += nums[i];
                currentMinimumSum += nums[i];
                totalSum += nums[i];
                maxSum = Math.max(maxSum, currentMaximumSum);
                minSum = Math.min(minSum, currentMinimumSum);
                if (currentMaximumSum < 0) {
                    currentMaximumSum = 0;
                }
                if (currentMinimumSum > 0) {
                    currentMinimumSum = 0;
                }

            }
            return maxSum > 0 ? Math.max(maxSum, totalSum - minSum) : maxSum;
        }

    }


    public char findTheDifference(String s, String t) {
        if (s.length() == 0) {
            return t.charAt(0);
        }
        int[] dp = new int[26];
        for (char c : s.toCharArray()) {
            int index = c - 'a';
            dp[index]++;
        }
        for (char c : t.toCharArray()) {
            int index = c - 'a';
            if (dp[index] == 0) return c;
            else {
                dp[index]--;
            }
        }
        return 'z';
    }

    public int findCircleNum(int[][] isConnected) {
        int n = isConnected.length;
        if (n == 1) {
            return 1;
        }
        Set<Integer> visited = new HashSet<>();
        int output = 0;
        for (int i = 0; i < n; i++) {
            if (visited.contains(i)) {
                continue;
            } else {
                output++;
                Deque<Integer> q = new ArrayDeque<>();
                q.add(i);
                while (!q.isEmpty()) {
                    Integer I = q.poll();
                    if (visited.contains(I)) {
                        continue;
                    } else {
                        visited.add(I);
                        int[] connections = isConnected[I];
                        for (int j = 0; j < n; j++) {
                            if (connections[j] == 1 && !visited.contains(j)) q.addLast(j);
                        }
                    }
                }
            }
        }
        return output;
    }


    public int findUnsortedSubarray(int[] nums) {
        int n = nums.length;
        if (n == 1) {
            return 0;
        } else if (n == 2) {
            if (nums[0] <= nums[1]) {
                return 0;
            } else {
                return 2;
            }
        } else {
            int subarrayMinimum = Integer.MAX_VALUE;
            int subarrayMaximum = Integer.MIN_VALUE;
            boolean flag = false;
            for (int i = 0; i < n - 1; i++) {
                int first = nums[i];
                int second = nums[i + 1];
                if (first > second) {
                    flag = true;
                    subarrayMaximum = Math.max(subarrayMaximum, first);
                    subarrayMinimum = Math.min(subarrayMinimum, second);
                }
            }
            if (!flag) return 0;

            int minIndex = 0;
            int current = nums[minIndex];
            while (current <= subarrayMinimum) {
                minIndex++;
                current = nums[minIndex];
            }

            int maxIndex = n - 1;
            current = nums[maxIndex];
            while (current >= subarrayMaximum) {
                maxIndex--;
                current = nums[maxIndex];
            }

            return (maxIndex - minIndex + 1);


        }

    }


    public int[] asteroidCollision(int[] asteroids) {
        Deque<Integer> q = new ArrayDeque<>();
        int n = asteroids.length;
        q.addLast(asteroids[0]);
        for (int i = 1; i < n; i++) {
            int asteroid = asteroids[i];
            if (asteroid > 0) {
                q.addLast(asteroid);
            } else {
                asteroid = Math.abs(asteroid);
                while (!q.isEmpty()) {
                    Integer I = q.peekLast();
                    if (I > asteroid || I < 0) {
                        if (I >= asteroid) asteroid = 0;
                        break;
                    } else {
                        q.pollLast();
                    }
                }
                if (asteroid != 0) q.addLast(asteroid * -1);
            }

        }
        int[] output = q.stream().mapToInt(i -> i).toArray();
        return output;
    }


    public boolean canVisitAllRooms(List<List<Integer>> rooms) {
        int n = rooms.size();
        Set<Integer> visited = new HashSet<>();
        Deque<Integer> q = new ArrayDeque<>();
        q.add(0);
        while (!q.isEmpty()) {
            Integer currentRoom = q.pollFirst();
            if (visited.contains(currentRoom)) {
                continue;
            } else {
                visited.add(currentRoom);
                List<Integer> keys = rooms.get(currentRoom);
                for (Integer key : keys) {
                    if (!visited.contains(key)) {
                        q.addLast(key);
                    }
                }
            }
        }
        return (visited.size() == n);
    }

    public int mincostTickets(int[] days, int[] costs) {
        int oneDay = costs[0];
        int sevenDay = costs[1];
        int thirtyDay = costs[2];
        int firstDay = days[0];
        Deque<Integer> q = new ArrayDeque<>();
        for (Integer I : days) {
            q.addFirst(I);
        }
        int endDay = q.peek() + 1;
        int[] dpArray = new int[endDay + 1];
        Arrays.fill(dpArray, Integer.MAX_VALUE);
        int pointerDay = q.poll();
        dpArray[pointerDay] = Math.min(oneDay, Math.min(sevenDay, thirtyDay));
        dpArray[endDay] = 0;
        while (!q.isEmpty()) {
            int day = q.poll();
            while (pointerDay != day + 1) {
                pointerDay--;
                dpArray[pointerDay] = Math.min(dpArray[pointerDay], dpArray[pointerDay + 1]);
            }
            int oneDayLater = day + 1;
            int sevenDayLater = Math.min(day + 7, endDay);
            int thirtyDayLater = Math.min(day + 30, endDay);
            oneDayLater = dpArray[oneDayLater] + oneDay;
            sevenDayLater = dpArray[sevenDayLater] + sevenDay;
            thirtyDayLater = dpArray[thirtyDayLater] + thirtyDay;
            dpArray[day] = Math.min(oneDayLater, Math.min(sevenDayLater, thirtyDayLater));
        }
        System.out.println(Arrays.toString(dpArray));
        return dpArray[firstDay];
    }


    public int minIncrementForUnique(int[] nums) {
        int n = nums.length;
        if (n == 1) {
            return 0;
        }
        Arrays.sort(nums);
        int minimum = -1;
        int output = 0;
        for (int i = 0; i < n; i++) {
            int current = nums[i];
            if (current > minimum) {
                minimum = current + 1;
            } else {
                output += (minimum - current);
                minimum = minimum + 1;
            }
        }
        return output;
    }


    public int[] prisonAfterNDays(int[] cells, int n) {
        n = (n - 1) % 14 + 1;
        int length = cells.length;
        if (n == 0) {
            return cells;
        } else {
            int[] nextCells = new int[length];
            for (int i = 1; i < n - 1; i++) {
                nextCells[i] = (cells[i - 1] == cells[i + 1]) ? 1 : 0;
            }
            return prisonAfterNDays(nextCells, n - 1);
        }
    }


    public int[][] kClosest(int[][] points, int k) {
        Arrays.sort(points, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                double x1 = a[0];
                double y1 = a[1];
                double x2 = b[0];
                double y2 = b[1];
                double distance1 = Math.sqrt(Math.pow(x1, 2) + Math.pow(y1, 2));
                double distance2 = Math.sqrt(Math.pow(x2, 2) + Math.pow(y2, 2));
                if (distance1 < distance2) return -1;
                else return 1;
            }
        });
        int[][] out = new int[k][];
        for (int i = 0; i < k; i++) {
            out[i] = points[i];
        }
        return out;
    }


    public int totalFruit(int[] fruits) {
        int n = fruits.length;
        if (n <= 2) return n;
        int fruit1Type = -1;
        int fruit2Type = fruits[0];
        int fruit1 = 0;
        int fruit2 = 0;
        int output = 0;
        for (int i = 0; i < n; i++) {
            int fruit = fruits[i];
            if (fruit != fruit1Type && fruit != fruit2Type) {
                output = Math.max(output, fruit1 + fruit2);
                fruit1 = fruit2;
                fruit1Type = fruit2Type;
                fruit2 = 1;
                fruit2Type = fruit;

            } else if (fruit == fruit1Type) {
                fruit1++;
            } else {
                fruit2++;
            }
        }
        output = Math.max(output, fruit1 + fruit2);
        return output;
    }


    public int twoCitySchedCost(int[][] costs) {
        Arrays.sort(costs, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                int a1 = a[0];
                int a2 = a[1];
                int b1 = b[0];
                int b2 = b[1];
                return ((a1 - a2) - (b1 - b2));
            }
        });
        Deque<int[]> q = new ArrayDeque<>();
        for (int[] c : costs) {
            q.addLast(c);
        }
        int out = 0;
        while (!q.isEmpty()) {
            int[] left = q.pollFirst();
            int[] right = q.pollLast();
            out += left[0];
            out += right[1];
        }
        return out;
    }


    public List<List<Integer>> combine(int n, int k) {
        return combine2(n, k, 1);
    }

    public List<List<Integer>> combine2(int n, int k, int start) {
        List<List<Integer>> output = new ArrayList<>();
        if (k == 1) {
            for (int i = start; i <= n; i++) {
                List<Integer> list = new ArrayList<>();
                list.add(i);
                output.add(list);
            }
            return output;
        } else {
            int newK = k - 1;
            int upperStartBoundIncl = n - newK;
            for (int i = start; i <= upperStartBoundIncl; i++) {
                List<List<Integer>> nextCombOutput = new ArrayList<>();
                nextCombOutput = combine2(n, newK, i + 1);
                for (var v : nextCombOutput) {
                    List<Integer> newList = new ArrayList<>(v);
                    newList.add(0, i);
                    output.add(newList);
                }
            }
            return output;
        }
    }


    HashMap<Integer, List<String>> stringHM;

    public List<String> letterCombinations(String digits) {
        if (digits.length() == 0) {
            return new ArrayList<>();
        }
        stringHM = new HashMap<>();
        List<String> a = Arrays.asList("a", "b", "c");
        List<String> b = Arrays.asList("d", "e", "f");
        List<String> c = Arrays.asList("g", "h", "i");
        List<String> d = Arrays.asList("j", "k", "l");
        List<String> e = Arrays.asList("m", "n", "o");
        List<String> f = Arrays.asList("p", "q", "r", "s");
        List<String> g = Arrays.asList("t", "u", "v");
        List<String> h = Arrays.asList("w", "x", "y", "z");
        stringHM.put(2, a);
        stringHM.put(3, b);
        stringHM.put(4, c);
        stringHM.put(5, d);
        stringHM.put(6, e);
        stringHM.put(7, f);
        stringHM.put(8, g);
        stringHM.put(9, h);
        stringHM.put(1, new ArrayList<>());
        return letterCombinations2(digits);
    }

    public List<String> letterCombinations2(String s) {
        List<String> output = new ArrayList<>();
        Integer I = Integer.valueOf(String.valueOf(s.charAt(0)));
        if (s.length() == 1) {
            return stringHM.get(I);
        } else {
            String xs = s.substring(1);
            List<String> currentList = stringHM.get(I);
            List<String> nextList = letterCombinations2(xs);
            for (String a : currentList) {
                for (String b : nextList) {
                    String c = a + b;
                    output.add(c);
                }
            }
            return output;
        }
    }


//    Boolean[][] cache;
//    public boolean canPartition(int[] nums) {
//        int n = nums.length;
//        if (n == 1) return false;
//        Integer I = 0;
//        for (int i : nums) I += i;
//        if (I % 2 != 0) return false;
//        int halfSum = I / 2;
//        cache = new Boolean[n+1][halfSum+1];
//        return canPartition2(nums, 0, halfSum);
//    }
//    public boolean canPartition2(int[] nums,int index, int halfSumTarget) {
//        int n = nums.length;
//        if (halfSumTarget == 0) return true;
//        if (index == n && halfSumTarget != 0) return false;
//        if (index == n && halfSumTarget == 0) return true;
//        if (halfSumTarget < 0) return false;
//        if (cache[index][halfSumTarget] != null) return cache[index][halfSumTarget];
//        int newTarget = halfSumTarget - nums[index];
//        cache[index][halfSumTarget] = (canPartition2(nums, index+1, newTarget) || canPartition2(nums, index+1, halfSumTarget));
//        return cache[index][halfSumTarget];
//    }


    public int findTargetSumWays(int[] nums, int target) {
        int n = nums.length;
        if (n == 1) {
            if (target - nums[0] == 0) return 1;
            if (target + nums[0] == 0) return 1;
            return 0;
        } else {
            int current = nums[0];
            int newTargetA = target - current;
            int newTargetB = target + current;
            int[] nextNums = Arrays.copyOfRange(nums, 1, nums.length);
            newTargetA = findTargetSumWays(nextNums, newTargetA);
            newTargetB = findTargetSumWays(nextNums, newTargetB);
            return newTargetA + newTargetB;

        }
    }


    public int maxTurbulenceSize(int[] arr) {
        int n = arr.length;
        if (n == 1) {
            return 1;
        }
        int output = 0;
        int turbulentSum = 0;
        int inverseTurbulentSum = 0;
        int previous = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            int current = arr[i];
            if (previous < current) {
                turbulentSum++;
                output = Math.max(output, inverseTurbulentSum);
                inverseTurbulentSum = 1;
            } else if (previous > current) {
                inverseTurbulentSum++;
                output = Math.max(output, turbulentSum);
                turbulentSum = 1;
            } else {
                output = Math.max(output, turbulentSum);
                output = Math.max(output, inverseTurbulentSum);
                inverseTurbulentSum = 1;
                turbulentSum = 1;
            }
            int cache = turbulentSum;
            turbulentSum = inverseTurbulentSum;
            inverseTurbulentSum = cache;
            previous = current;
        }
        output = Math.max(output, turbulentSum);
        output = Math.max(output, inverseTurbulentSum);
        return output;
    }


    Integer[][] cache;

    public int subarraySum(int[] nums, int k) {
        int n = nums.length;
        if (n == 1) return (nums[0] == k) ? 1 : 0;
        cache = new Integer[n + 1][k + 1];
        return subarraySum2(nums, 0, k);
    }

    public int subarraySum2(int[] nums, int index, int target) {
        int n = nums.length;
        if (target == 0) return 1;
        if (target < 0) return 0;
        if (index == n) return (target == 0) ? 1 : 0;
        if (cache[index][target] != null) return cache[index][target];
        else {
            int newTarget = target - nums[index];
            cache[index][target] = subarraySum2(nums, index + 1, target) + subarraySum2(nums, index + 1, newTarget);
            return cache[index][target];
        }
    }

    public int nextGreaterElement(int n) {
        if (n <= 11) return -1;
        String s = String.valueOf(n);
        List<Integer> digits = new ArrayList<>();
        for (int i = 0; i < s.length(); i++) {
            digits.add(s.charAt(i) - '0');
        }
        Collections.sort(digits, Collections.reverseOrder());
        int output = 0;
        for (Integer i : digits) {
            output = output * 10;
            output += i;
        }
        return (output == n) ? -1 : output;
    }


    public List<String> topKFrequent(String[] words, int k) {
        HashMap<String, Integer> frequencyMap = new HashMap<>();
        for (String s : words) {
            frequencyMap.merge(s, 1, Integer::sum);
        }
        List<String> keyList = new ArrayList<>(frequencyMap.keySet());
        Collections.sort(keyList, new Comparator<String>() {
            public int compare(String a, String b) {
                int freqA = frequencyMap.get(a);
                int freqB = frequencyMap.get(b);
                if (freqA != freqB) return freqB - freqA;
                else return a.compareTo(b);
            }
        });
        List<String> output = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            output.add(keyList.get(i));
        }
        return output;
    }


    public boolean checkPossibility(int[] nums) {
        int count = 0;
        int n = nums.length;
        boolean flag = false;
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            int current = nums[i];
            if (current < max) {
                if (flag) return false;
                else {
                    flag = true;
                }
            } else {
                max = current;
            }
        }
        return true;
    }


    int univaluePathLength;

    public int longestUnivaluePath(TreeNode root) {
        univaluePathLength = 0;
        if (root == null) return univaluePathLength;
        univaluePathSearch(root, 0);
        return univaluePathLength;
    }

    public void univalueTraverse(TreeNode root) {
        if (root == null) return;
        Integer current = root.val;
        Integer leftChild = (root.left != null) ? root.left.val : null;
        Integer rightChild = (root.right != null) ? root.right.val : null;
        if ((leftChild != null && leftChild == current)
                || (rightChild != null && rightChild == current)) {
            univaluePathSearch(root, current);
        } else {
            univalueTraverse(root.left);
            univalueTraverse(root.right);
        }
    }

    public void univaluePathSearch(TreeNode root, int target) {
        if (root == null) return;
        Deque<TreeNode> q = new ArrayDeque<>();
        q.add(root);
        int counter = 0;
        while (!q.isEmpty()) {
            TreeNode currentNode = q.pop();
            if (currentNode != null) {
                int currentVal = currentNode.val;
                if (currentVal == target) {
                    counter++;
                    q.add(root.left);
                    q.add(root.right);
                }
            }
        }
        univaluePathLength = Math.max(univaluePathLength, counter);
    }


    public int repeatedStringMatch(String a, String b) {
        if (b.equals("")) return 0;
        int aLength = a.length();
        int bIndex = 0;
        boolean flag = false;
        for (int i = 0; i < aLength; i++) {
            char aChar = a.charAt(i);
            char bChar = b.charAt(bIndex);
            if (aChar == bChar) {
                flag = true;
                bIndex++;
            } else {
                if (flag) return 0;
            }
        }
        if (bIndex == 0) return 0;
        else {
            return (1 + repeatedStringMatch(a, b.substring(bIndex)));
        }
    }

    public int leastInterval(char[] tasks, int n) {
        int[] jobMap = new int[26];
        for (char c : tasks) {
            jobMap[(c - 'A')]++;
        }
        Arrays.sort(jobMap);
        //Index 25 is the job with the most frequency
        int minimumDuration = n * jobMap[25];
        int vacantSlots = (n - 1) * jobMap[25];
        int remainingJobs = tasks.length - jobMap[25];
        if (remainingJobs > vacantSlots) return tasks.length;
        else return minimumDuration;
    }


    double[][] dpDouble;

    public double soupServings(int n) {
        if (n <= 25) {
            return 0.25 * (1 + 0.5 + 0.5 + 0.5);
        }
        dpDouble = new double[n + 1][n + 1];
        for (double[] d : dpDouble) Arrays.fill(d, -1);
        return soupProbability(n, n);
    }

    public double soupProbability(int a, int b) {
        if (a == b && a <= 0) return 0.5;
        if (a <= 0 && b > 0) return 1;
        if (a > 0 && b <= 0) return 0;
        if (dpDouble[a][b] != -1) return dpDouble[a][b];
        double Q = soupProbability(Math.max(a - 100, 0), b);
        double W = soupProbability(Math.max(a - 75, 0), Math.max(b - 25, 0));
        double E = soupProbability(Math.max(a - 50, 0), Math.max(b - 50, 0));
        double R = soupProbability(Math.max(a - 25, 0), Math.max(b - 75, 0));
        double T = 0.25 * (Q + W + E + R);
        return dpDouble[a][b] = T;
    }


    public double[] sampleStats(int[] count) {
        int n = count.length;
        double sampleSize = 0;
        long sum = 0;
        double minimum = -1, maximum = 0, mean = 0, median = 0, mode = 0;
        int modeFrequency = -1;

        // Process each value in 'count'
        for (int i = 0; i < n; i++) {
            int currentFrequency = count[i];
            if (currentFrequency > 0) {
                sampleSize += currentFrequency;
                sum += ((long) i * currentFrequency);
                maximum = i;
                if (minimum == -1) minimum = i;
                if (modeFrequency < currentFrequency) {
                    modeFrequency = currentFrequency;
                    mode = i;
                }
            }
        }

        if (sampleSize == 0) return new double[]{minimum, maximum, mean, median, mode};

        mean = sum / sampleSize;

        // Find median
        int countSoFar = 0;
        int medianIndex1 = ((int) sampleSize - 1) / 2;
        int medianIndex2 = (int) sampleSize / 2;
        for (int i = 0; i < n; i++) {
            if (count[i] > 0) {
                countSoFar += count[i];
                if (median == 0 && countSoFar > medianIndex1) {
                    median += i; // Found the lower median index
                }
                if (countSoFar > medianIndex2) {
                    median += i; // Found the upper median index
                    median /= 2.0; // Calculate the average of the two indices if necessary
                    break;
                }
            }
        }

        return new double[]{minimum, maximum, mean, median, mode};
    }

    public List<Boolean> camelMatch(String[] queries, String pattern) {
        List<Boolean> output = new ArrayList<>();
        // char c - 'A';
        //32 <= lowercase <= 57
        //0 <= uppercase <= 25
        for (String s : queries) {
            int n = s.length();
            int patternIndex = 0;
            for (int i = 0; i < n; i++) {
                int sCharIndex = s.charAt(i) - 'A';
                if (patternIndex == pattern.length()) {
                    if (sCharIndex <= 25) {
                        patternIndex = 0;
                        break;
                    } else {
                        continue;
                    }
                }
                int patternCharIndex = pattern.charAt(patternIndex) - 'A';
                if (sCharIndex == patternCharIndex) {
                    patternIndex++;
                } else if (sCharIndex <= 25 && patternCharIndex != sCharIndex) {
                    break;
                }
            }
            if (patternIndex == pattern.length()) {
                output.add(true);
            } else {
                output.add(false);
            }
        }
        return output;
    }

    public boolean carPooling(int[][] trips, int capacity) {
        int[] prefix = new int[1001];
        int maxTripEnd = 0;
        for (int[] trip : trips) {
            int a = trip[0];
            int b = trip[1];
            int c = trip[2];
            prefix[b] += a;
            prefix[c] -= a;
            maxTripEnd = Math.max(maxTripEnd, c);
        }
        int currentCapacity = capacity;
        for (int i = 0; i < maxTripEnd + 1; i++) {
            currentCapacity -= prefix[i];
            if (currentCapacity < 0) return false;
        }
        return (currentCapacity < 0) ? false : true;
    }


    public int getLastMoment(int n, int[] left, int[] right) {
        int out = 0;
        for (int i : left) out = Math.max(out, i);
        for (int i : right) out = Math.max(out, n - i);
        return out;
    }


    public int digArtifacts(int n, int[][] artifacts, int[][] dig) {
        int[][] dpMatrix = new int[n][n];
        for (int[] d : dig) {
            int y = d[0];
            int x = d[1];
            dpMatrix[y][x] = 1;
        }
        int output = 0;
        for (int[] artifact : artifacts) {
            int yStart = artifact[0];
            int xStart = artifact[1];
            int yEnd = artifact[2];
            int xEnd = artifact[3];
            boolean flag = true;
            for (int y = yStart; y <= yEnd; y++) {
                for (int x = xStart; x <= xEnd; x++) {
                    if (dpMatrix[y][x] == 0) {
                        flag = false;
                        break;
                    }
                }
                if (!flag) break;
            }
            if (flag) output++;
        }
        return output;
    }


    public int[] deckRevealedIncreasing(int[] deck) {
        int n = deck.length;
        int[] out = new int[n];
        if (n == 1) return deck;
        if (n == 2) {
            out[0] = Math.min(deck[0], deck[1]);
            out[1] = Math.max(deck[0], deck[1]);
            return out;
        } else {
            Arrays.sort(deck);
            Deque<Integer> q = new ArrayDeque<>();
            q.addFirst(deck[n - 1]);
            q.addFirst(deck[n - 2]);
            for (int i = n - 3; i >= 0; i--) {
                Integer I = q.pollLast();
                q.addFirst(I);
                q.addFirst(deck[i]);
            }
            out = q.stream().mapToInt(i -> i).toArray();
            return out;
        }
    }


    public int numWaterBottles(int numBottles, int numExchange) {
        return numWaterBottles2(numBottles, 0, numExchange);
    }

    public int numWaterBottles2(int numFull, int numEmpty, int numExchange) {
        if (numFull == 0 && numEmpty < numExchange) return 0;
        if (numFull == 0 && numEmpty >= numExchange) {
            return numWaterBottles2((numEmpty / numExchange), (numEmpty % numExchange), numExchange);
        } else {
            return (numFull + numWaterBottles2(0, numEmpty + numFull, numExchange));
        }
    }

    public int maxBottlesDrunk(int numBottles, int numExchange) {
        return maxBottlesDrunk(numBottles, 0, numExchange);
    }

    public int maxBottlesDrunk(int numFull, int numEmpty, int numExchange) {
        if (numFull == 0 && numEmpty < numExchange) return 0;
        if (numFull == 0 && numEmpty >= numExchange) {
            return maxBottlesDrunk(1, numEmpty - numExchange, numExchange + 1);
        } else {
            return (numFull + maxBottlesDrunk(0, numEmpty + numFull, numExchange));
        }
    }


    public void gameOfLife(int[][] board) {
        if (board == null || board.length == 0 || board[0].length == 0) return;
        GoL gameState = new GoL(board);
        int[][] newState = gameState.updateState();
        for (int y = 0; y < board.length; y++) {
            for (int x = 0; x < board[0].length; x++) {
                board[y][x] = newState[y][x];
            }
        }
    }

    class GoL {
        int[][] gameState;
        int yMax;
        int xMax;

        public GoL(int[][] gameState) {
            this.gameState = gameState;
            yMax = gameState.length;
            xMax = gameState[0].length;
        }

        public int get(int y, int x) {
            if (x < 0 || y < 0 || x >= xMax || y >= yMax) {
                return 0;
            } else {
                return gameState[y][x];
            }
        }

        public int updateCell(int y, int x) {
            int neighbourSum = 0;
            // Correcting the order of x and y in get() calls
            neighbourSum += get(y-1, x-1);
            neighbourSum += get(y-1, x);
            neighbourSum += get(y-1, x+1);
            neighbourSum += get(y, x-1);
            neighbourSum += get(y, x+1);
            neighbourSum += get(y+1, x-1);
            neighbourSum += get(y+1, x);
            neighbourSum += get(y+1, x+1);

            boolean alive = (gameState[y][x] == 1);
            if (alive) {
                if (neighbourSum < 2 || neighbourSum > 3) {
                    return 0;
                } else {
                    return 1;
                }
            } else {
                if (neighbourSum == 3) return 1;
                else return 0;
            }
        }

        public int[][] updateState() {
            int[][] output = new int[yMax][xMax];
            for (int y = 0; y < yMax; y++) {
                for (int x = 0; x < xMax; x++) {
                    output[y][x] = updateCell(y, x);
                }
            }
            return output;
        }
    }

    public String frequencySort(String s) {
        int n = s.length();
        if (n <= 2) return s;
        HashMap<Character, Integer> charFrequencyMap = new HashMap<>();
        for (Character c : s.toCharArray()) {
            charFrequencyMap.merge(c, 1, Integer::sum);
        }
        List<Character> keyList = new ArrayList<>(charFrequencyMap.keySet());
        Collections.sort(keyList, new Comparator<Character>() {
            public int compare(Character a, Character b) {
                return (charFrequencyMap.get(b) - charFrequencyMap.get(a));
            }
        });
        String output = "";
        for (Character c : keyList) {
            int count = charFrequencyMap.get(c);
            for (int i = 0; i < count; i++) {
                output = output + c;
            }
        }
        return output;
    }






    public static void main(String[] args) {
        Deque<Integer> dQ = new ArrayDeque<>();
        dQ.add(5);
        dQ.add(3);
        dQ.add(2);
        dQ.add(6);
        dQ.add(1);
        dQ.add(1);
        dQ.add(1);
        dQ.add(1);
        dQ.add(8);
        dQ.add(2);
        dQ.add(6);
        dQ.add(4);
        int[] intArray = new int[4];
        List<List<Integer>> doubleList = new ArrayList<>();
        List<List<Integer>> emptyDoubleList = new ArrayList<>();
        List<Integer> emptyList = new ArrayList<>();
        emptyDoubleList.add(emptyList);
        for (var v : emptyDoubleList) {
            v.add(10);
            doubleList.add(v);
        }
        System.out.println(doubleList);


        List<List<Integer>> out = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        out.add(list);
        System.out.println(out.size());

        System.out.println('a' - 'A');
        System.out.println('b' - 'A');
        System.out.println('z' - 'A');
        System.out.println('A' - 'A');
        System.out.println('B' - 'A');
        System.out.println('Z' - 'A');


        System.out.println(9 / 3);

    }


}


/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * boolean param_2 = obj.search(word);
 * boolean param_3 = obj.startsWith(prefix);
 */

class Trie {
    Node root;

    public Trie() {
        root = new Node();
    }

    public void insert(String word) {
        root.insert(word, 0);
    }

    public boolean search(String word) {
        return root.search(word, 0);
    }

    class Node {
        Node[] children;
        boolean eow;

        public Node() {
            children = new Node[26];
        }

        public void insert(String s, int index) {
            if (index == s.length()) {
                return;
            } else {
                char c = s.charAt(index);
                int cIndex = c - 'a';
                if (children[cIndex] == null) {
                    children[cIndex] = new Node();
                }
                children[cIndex].insert(s, index + 1);
                if (index == s.length() - 1) children[cIndex].eow = true;
            }
        }

        public boolean search(String s, int index) {
            if (index == s.length()) return false;
            char c = s.charAt(index);
            int cIndex = c - 'a';
            Node n = children[cIndex];
            if (n == null) return false;
            if (index == s.length() - 1) return n.eow;
            return n.search(s, index + 1);
        }
    }
}
//          return (int) (output % (Math.pow(10,9) + 7));
//          long mod = 1000000007;
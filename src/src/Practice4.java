import java.math.*;
import java.sql.*;
import java.util.*;

public class Practice4 {

    public static List<Integer> maxSubarray(List<Integer> arr) {
        int maxSubArray = Integer.MIN_VALUE;
        int maxSubsequence = 0;
        int currentSum = 0;
        int maxValue = Integer.MIN_VALUE;
        for (int i : arr) {
            if (i > 0) maxSubsequence += i;
            currentSum += i;
            maxSubArray = Math.max(maxSubArray, currentSum);
            currentSum = Math.max(currentSum, 0);
            maxValue = Math.max(maxValue, i);
        }
        if (maxSubsequence == 0) maxSubsequence = maxValue;
        return Arrays.asList(maxSubArray, maxSubsequence);
    }


    public static int cost(List<Integer> B) {
        int n = B.size();
        int[] dpLow = new int[n];
        int[] dpHigh = new int[n];

        dpLow[0] = 0;
        dpHigh[0] = 0;

        for (int i = 1; i < n; i++) {
            dpLow[i] = Math.max(dpLow[i - 1], dpHigh[i - 1] + Math.abs(1 - B.get(i - 1)));
            dpHigh[i] = Math.max(dpLow[i - 1] + Math.abs(B.get(i) - 1), dpHigh[i - 1] + Math.abs(B.get(i) - B.get(i - 1)));
        }
        return Math.max(dpLow[n - 1], dpHigh[n - 1]);

    }


    public int longestOnes(int[] nums, int k) {
        int output = 0;
        int counter = 0;
        if (k == 0) {
            for (int i : nums) {
                switch (i) {
                    case 0 -> {
                        counter = 0;
                    }
                    case 1 -> {
                        counter++;
                        output = Math.max(output, counter);
                    }
                }
            }
        } else {
            Deque<Integer> dq = new ArrayDeque<>();
            for (int i : nums) {
                dq.addLast(i);
                if (i == 0) {
                    if (k == 0) {
                        int front = dq.pollFirst();
                        while (front != 0) front = dq.pollFirst();
                    } else {
                        k = k - 1;
                    }
                }
                output = Math.max(output, dq.size());
            }

        }
        return output;
    }


    public String decodeString(String s) {
        return decodeString(s.toCharArray(), new int[]{0});
    }

    private String decodeString(char[] sChar, int[] index) {
        StringBuilder result = new StringBuilder();
        while (index[0] < sChar.length && sChar[index[0]] != ']') {
            if (Character.isDigit(sChar[index[0]])) {
                int repeat = 0;
                while (Character.isDigit(sChar[index[0]])) {
                    repeat = repeat * 10 + sChar[index[0]] - '0';
                    index[0]++;
                }
                index[0]++; // Skip the '['
                String decodedPart = decodeString(sChar, index);
                index[0]++; // Skip the ']'
                while (repeat-- > 0) {
                    result.append(decodedPart);
                }
            } else {
                result.append(sChar[index[0]++]);
            }
        }
        return result.toString();
    }


    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }


    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        List<Integer> leaves1 = new ArrayList<>();
        List<Integer> leaves2 = new ArrayList<>();
        collectLeaves(root1, leaves1);
        collectLeaves(root2, leaves2);
        return leaves1.equals(leaves2);
    }

    private void collectLeaves(TreeNode node, List<Integer> leaves) {
        if (node == null) {
            return;
        }
        if (node.left == null && node.right == null) {
            leaves.add(node.val);
        } else {
            collectLeaves(node.left, leaves);
            collectLeaves(node.right, leaves);
        }
    }

    public TreeNode searchBST(TreeNode root, int val) {
        if (root == null) return null;
        if (root.val == val) return root;
        if (root.val > val) return searchBST(root.left, val);
        else return searchBST(root.right, val);
    }


    char[][] maze;
    int minSteps = Integer.MAX_VALUE;

    public int nearestExit(char[][] maze, int[] entrance) {
        this.maze = maze;
        int rows = maze.length;
        int cols = maze[0].length;

        Queue<int[]> queue = new LinkedList<>();
        queue.add(new int[]{entrance[0], entrance[1], 0}); // {row, col, steps}
        boolean[][] visited = new boolean[rows][cols];
        visited[entrance[0]][entrance[1]] = true;

        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

        while (!queue.isEmpty()) {
            int[] currentPos = queue.poll();
            int y = currentPos[0];
            int x = currentPos[1];
            int steps = currentPos[2];

            for (int[] dir : directions) {
                int newY = y + dir[0];
                int newX = x + dir[1];

                if (newY >= 0 && newY < rows && newX >= 0 && newX < cols && maze[newY][newX] == '.' && !visited[newY][newX]) {
                    // Check if it's an exit (not the entrance)
                    if (newY == 0 || newY == rows - 1 || newX == 0 || newX == cols - 1) {
                        if (!(newY == entrance[0] && newX == entrance[1])) {
                            return steps + 1;
                        }
                    }

                    // Add the new position to the queue
                    queue.add(new int[]{newY, newX, steps + 1});
                    visited[newY][newX] = true;
                }
            }
        }
        return -1; // If no exit is found
    }


    public long totalCost(int[] costs, int k, int candidates) {
        int n = costs.length;
        long total = 0;

        if (candidates * 2 >= n) {
            Arrays.sort(costs);
            for (int i = 0; i < k; i++) {
                total += costs[i];
            }
            return total;
        }
        PriorityQueue<Integer> left = new PriorityQueue<>(candidates);
        PriorityQueue<Integer> right = new PriorityQueue<>(candidates);
        for (int i = 0; i < candidates; i++) {
            left.offer(costs[i]);
        }
        for (int i = n - 1; i >= n - candidates; i--) {
            right.offer(costs[i]);
        }
        int leftIndex = candidates;
        int rightIndex = n - candidates - 1;
        while (k-- > 0) {
            if (!left.isEmpty() && (right.isEmpty() || left.peek() <= right.peek())) {
                total += left.poll();
                if (leftIndex <= rightIndex) {
                    left.offer(costs[leftIndex++]);
                }
            } else {
                total += right.poll();
                if (leftIndex <= rightIndex) {
                    right.offer(costs[rightIndex--]);
                }
            }
        }
        return total;
    }


    public int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        TreeMap<Integer, Integer> tm = new TreeMap<>();
        for (int i = n - 1; i >= 0; i--) {
            int temperature = temperatures[i];
            int next = Integer.MAX_VALUE;
            Integer higherTemperature = tm.higherKey(temperature);
            while (higherTemperature != null) {
                next = Math.min(next, tm.get(higherTemperature) - i);
                higherTemperature = tm.higherKey(higherTemperature);
            }
            if (next == Integer.MAX_VALUE) next = 0;
            temperatures[i] = next;
            tm.put(temperature, i);
        }
        return temperatures;
    }


    public int[] getConcatenation(int[] nums) {
        int n = nums.length;
        int[] output = new int[2 * n];
        int index = 0;
        for (int i : nums) {
            output[index] = i;
            output[n] = i;
            index++;
            n++;
        }
        return output;
    }

    public int[] buildArray(int[] nums) {
        int n = nums.length;
        int[] output = new int[n];
        for (int i = 0; i < n; i++) {
            output[i] = nums[nums[i]];
        }
        return output;
    }

    public int[] createTargetArray(int[] nums, int[] index) {
        List<Integer> list = new ArrayList<>();
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            int ind = index[i];
            list.add(ind, num);
            System.out.println(num);
            System.out.println(ind);
            System.out.println(list);
        }
        return list.stream().mapToInt(i -> i).toArray();
    }


    public static int substrings(String n) {
        int sLength = n.length();
        long mod = (long) (Math.pow(10, 9) + 7);
        long result = 0;
        long f = 0;

        for (int i = 0; i < sLength; i++) {
            int digit = n.charAt(i) - '0';
            f = (f * 10 + (i + 1) * digit) % mod;
            result = (result + f) % mod;
        }
        return (int) result;
    }


    public int countArrangement(int n) {
        HashMap<Integer, List<Integer>> hm = new HashMap<>();
        for (int i = 1; i <= n; i++) {
            List<Integer> list = new ArrayList<>();
            for (int j = 1; j <= n; j++) {
                if (j % i == 0) {
                    list.add(j);
                    continue;
                }
                if (i % j == 0) list.add(j);
            }
            hm.put(i, list);
        }
        return countArrangement(hm, 0, n, new HashSet<Integer>());
    }

    public int countArrangement(HashMap<Integer, List<Integer>> hm, int index, int n, HashSet<Integer> set) {
        if (index == n + 1) return 1;
        List<Integer> list = hm.get(index);
        int output = 0;
        for (Integer I : list) {
            if (!set.contains(I)) {
                set.add(I);
                output += countArrangement(hm, index + 1, n, set);
                set.remove(I);
            }
        }
        return output;
    }


    public static String compressedString(String word) {
        int n = word.length();
        if (n == 1) return "1" + word;
        char[] charArray = word.toCharArray();
        char prior = charArray[0];
        int length = 0;
        List<Character> list = new ArrayList<>();
        for (int i = 0; i <= n; i++) {
            if (i == n) {
                while (length != 0) {
                    int currentLength = Math.min(9, length);
                    length -= currentLength;
                    list.add(Character.forDigit(currentLength, 10));
                    list.add(prior);
                }
            } else {
                char c = charArray[i];
                if (prior != c) {
                    while (length != 0) {
                        int currentLength = Math.min(9, length);
                        length -= currentLength;
                        list.add(Character.forDigit(currentLength, 10));
                        list.add(prior);
                    }
                }
                length++;
                prior = c;
            }
        }
        String str = list.stream()
                .map(e -> e.toString())
                .reduce((acc, e) -> acc + e)
                .get();
        return str;
    }


    public int minimumCardPickup(int[] cards) {
        int n = cards.length;
        HashMap<Integer, Integer> hm = new HashMap<>();
        int difference = Integer.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            int card = cards[i];
            if (hm.containsKey(card)) {
                difference = Math.min(difference, i - hm.get(card) + 1);
            }
            hm.put(card, i);
        }
        if (difference == Integer.MAX_VALUE) return -1;
        return difference;
    }


    public static long stockmax(List<Integer> prices) {
        long output = 0;
        int n = prices.size();
        if (n == 1) return output;
        Integer maxPrice = -1;
        for (int i = n - 1; i >= 0; i--) {
            Integer currentPrice = prices.get(i);
            if (currentPrice >= maxPrice) maxPrice = currentPrice;
            else {
                output += maxPrice - currentPrice;
            }
        }
        return output;
    }

    public static int redJohn(int n) {
        int[] dpArray = new int[n + 1];
        int prior = 1;
        for (int i = 0; i <= n; i++) {
            dpArray[i] += prior;
            if (i + 4 <= n) dpArray[i + 4] = dpArray[i];
            prior = dpArray[i];
        }
        int permutations = dpArray[n];
        return countPrimes(permutations);
    }

    private static int countPrimes(int n) {
        boolean[] isPrime = new boolean[n + 1];
        Arrays.fill(isPrime, true);
        isPrime[0] = isPrime[1] = false;

        for (int i = 2; i * i <= n; i++) {
            if (isPrime[i]) {
                for (int j = i * i; j <= n; j += i) {
                    isPrime[j] = false;
                }
            }
        }

        int primeCount = 0;
        for (int i = 2; i <= n; i++) {
            if (isPrime[i]) primeCount++;
        }
        return primeCount;
    }


    public static long mandragora(List<Integer> H) {
        int n = H.size();
        Collections.sort(H);
        long[] dpArray = new long[n + 1];
        long currentSum = 0;
        for (int i = n - 1; i >= 0; i--) {
            long multiplier = i + 1;
            currentSum += H.get(i);
            dpArray[i] = Math.max(dpArray[i + 1], currentSum * multiplier);
        }
        return dpArray[0];
    }


    public static int unboundedKnapsack(int k, List<Integer> arr) {
        if (arr.contains(1)) return k;
        boolean[] dpArray = new boolean[k + 1];
        dpArray[0] = true;
        Collections.sort(arr, Collections.reverseOrder());
        for (Integer I : arr) {
            for (int i = 0; i <= k - I; i++) {
                if (dpArray[i]) {
                    if (i == k - I) return k;
                    dpArray[i + I] = true;
                }
            }
        }
        boolean flag = dpArray[k];
        while (!flag) {
            k--;
            flag = dpArray[k];
        }
        return k;
    }


    public List<String> generateParenthesis(int n) {
        // Initialize the dpMatrix
        List<StringBuilder>[][] dpMatrix = new List[n + 1][n + 1];

        // Base case: an empty list for dpMatrix[0][0]
        dpMatrix[0][0] = new ArrayList<>();
        dpMatrix[0][0].add(new StringBuilder());

        // Fill the dpMatrix
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= i; j++) {
                // Initialize the current cell if not already initialized
                if (dpMatrix[i][j] == null) {
                    dpMatrix[i][j] = new ArrayList<>();
                }

                // Add an opening bracket '(' if possible
                if (i > 0 && dpMatrix[i - 1][j] != null) {
                    for (StringBuilder sb : dpMatrix[i - 1][j]) {
                        dpMatrix[i][j].add(new StringBuilder(sb).append('('));
                    }
                }
                // Add a closing bracket ')' if possible
                if (j > 0 && dpMatrix[i][j - 1] != null) {
                    for (StringBuilder sb : dpMatrix[i][j - 1]) {
                        dpMatrix[i][j].add(new StringBuilder(sb).append(')'));
                    }
                }
            }
        }

        // Collect results from dpMatrix[n][n]
        List<String> output = new ArrayList<>();
        for (StringBuilder sb : dpMatrix[n][n]) {
            output.add(sb.toString());
        }

        return output;
    }


    public static String abbreviationOG(String a, String b) {
        char[] yCharArray = b.toCharArray();
        char[] xCharArray = a.toCharArray();
        int yMax = b.length();
        int xMax = a.length();
        boolean[][] dpMatrix = new boolean[yMax + 1][xMax + 1];
        boolean[] last = dpMatrix[yMax];
        last[xMax] = true;
        for (int i = xMax - 1; i >= 0; i--) {
            if (Character.isUpperCase(xCharArray[i])) last[i] = false;
            else last[i] = last[i + 1];
        }
        dpMatrix[yMax] = last;
        for (int y = yMax - 1; y >= 0; y--) {
            for (int x = xMax - 1; x >= 0; x--) {
                char yChar = yCharArray[y];
                char xChar = xCharArray[x];
                if (yChar == xChar) {
                    dpMatrix[y][x] = dpMatrix[y + 1][x + 1];
                } else if (Character.toUpperCase(xChar) == yChar) {
                    dpMatrix[y][x] = dpMatrix[y + 1][x + 1] || dpMatrix[y][x + 1];
                } else {
                    dpMatrix[y][x] = dpMatrix[y][x + 1];
                }
            }
        }
        System.out.println((dpMatrix[0][0]) ? "YES" : "NO");
        return (dpMatrix[0][0]) ? "YES" : "NO";
    }


    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        Arrays.sort(nums);
        for (int i : nums) {
            if (set.contains(i)) return true;
            set.add(i);
        }
        return false;
    }


    public boolean containsNearbyDuplicate(int[] nums, int k) {
        HashMap<Integer, Integer> hm = new HashMap<>();
        int index = 0;
        for (int i : nums) {
            if (hm.containsKey(i)) {
                int priorIndex = hm.get(i);
                if (index - priorIndex <= k) return true;
            }
            hm.put(i, index);
            index++;
        }
        return false;
    }


    public boolean containsNearbyAlmostDuplicate(int[] nums, int indexDiff, int valueDiff) {
        HashMap<Integer, PriorityQueue<Integer>> indexMap = new HashMap<>();
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            if (!indexMap.containsKey(num)) indexMap.put(num, new PriorityQueue<>());
            indexMap.get(num).offer(i);
        }
        List<Integer> keyList = new ArrayList<>(indexMap.keySet());
        Collections.sort(keyList);
        if (valueDiff == 0) {
            for (Integer key : keyList) {
                List<Integer> indexes = new ArrayList<>(indexMap.get(key));
                int size = indexes.size();
                if (size <= 1) continue;
                Collections.sort(indexes);
                for (int i = 1; i < size; i++) {
                    if (indexes.get(i) - indexes.get(i - 1) <= indexDiff) return true;
                }
            }
            return false;
        } else {
            int keyListSize = keyList.size();
            for (int i = 0; i < keyListSize - 1; i++) {
                int lowerKey = keyList.get(i);
                for (int j = i; j < keyListSize; j++) {
                    if (j == i) {
                        List<Integer> indexes = new ArrayList<>(indexMap.get(lowerKey));
                        for (int k = 1; k < indexes.size(); k++) {
                            if (indexes.get(k) - indexes.get(k - 1) <= indexDiff) return true;
                        }
                    } else {
                        int upperKey = keyList.get(j);
                        if (Math.abs(upperKey - lowerKey) > valueDiff) break;
                        PriorityQueue<Integer> lowerPQ = new PriorityQueue<>(indexMap.get(lowerKey));
                        PriorityQueue<Integer> upperPQ = new PriorityQueue<>(indexMap.get(upperKey));
                        while (!lowerPQ.isEmpty() && !upperPQ.isEmpty()) {
                            int a = lowerPQ.peek();
                            int b = upperPQ.peek();
                            if (Math.abs(a - b) <= indexDiff) return true;
                            if (a < b) {
                                lowerPQ.poll();
                            } else {
                                upperPQ.poll();
                            }
                        }
                    }
                }
            }
            return false;
        }
    }


    boolean[][] boolGrid;
    int gridPaths;
    int remainingPaths;

    public int uniquePathsIII(int[][] grid) {
        int yMax = grid.length;
        int xMax = grid[0].length;
        gridPaths = 0;
        remainingPaths = 0;
        boolGrid = new boolean[yMax][xMax];
        int startY = -1;
        int startX = -1;
        int finishY = -1;
        int finishX = -1;
        for (int y = 0; y < yMax; y++) {
            for (int x = 0; x < xMax; x++) {
                int i = grid[y][x];
                if (i == 1) {
                    startY = y;
                    startX = x;
                } else if (i == 2) {
                    finishY = y;
                    finishX = x;
                }
                if (i != -1) {
                    remainingPaths++;
                }
            }
        }
        return gridDFS(startY, startX, finishY, finishX, remainingPaths, grid);
    }

    public int gridDFS(int y, int x, int finishY, int finishX, int remainingPaths, int[][] grid) {
        if (y < 0 || y >= grid.length || x < 0 || x >= grid[0].length || grid[y][x] == -1) {
            return 0;
        }
        if (y == finishY && x == finishX) {
            return (remainingPaths == 1) ? 1 : 0;
        }
        int temp = grid[y][x];
        grid[y][x] = -1;
        remainingPaths--;

        int output = 0;
        output += gridDFS(y + 1, x, finishY, finishX, remainingPaths, grid);
        output += gridDFS(y - 1, x, finishY, finishX, remainingPaths, grid);
        output += gridDFS(y, x + 1, finishY, finishX, remainingPaths, grid);
        output += gridDFS(y, x - 1, finishY, finishX, remainingPaths, grid);
        grid[y][x] = temp;
        remainingPaths++;
        return output;
    }

    char[][] sudokuBoard;
    boolean[][] rowGrid;
    boolean[][] colGrid;
    boolean[][][] boxGrid;

    public void solveSudoku(char[][] board) {
        sudokuBoard = board;
        rowGrid = new boolean[9][9];
        colGrid = new boolean[9][9];
        boxGrid = new boolean[3][3][9];
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    int value = board[i][j] - '1';
                    rowGrid[i][value] = true;
                    colGrid[j][value] = true;
                    boxGrid[i / 3][j / 3][value] = true;
                }
            }
        }
        solve(0, 0);
    }

    private boolean solve(int y, int x) {
        if (y == 9) return true;
        if (x == 9) return solve(y + 1, 0);
        if (sudokuBoard[y][x] != '.') return solve(y, x + 1);
        for (int value = 0; value < 9; value++) {
            if (!rowGrid[y][value] && !colGrid[x][value] && !boxGrid[y / 3][x / 3][value]) {
                sudokuBoard[y][x] = (char) (value + '1');
                rowGrid[y][value] = true;
                colGrid[x][value] = true;
                boxGrid[y / 3][x / 3][value] = true;
                if (solve(y, x + 1)) return true;
                sudokuBoard[y][x] = '.';
                rowGrid[y][value] = false;
                colGrid[x][value] = false;
                boxGrid[y / 3][x / 3][value] = false;
            }
        }
        return false;
    }


    public int maxSatisfaction(int[] satisfaction) {
        int n = satisfaction.length;
        int output = 0;
        Arrays.sort(satisfaction);
        for (int i = 0; i < n; i++) {
            int points = 1;
            int current = 0;
            for (int j = i; j < n; j++) {
                current += (points * satisfaction[j]);
                points++;
            }
            if (current > output) output = current;
            else break;
        }
        return output;
    }


    public int shortestSequence(int[] rolls, int k) {
        boolean[] flagArray = new boolean[k + 1];
        int flagCount = k;
        int sequenceCounter = 0;
        for (int i : rolls) {
            if (!flagArray[i]) {
                flagArray[i] = true;
                flagCount--;
            }
            if (flagCount == 0) {
                sequenceCounter++;
                Arrays.fill(flagArray, false);
            }
        }
        return sequenceCounter;
    }


    public static int longestIncreasingSubsequence(List<Integer> arr) {
        if (arr == null || arr.size() == 0) {
            return 0;
        }
        int n = arr.size();
        Integer[] dpArray = new Integer[n];
        int length = 0;
        for (Integer num : arr) {
            int index = Arrays.binarySearch(dpArray, 0, length, num);
            if (index < 0) {
                index = -(index + 1);
            }
            dpArray[index] = num;
            if (index == length) {
                length++;
            }
        }
        return length;
    }


    public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        if (n == 1) return n;
        int[] dpArray = new int[n];
        int length = 0;
        for (int i : nums) {
            int index = Arrays.binarySearch(dpArray, 0, length, i);
            if (index < 0) {
                index = -(index + 1);
            }
            dpArray[index] = i;
            if (index == length) {
                length++;
            }
        }
        return length;
    }


    public static List<Integer> longestCommonSubsequence(List<Integer> a, List<Integer> b) {
        int yMax = a.size();
        int xMax = b.size();
        int[][] dpMatrix = new int[yMax + 1][xMax + 1];
        for (int y = yMax - 1; y >= 0; y--) {
            for (int x = xMax - 1; x >= 0; x--) {
                if (a.get(y).equals(b.get(x))) {
                    dpMatrix[y][x] = dpMatrix[y + 1][x + 1] + 1;
                } else {
                    dpMatrix[y][x] = Math.max(dpMatrix[y + 1][x], dpMatrix[y][x + 1]);
                }
            }
        }
        List<Integer> lcs = new ArrayList<>();
        int i = 0, j = 0;
        while (i < yMax && j < xMax) {
            if (a.get(i).equals(b.get(j))) {
                lcs.add(a.get(i));
                i++;
                j++;
            } else if (dpMatrix[i + 1][j] >= dpMatrix[i][j + 1]) {
                i++;
            } else {
                j++;
            }
        }
        return lcs;
    }


    public int[] findArray(int[] pref) {
        int n = pref.length;
        if (n == 1) return pref;
        for (int i = n - 1; i > 0; i--) {
            pref[i] = pref[i - 1] ^ pref[i];
        }
        return pref;
    }


    public boolean isStrictlyPalindromic(int n) {
        int max = n - 2;
        for (int i = 2; i <= max; i++) {
            String binaryString = Integer.toString(n, i);
            if (!binaryString.equals(new StringBuilder(binaryString).reverse().toString())) return false;
        }
        return true;
    }

    public int deepestLeavesSum(TreeNode root) {
        HashMap<Integer, Integer> hm = new HashMap<>();
        deepestLeavesSum(root, 0, hm);
        Integer maxKey = Collections.max(hm.keySet());
        return hm.get(maxKey);
    }

    public void deepestLeavesSum(TreeNode root, int level, HashMap<Integer, Integer> hm) {
        if (root == null) return;
        if (root.right == null && root.left == null) {
            int i = root.val;
            hm.merge(level, i, Integer::sum);
        } else {
            deepestLeavesSum(root.left, level + 1, hm);
            deepestLeavesSum(root.right, level + 1, hm);
        }
    }


    public int averageOfSubtree(TreeNode root) {
        int[] a = averageOfSubtree2(root);
        return a[2];
    }

    public int[] averageOfSubtree2(TreeNode root) {
        if (root == null) return new int[3];
        int[] left = averageOfSubtree2(root.left);
        int[] right = averageOfSubtree2(root.right);
        int sum = root.val + left[0] + right[0];
        int nodeCount = 1 + left[1] + right[1];
        int count = left[2] + right[2];
        if (sum / nodeCount == root.val) {
            count++;
        }
        return new int[]{sum, nodeCount, count};
    }


    public double maxProbability(int n, int[][] edges, double[] succProb, int start_node, int end_node) {
        Map<Integer, List<double[]>> graph = new HashMap<>();
        double[] maxProbabilities = new double[n];
        Arrays.fill(maxProbabilities, 0d);
        for (int i = 0; i < edges.length; i++) {
            int a = edges[i][0];
            int b = edges[i][1];
            double probability = succProb[i];
            graph.computeIfAbsent(a, key -> new ArrayList<>()).add(new double[]{b, probability});
            graph.computeIfAbsent(b, key -> new ArrayList<>()).add(new double[]{a, probability});
        }
        PriorityQueue<double[]> pq = new PriorityQueue<>(Comparator.comparingDouble(a -> -a[1]));
        pq.offer(new double[]{start_node, 1.0});
        maxProbabilities[start_node] = 1.0;
        while (!pq.isEmpty()) {
            double[] current = pq.poll();
            int node = (int) current[0];
            double probability = current[1];
            if (node == end_node) {
                return probability;
            }
            if (probability < maxProbabilities[node]) {
                continue;
            }
            if (!graph.containsKey(node)) {
                continue;
            }
            for (double[] neighbor : graph.get(node)) {
                int neighborNode = (int) neighbor[0];
                double edgeProbability = neighbor[1];
                double newProbability = probability * edgeProbability;

                if (newProbability > maxProbabilities[neighborNode]) {
                    maxProbabilities[neighborNode] = newProbability;
                    pq.offer(new double[]{neighborNode, newProbability});
                }
            }
        }
        return maxProbabilities[end_node];
    }


    public int shortestPathLength(int[][] graph) {
        int n = graph.length;
        int finalState = (1 << n) - 1;
        Queue<int[]> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();
        for (int i = 0; i < n; i++) {
            int mask = 1 << i;
            queue.offer(new int[]{i, mask, 0});
            visited.add(i + " " + mask);
        }
        while (!queue.isEmpty()) {
            int[] node = queue.poll();
            int u = node[0], mask = node[1], dist = node[2];
            if (mask == finalState) {
                return dist;
            }
            for (int v : graph[u]) {
                int newMask = mask | (1 << v);
                String state = v + " " + newMask;
                if (!visited.contains(state)) {
                    queue.offer(new int[]{v, newMask, dist + 1});
                    visited.add(state);
                }
            }
        }
        return -1;
    }


    public int countCompleteComponents(int n, int[][] edges) {
        HashMap<Integer, List<Integer>> graph = new HashMap<>();
        for (int i = 0; i < n; i++) {
            graph.put(i, new ArrayList<>());
        }
        for (int[] edge : edges) {
            int a = edge[0];
            int b = edge[1];
            graph.get(a).add(b);
            graph.get(b).add(a);
        }
        Set<Integer> visited = new HashSet<>();
        int completeComponentsCount = 0;
        for (int i = 0; i < n; i++) {
            if (!visited.contains(i)) {
                List<Integer> component = new ArrayList<>();
                dfs(graph, i, visited, component);
                if (isComplete(component, graph)) {
                    completeComponentsCount++;
                }
            }
        }
        return completeComponentsCount;
    }

    private void dfs(HashMap<Integer, List<Integer>> graph, int node, Set<Integer> visited, List<Integer> component) {
        visited.add(node);
        component.add(node);
        for (int neighbor : graph.get(node)) {
            if (!visited.contains(neighbor)) {
                dfs(graph, neighbor, visited, component);
            }
        }
    }

    private boolean isComplete(List<Integer> component, HashMap<Integer, List<Integer>> graph) {
        int size = component.size();
        for (int node : component) {
            if (graph.get(node).size() != size - 1) {
                return false;
            }
        }
        return true;
    }


    public int[] sortArray(int[] nums) {
        int n = nums.length;
        List<Integer> list = new ArrayList<>();
        int counter = 0;
        for (int i : nums) {
            int index = Collections.binarySearch(list, 0);
            System.out.println(index);
            if (index < 0) index = -(index - 1);
            list.add(index, i);
        }
        return list.stream().mapToInt(i -> i).toArray();
    }


    public static int minSessions(int[] tasks, int sessionTime) {
        Arrays.sort(tasks);
        int left = 1, right = tasks.length;
        while (left < right) {
            int mid = (left + right) / 2;
            if (canScheduleTasks(tasks, sessionTime, mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    private static boolean canScheduleTasks(int[] tasks, int sessionTime, int numSessions) {
        int[] sessions = new int[numSessions];
        return backtrack(tasks, sessions, tasks.length - 1, sessionTime);
    }

    private static boolean backtrack(int[] tasks, int[] sessions, int index, int sessionTime) {
        if (index < 0) {
            return true;
        }
        int currentTask = tasks[index];
        for (int i = 0; i < sessions.length; i++) {
            if (sessions[i] + currentTask <= sessionTime) {
                sessions[i] += currentTask;
                if (backtrack(tasks, sessions, index - 1, sessionTime)) {
                    return true;
                }
                sessions[i] -= currentTask;
            }
            if (sessions[i] == 0) {
                break;
            }
        }

        return false;
    }


    public static int[] successfulPairs(int[] spells, int[] potions, long success) {
        int n = potions.length;
        HashMap<Long, Integer> hm = new HashMap<>();
        Set<Long> spellSet = new HashSet<>();
        for (int i : spells) spellSet.add((long) i);
        List<Long> spellList = new ArrayList<>(spellSet);
        Collections.sort(spellList);
        Arrays.sort(potions);
        int potionIndex = n - 1;
        int potionCounter = 0;
        for (Long spell : spellList) {
            while (potionIndex != -1) {
                long currentPotion = potions[potionIndex];
                long currentSuccess = spell * currentPotion;
                if (currentSuccess >= success) {
                    potionCounter++;
                    potionIndex--;
                } else {
                    break;
                }
            }
            hm.put(spell, potionCounter);
        }
        for (int i = 0; i < spells.length; i++) {
            spells[i] = hm.get((long) spells[i]);
        }
        return spells;
    }


    public int minOperations(int[] nums, int x) {
        int totalSum = 0;
        for (int num : nums) {
            totalSum += num;
        }
        int target = totalSum - x;
        if (target < 0) return -1;
        int left = 0;
        int currentSum = 0;
        int maxLength = -1;

        for (int right = 0; right < nums.length; right++) {
            currentSum += nums[right];
            while (currentSum > target && left <= right) {
                currentSum -= nums[left];
                left++;
            }
            if (currentSum == target) {
                maxLength = Math.max(maxLength, right - left + 1);
            }
        }
        return maxLength == -1 ? -1 : nums.length - maxLength;
    }

    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        Map<Integer, Integer> sumCountMap = new HashMap<>();
        for (int num1 : nums1) {
            for (int num2 : nums2) {
                int sum = num1 + num2;
                sumCountMap.put(sum, sumCountMap.getOrDefault(sum, 0) + 1);
            }
        }
        int count = 0;
        for (int num3 : nums3) {
            for (int num4 : nums4) {
                int sum = num3 + num4;
                count += sumCountMap.getOrDefault(-sum, 0);
            }
        }

        return count;
    }

    public int[] findRightInterval(int[][] intervals) {
        int n = intervals.length;
        TreeMap<Integer, Integer> tm = new TreeMap<>();
        for (int i = 0; i < n; i++) {
            int[] interval = intervals[i];
            int start = interval[0];
            tm.put(start, i);
        }
        int[] output = new int[n];
        for (int i = 0; i < n; i++) {
            int[] interval = intervals[i];
            int end = interval[1];
            Integer key = tm.ceilingKey(end);
            output[i] = (key == null) ? -1 : tm.get(key);
        }
        return output;
    }


    public int findGCD(int[] nums) {
        Arrays.sort(nums);
        int a = nums[0];
        int b = nums[nums.length - 1];
        for (int i = a; i >= 1; i--) {
            if (a % i == 0 && b % i == 0) return i;
        }
        return -1;
    }


    class Bomb {
        int x;
        int y;
        int r;
        List<Bomb> bombsWithinRange;

        public Bomb(int x, int y, int r) {
            this.x = x;
            this.y = y;
            this.r = r;
            this.bombsWithinRange = new ArrayList<>();
        }
    }

    public int maximumDetonation(int[][] bombs) {
        int n = bombs.length;
        Bomb[] bombObjects = new Bomb[n];

        for (int i = 0; i < n; i++) {
            bombObjects[i] = new Bomb(bombs[i][0], bombs[i][1], bombs[i][2]);
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j && isInRange(bombObjects[i], bombObjects[j])) {
                    bombObjects[i].bombsWithinRange.add(bombObjects[j]);
                }
            }
        }

        int maxDetonated = 0;
        for (int i = 0; i < n; i++) {
            boolean[] visited = new boolean[n];
            maxDetonated = Math.max(maxDetonated, dfs(bombObjects, visited, i));
        }

        return maxDetonated;
    }

    private boolean isInRange(Bomb a, Bomb b) {
        long dx = a.x - b.x;
        long dy = a.y - b.y;
        long distanceSquared = dx * dx + dy * dy;
        long radiusSquared = (long) a.r * a.r;
        return distanceSquared <= radiusSquared;
    }

    private int dfs(Bomb[] bombObjects, boolean[] visited, int index) {
        visited[index] = true;
        int count = 1;

        for (Bomb b : bombObjects[index].bombsWithinRange) {
            int nextIndex = Arrays.asList(bombObjects).indexOf(b);
            if (!visited[nextIndex]) {
                count += dfs(bombObjects, visited, nextIndex);
            }
        }

        return count;
    }


    public List<Integer> twoOutOfThree(int[] nums1, int[] nums2, int[] nums3) {
        Set<Integer> set1 = new HashSet<>(Arrays.stream(nums1).boxed().toList());
        Set<Integer> set2 = new HashSet<>(Arrays.stream(nums2).boxed().toList());
        Set<Integer> set3 = new HashSet<>(Arrays.stream(nums3).boxed().toList());
        List<Integer> output = new ArrayList<>();
        int[] frequencyArray = new int[101];
        for (Integer I : set1) frequencyArray[I]++;
        for (Integer I : set2) frequencyArray[I]++;
        for (Integer I : set3) frequencyArray[I]++;
        for (int i = 1; i < 101; i++) {
            if (frequencyArray[i] >= 2) output.add(i);

        }
        return output;
    }


    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) return false;
        char[] sChar = s.toCharArray();
        char[] tChar = t.toCharArray();
        Arrays.sort(sChar);
        Arrays.sort(tChar);
        for (int i = 0; i < s.length(); i++) {
            if (sChar[i] != tChar[i]) return false;
        }
        return true;
    }


    public int numberOfChild(int n, int k) {
        int effectiveTime = k % (2 * (n - 1));
        if (effectiveTime < n) {
            return effectiveTime;
        } else {
            return 2 * (n - 1) - effectiveTime;
        }
    }


    public static String abbreviation(String a, String b) {
        int yMax = a.length();
        int xMax = b.length();
        char[] yCharArray = a.toCharArray();
        char[] xCharArray = b.toCharArray();
        boolean[][] dpMatrix = new boolean[yMax + 1][xMax + 1];
        dpMatrix[0][0] = true;
        for (int y = 1; y < yMax; y++) {
            if (Character.isLowerCase(yCharArray[y - 1])) {
                dpMatrix[y][0] = true;
            } else {
                break;
            }
        }
        for (int y = 1; y <= yMax; y++) {
            for (int x = 1; x <= xMax; x++) {
                char yChar = yCharArray[y - 1];
                char xChar = xCharArray[x - 1];

                if (Character.toUpperCase(yChar) == xChar) {
                    dpMatrix[y][x] = dpMatrix[y - 1][x - 1] || (Character.isLowerCase(yChar) && dpMatrix[y - 1][x]);
                } else if (Character.isLowerCase(yChar)) {
                    dpMatrix[y][x] = dpMatrix[y - 1][x];
                }
            }
        }
        return dpMatrix[yMax][xMax] ? "YES" : "NO";
    }


    public int findPoisonedDuration(int[] timeSeries, int duration) {
        int n = timeSeries.length;
        if (n == 0) return 0;
        int totalPoisonedDuration = 0;
        for (int i = 1; i < n; i++) {
            int gap = timeSeries[i] - timeSeries[i - 1];
            totalPoisonedDuration += Math.min(gap, duration);
        }
        totalPoisonedDuration += duration;
        return totalPoisonedDuration;
    }

    public long minimumFuelCost(int[][] roads, int seats) {
        int n = roads.length + 1;
        List<Integer>[] graph = new ArrayList[n];
        for (int i = 0; i < n; i++) {
            graph[i] = new ArrayList<>();
        }
        for (int[] road : roads) {
            int u = road[0], v = road[1];
            graph[u].add(v);
            graph[v].add(u);
        }
        long[] totalFuelCost = new long[1];
        dfs(graph, 0, -1, seats, totalFuelCost);
        return totalFuelCost[0];
    }

    private int dfs(List<Integer>[] graph, int current, int parent, int seats, long[] totalFuelCost) {
        int representatives = 1;
        for (int neighbor : graph[current]) {
            if (neighbor != parent) {
                representatives += dfs(graph, neighbor, current, seats, totalFuelCost);
            }
        }
        if (current != 0) {
            int trips = (representatives + seats - 1) / seats;
            totalFuelCost[0] += trips;
        }
        return representatives;
    }


    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode a = headA;
        while (a != null) {
            ListNode b = headB;
            while (b != null) {
                if (a == b) return a;
                b = b.next;
            }
            a = a.next;
        }
        return null;
    }


    public static int calculateMinimumHP(int[][] dungeon) {
        int yMax = dungeon.length;
        int xMax = dungeon[0].length;
        Integer[][] healthRequirement = new Integer[yMax][xMax];

        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt((int[] x) -> x[2]));
        pq.add(new int[]{yMax - 1, xMax - 1, 1});

        while (!pq.isEmpty()) {
            int[] currentPosition = pq.poll();
            int y = currentPosition[0];
            int x = currentPosition[1];
            int health = currentPosition[2];
            if (healthRequirement[y][x] != null && healthRequirement[y][x] <= health) continue;
            healthRequirement[y][x] = health;
            int nextHealth = Math.max(1, health - dungeon[y][x]);
            if (y != 0) pq.offer(new int[]{y - 1, x, nextHealth});
            if (x != 0) pq.offer(new int[]{y, x - 1, nextHealth});
        }

        // Calculate the minimum initial health required at the start (top-left)
        int initialHealth = healthRequirement[0][0] == 0 ? 1 : healthRequirement[0][0];
        if (dungeon[0][0] < 0) {
            initialHealth = Math.max(1, initialHealth - dungeon[0][0]);
        } else {
            initialHealth = Math.max(1, initialHealth - dungeon[0][0]);
        }

        return initialHealth;
    }


    public static void plusMinus(List<Integer> arr) {
        Collections.sort(arr);
        double pos = 0, neg = 0;
        double n = arr.size();
        for (int i : arr) {
            if (i < 0) neg++;
            if (i > 0) pos++;
        }
        System.out.println(pos / n);
        System.out.println(neg / n);
        System.out.println((n - pos - neg) / n);
    }


    public static int equalStacks(List<Integer> h1, List<Integer> h2, List<Integer> h3) {
        Deque<Integer> q1 = new ArrayDeque<>();
        Deque<Integer> q2 = new ArrayDeque<>();
        Deque<Integer> q3 = new ArrayDeque<>();
        long a = 0, b = 0, c = 0;
        for (int i : h1) {
            q1.addLast(i);
            a += i;
        }
        for (int i : h2) {
            q2.addLast(i);
            b += i;
        }
        for (int i : h3) {
            q3.addLast(i);
            c += i;
        }
        while (!(a == b && b == c)) {
            if (a >= b && a >= c) {
                a -= q1.pollFirst();
            } else if (b >= a && b >= c) {
                b -= q2.pollFirst();
            } else {
                c -= q3.pollFirst();
            }
        }

        return (int) a;
    }


    public boolean isItPossible(String word1, String word2) {
        int[] aFrequency = new int[26];
        int[] bFrequency = new int[26];
        for (char c : word1.toCharArray()) aFrequency[c - 'a']++;
        for (char c : word2.toCharArray()) bFrequency[c - 'a']++;
        int aUniqueCount = 0;
        int bUniqueCount = 0;
        for (int i = 0; i < 26; i++) {
            if (aFrequency[i] > 0) aUniqueCount++;
            if (bFrequency[i] > 0) bUniqueCount++;
        }
        for (int i = 0; i < 26; i++) {
            for (int j = 0; j < 26; j++) {
                if (aFrequency[i] > 0 && bFrequency[j] > 0) {
                    int newAUniqueCount = aUniqueCount;
                    int newBUniqueCount = bUniqueCount;
                    if (i != j) {
                        if (aFrequency[i] == 1) newAUniqueCount--;
                        if (aFrequency[j] == 0) newAUniqueCount++;
                        if (bFrequency[j] == 1) newBUniqueCount--;
                        if (bFrequency[i] == 0) newBUniqueCount++;
                    }
                    if (newAUniqueCount == newBUniqueCount) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    public int appendCharacters(String s, String t) {
        int output = 0;
        char[] sCharArray = s.toCharArray();
        char[] tCharArray = t.toCharArray();
        int tMax = t.length();
        int sMax = s.length();
        int tIndex = 0;
        int sIndex = 0;
        while (tIndex < tMax && sIndex < sMax) {
            char sChar = sCharArray[sIndex];
            char tChar = tCharArray[tIndex];
            if (sChar == tChar) {
                tIndex++;
            }
            sIndex++;
        }
        return (tMax - tIndex);
    }


    public long repairCars(int[] ranks, int cars) {
        long left = 1;
        long right = (long) ranks[0] * (long) cars * (long) cars;
        for (int rank : ranks) {
            right = Math.max(right, (long) rank * (long) cars * (long) cars);
        }
        while (left < right) {
            long mid = left + (right - left) / 2;
            if (canRepairInTime(mid, ranks, cars)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    boolean canRepairInTime(long time, int[] ranks, int cars) {
        long totalCarsRepaired = 0;
        for (int rank : ranks) {
            long carsByMechanic = (long) Math.sqrt(time / rank);
            totalCarsRepaired += carsByMechanic;
            if (totalCarsRepaired >= cars) {
                return true;
            }
        }
        return totalCarsRepaired >= cars;
    }


    public int closestMeetingNode(int[] edges, int node1, int node2) {
        HashMap<Integer, Integer> hmA = new HashMap<>();
        //BFS
        int currentNode = node1;
        int currentDistance = 0;
        while (currentNode != -1) {
            if (hmA.containsKey(currentNode)) break;
            hmA.put(currentNode, currentDistance);
            currentNode = edges[currentNode];
            currentDistance++;
        }
        int output = Integer.MAX_VALUE;
        int minDistance = Integer.MAX_VALUE;
        currentNode = node2;
        currentDistance = 0;
        HashSet<Integer> visited = new HashSet<>();
        while (currentNode != -1) {
            if (visited.contains(currentNode)) break;
            visited.add(currentNode);
            if (hmA.containsKey(currentNode)) {
                int distance = Math.max(currentDistance, hmA.get(currentNode));
                if (distance < minDistance) {
                    minDistance = distance;
                    output = currentNode;
                } else if (distance == minDistance && currentNode < output) {
                    output = currentNode;
                }
            }
            currentNode = edges[currentNode];
            currentDistance++;
        }
        return output;
    }

    public int minScore(int n, int[][] roads) {
        Map<Integer, List<int[]>> graph = new HashMap<>();
        for (int i = 1; i <= n; i++) {
            graph.put(i, new ArrayList<>());
        }
        for (int[] road : roads) {
            int a = road[0];
            int b = road[1];
            int distance = road[2];
            graph.get(a).add(new int[]{b, distance});
            graph.get(b).add(new int[]{a, distance});
        }
        Queue<Integer> queue = new LinkedList<>();
        boolean[] visited = new boolean[n + 1];
        queue.add(1);
        visited[1] = true;
        int minScore = Integer.MAX_VALUE;
        while (!queue.isEmpty()) {
            int currentNode = queue.poll();
            for (int[] neighbor : graph.get(currentNode)) {
                int nextNode = neighbor[0];
                int distance = neighbor[1];
                minScore = Math.min(minScore, distance);
                if (!visited[nextNode]) {
                    visited[nextNode] = true;
                    queue.add(nextNode);
                }
            }
        }
        return minScore;
    }


    public int lengthOfLastWord(String s) {
        s = s.trim();
        String[] sArray = s.split(" ");
        int n = sArray.length;
        String last = sArray[n - 1];
        return last.length();
    }


    public String rankTeams(String[] votes) {
        int yMax = votes.length;
        if (yMax == 1) return votes[0];
        int xMax = votes[0].length();
        HashMap<Character, int[]> hm = new HashMap<>();
        for (String s : votes) {
            char[] sCharArray = s.toCharArray();
            for (int i = 0; i < xMax; i++) {
                char c = sCharArray[i];
                if (!hm.containsKey(c)) hm.put(c, new int[xMax]);
                hm.get(c)[i]++;
            }
        }
        List<Character> keyList = new ArrayList<>(hm.keySet());
        Collections.sort(keyList, (a, b) -> rankCompare(a, b, hm.get(a), hm.get(b), 0));
        StringBuilder sb = new StringBuilder();
        for (Character c : keyList) sb.append(c);
        return sb.toString();
    }

    public int rankCompare(char a, char b, int[] aRanks, int[] bRanks, int index) {
        int maxIndex = aRanks.length;
        if (index == maxIndex) {
            return (Character.compare(a, b));
        }
        int aRank = aRanks[index];
        int bRank = bRanks[index];
        if (aRank == bRank) return rankCompare(a, b, aRanks, bRanks, index + 1);
        else return (bRank - aRank);
    }


    public static boolean find132pattern(int[] nums) {
        int n = nums.length;
        if (n <= 2) return false;
        Deque<Integer> dq = new ArrayDeque<>();
        int firstRequirement = Integer.MIN_VALUE;
        for (int i = n - 1; i >= 0; i--) {
            int num = nums[i];
            if (num < firstRequirement) return true;
            while (!dq.isEmpty() && dq.peek() < num) {
                firstRequirement = dq.pollFirst();
            }
            dq.addFirst(num);
        }
        return false;
    }


    public static int hackerlandRadioTransmitters(List<Integer> x, int k) {
        Set<Integer> set = new HashSet<>(x);
        x = new ArrayList<>(set);
        Collections.sort(x);
        int output = 0;
        int currentRange = -1;
        for (int i : x) {
            if (currentRange < i) {
                for (int j = i + k; j >= i; j--) {
                    if (set.contains(j)) {
                        currentRange = j + k;
                        output++;
                        break;
                    }

                }
            }
        }
        return output;
    }


    public static long arrayManipulation(int n, List<List<Integer>> queries) {
        HashMap<Integer, Integer> hm = new HashMap<>();
        for (var query : queries) {
            int start = query.get(0);
            int end = query.get(1);
            int value = query.get(2);
            hm.merge(start, value, Integer::sum);
            hm.merge(end + 1, -value, Integer::sum);
        }
        long sum = 0;
        long output = Long.MIN_VALUE;
        List<Integer> keyList = new ArrayList<>(hm.keySet());
        Collections.sort(keyList);
        for (Integer key : keyList) {
            sum += hm.get(key);
            output = Math.max(output, sum);
        }
        return output;
    }

//    public static void main(String[] args) {
//        Scanner scanner = new Scanner(System.in);
//        int n = scanner.nextInt();
//        int[] values = new int[n];
//        for (int i = 0; i < n; i++) {
//            values[i] = scanner.nextInt();
//        }
//        int min = scanner.nextInt();
//        int max = scanner.nextInt();
//        for (int value : values) {
//            if ((min <= value && max >= value) || (min >= value && max <= value)) {
//                System.out.println(value);
//                break;
//            }
//        }
//    }


    public int[][] merge(int[][] intervals) {
        int n = intervals.length;
        if (n == 1) return intervals;
        Arrays.sort(intervals, Comparator.comparingInt((int[] a) -> a[0]));
        List<int[]> list = new ArrayList<>();
        int currentStart = intervals[0][0];
        int currentEnd = intervals[0][1];
        for (int i = 1; i <= n; i++) {
            if (i == n) {
                list.add(new int[]{currentStart, currentEnd});
            } else {
                int start = intervals[i][0];
                int end = intervals[i][1];
                if (currentEnd < start) {
                    list.add(new int[]{currentStart, currentEnd});
                    currentStart = start;
                    currentEnd = end;
                } else {
                    currentEnd = Math.max(currentEnd, end);
                }
            }
        }
        int m = list.size();
        int[][] output = new int[m][];
        for (int i = 0; i < m; i++) {
            output[i] = list.get(i);
        }
        return output;
    }

    int[] redundantConnection = new int[2];

    public int[] findRedundantConnection(int[][] edges) {
        int n = edges.length;
        boolean[][] graph = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            int[] edge = edges[i];
            int from = edge[0] - 1;
            int to = edge[1] - 1;
            graph[from][to] = true;
            graph[to][from] = true;
        }
        findRedundantConnection(new HashSet<>(), graph, 0, -1);
        return redundantConnection;
    }

    public void findRedundantConnection(HashSet<Integer> visited, boolean[][] graph, int currentNode, int fromNode) {
        visited.add(currentNode);
        boolean[] to = graph[currentNode];
        for (int i = 0; i < to.length; i++) {
            if (i == currentNode || i == fromNode) continue;
            if (to[i] && visited.contains(i)) {
                if (i != fromNode) {
                    redundantConnection = new int[]{i, currentNode};
                } else {
                    findRedundantConnection(visited, graph, i, currentNode);
                }
            }
        }
    }


    public static String removeKdigits(String num, int k) {
        Deque<Character> dq = new ArrayDeque<>();
        for (Character c : num.toCharArray()) {
            while (!dq.isEmpty() && dq.peekLast() > c && k > 0) {
                k--;
                dq.pollLast();
            }
            dq.addLast(c);
        }
        while (k > 0 && dq.size() > 0) {
            dq.pollLast();
            k--;
        }
        if (dq.size() == 0) return "0";
        StringBuilder sb = new StringBuilder();
        for (Character digit : dq) {
            sb.append(digit);
        }
        while (sb.length() > 1 && sb.charAt(0) == '0') {
            sb.deleteCharAt(0);
        }
        return sb.toString();
    }

    public List<Interval> employeeFreeTime(List<List<Interval>> schedule) {
        HashMap<Integer, Integer> hm = new HashMap<>();
        for (List<Interval> List : schedule) {
            for (Interval I : List) {
                Integer start = I.start;
                Integer end = I.end;
                hm.merge(start, 1, Integer::sum);
                hm.merge(end, -1, Integer::sum);
            }
        }
        List<Interval> output = new ArrayList<>();
        List<Integer> keyList = new ArrayList<>(hm.keySet());
        Collections.sort(keyList);
        int n = keyList.size();
        Integer sum = hm.get(keyList.get(0));
        Integer currentIntervalStart = -1;
        for (int i = 1; i < n; i++) {
            Integer key = keyList.get(i);
            Integer value = hm.get(key);
            if (sum == 0) {
                output.add(new Interval(currentIntervalStart, value));
            }
            sum += value;
            if (sum == 0) currentIntervalStart = key;
        }
        return output;
    }


    public static int minEatingSpeed(int[] piles, int h) {
        long max = Arrays.stream(piles).max().getAsInt();
        long low = 1;
        long high = max;
        long output = max;
        while (low <= high) {
            long mid = low + (high - low) / 2;
            long hours = 0;
            for (int pile : piles) {
                hours += (pile + mid - 1) / mid;
            }
            if (hours <= h) {
                output = mid;
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return (int) output;
    }


    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1.length > nums2.length) {
            return findMedianSortedArrays(nums2, nums1);
        }
        int m = nums1.length;
        int n = nums2.length;
        int low = 0, high = m;
        while (low <= high) {
            int i = (low + high) / 2;
            int j = (m + n + 1) / 2 - i;
            int nums1LeftMax = (i == 0) ? Integer.MIN_VALUE : nums1[i - 1];
            int nums1RightMin = (i == m) ? Integer.MAX_VALUE : nums1[i];
            int nums2LeftMax = (j == 0) ? Integer.MIN_VALUE : nums2[j - 1];
            int nums2RightMin = (j == n) ? Integer.MAX_VALUE : nums2[j];
            if (nums1LeftMax <= nums2RightMin && nums2LeftMax <= nums1RightMin) {
                if ((m + n) % 2 == 0) {
                    return ((double) Math.max(nums1LeftMax, nums2LeftMax) + Math.min(nums1RightMin, nums2RightMin)) / 2;
                } else {
                    return Math.max(nums1LeftMax, nums2LeftMax);
                }
            } else if (nums1LeftMax > nums2RightMin) {
                high = i - 1;
            } else {
                low = i + 1;
            }
        }
        return -1d;
    }


    public int visibleMountains(int[][] peaks) {
        Arrays.sort(peaks, Comparator.comparingInt((int[] a) -> a[0]).thenComparingInt(a -> a[1]));
        Deque<int[]> dq = new ArrayDeque<>();
        for (int[] peak : peaks) {
            boolean isVisible = true;
            int pos = peak[0];
            int height = peak[1];
            while (!dq.isEmpty()) {
                int[] last = dq.peekLast();
                int lastPos = last[0];
                int lastHeight = last[1];
                int posDifference = pos - lastPos;
                if (height - lastHeight >= posDifference) {
                    dq.pollLast();
                } else if (lastHeight - height >= posDifference) {
                    isVisible = false;
                    break;
                } else {
                    break;
                }
            }
            if (isVisible) {
                dq.addLast(peak);
            }
        }
        HashMap<String, Integer> frequencyMap = new HashMap<>();
        for (int[] peak : peaks) {
            String key = Arrays.toString(peak);
            frequencyMap.merge(key, 1, Integer::sum);
        }
        List<int[]> uniqueVisiblePeaks = new ArrayList<>();
        for (int[] peak : dq) {
            String key = Arrays.toString(peak);
            if (frequencyMap.get(key) == 1) {
                uniqueVisiblePeaks.add(peak);
            }
        }
        return uniqueVisiblePeaks.size();
    }

    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Deque<Integer> dq = new ArrayDeque<>();
        Deque<Integer> poppedDQ = new ArrayDeque<>(Arrays.stream(popped).boxed().toList());
        for (int i : pushed) {
            dq.addLast(i);
            while (!dq.isEmpty() && !poppedDQ.isEmpty() && dq.peekLast().equals(poppedDQ.peekFirst())) {
                dq.pollLast();
                poppedDQ.pollFirst();
            }
        }
        return (dq.isEmpty() && poppedDQ.isEmpty());
    }


    public boolean isThereAPath(int[][] grid) {
        int yMax = grid.length;
        int xMax = grid[0].length;
        int maxDifference = yMax + xMax - 1;
        int rangeDifference = (maxDifference * 2) + 1;
        int zeroDifferenceIndex = maxDifference;
        boolean[][][] dp = new boolean[yMax + 1][xMax + 1][rangeDifference];
        dp[yMax][xMax - 1][zeroDifferenceIndex] = true;
        dp[yMax + 1][xMax][zeroDifferenceIndex] = true;
        for (int y = yMax - 1; y >= 0; y--) {
            for (int x = xMax - 1; x >= 0; x--) {
                int i = grid[y][x];
                if (i == 0) i = -1;
                boolean[] right = dp[y][x + 1];
                boolean[] down = dp[y + 1][x];
                for (int j = 0; j < rangeDifference; j++) {
                    boolean d = down[j];
                    if (d) dp[y][x][j + i] = true;
                    boolean u = right[j];
                    if (u) dp[y][x][j + i] = true;
                }
            }
        }
        return dp[0][0][zeroDifferenceIndex];
    }

    public boolean pathDFS(int[][] grid, int y, int x, int count, int yMax, int xMax) {
        if (grid[y][x] == 1) count++;
        else count--;
        if (y == yMax && x == xMax) return (count == 0);
        boolean a = (y != yMax) && pathDFS(grid, y + 1, x, count, yMax, xMax);
        boolean b = (x != xMax) && pathDFS(grid, y, x + 1, count, yMax, xMax);
        return (a || b);
    }

    class Logger {
        Set<String> set;
        Integer currentTime;
        Deque<Message> messageQueue;

        public Logger() {
            set = new HashSet<>();
            currentTime = 0;
            messageQueue = new ArrayDeque<>();
        }

        public boolean shouldPrintMessage(int timestamp, String message) {
            Boolean output;
            Integer expiryTime = timestamp - 10;
            while (!messageQueue.isEmpty() && messageQueue.peekFirst().timeStamp <= expiryTime) {
                set.remove(messageQueue.pollFirst().message);
            }
            output = !set.contains(message);
            if (output) {
                messageQueue.addLast(new Message(message, timestamp));
                set.add(message);
            }
            return output;
        }

        class Message {
            public String message;
            public Integer timeStamp;

            public Message(String message, Integer timeStamp) {
                this.message = message;
                this.timeStamp = timeStamp;
            }
        }
    }

    public static int[] computeZArray(String s) {
        int n = s.length();
        int[] Z = new int[n];
        int L = 0, R = 0, K;
        for (int i = 1; i < n; ++i) {
            System.out.println(s.charAt(i));
            if (i > R) { //If we are outside the Z box
                L = R = i;
                while (R < n && s.charAt(R) == s.charAt(R - L)) {
                    R++;
                }
                Z[i] = R - L;
                R--;
            } else { //We are inside the Z box
                K = i - L; //K is the index of the matched character, wrt the pattern character
                if (Z[K] < R - i + 1) { //It is safe to copy the pre-computed Z value
                    Z[i] = Z[K]; //Copy the pre-computed Z value from the respective K index in the pattern String
                } else { //!! There is a match within the Z box (A match within a match)
                    L = i;
                    while (R < n && s.charAt(R) == s.charAt(R - L)) {
                        R++;
                    }
                    Z[i] = R - L;
                    R--;
                }
            }
        }
        return Z;
    }


    public List<String> findRepeatedDnaSequences(String s) {
        int n = s.length();
        List<String> output = new ArrayList<>();
        if (n < 10) return output;
        Set<String> set = new HashSet<>();
        Set<String> repeatedSet = new HashSet<>();
        for (int i = 0; i < n - 9; i++) {
            String sub = s.substring(i, i + 10);
            if (set.contains(sub)) repeatedSet.add(sub);
            set.add(sub);
        }
        for (String ss : repeatedSet) {
            output.add(ss);
        }
        return output;
    }


    public int tribonacci(int n) {
        if (n == 0) return 0;
        if (n == 1 || n == 2) return 1;

        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        dp[2] = 1;

        for (int index = 3; index <= n; index++) {
            dp[index] = dp[index - 1] + dp[index - 2] + dp[index - 3];
        }

        return dp[n];
    }


    public int maximumScore(int[] nums, int[] multipliers) {
        int n = nums.length;
        int m = multipliers.length;
        int[][] dp = new int[m + 1][m + 1];
        for (int i = m - 1; i >= 0; i--) {
            for (int left = i; left >= 0; left--) {
                int mult = multipliers[i];
                int right = n - 1 - (i - left);
                dp[i][left] = Math.max(
                        nums[left] * mult + dp[i + 1][left + 1],
                        nums[right] * mult + dp[i + 1][left]
                );
            }
        }
        return dp[0][0];
    }

    public int minAreaRect(int[][] points) {
        Set<String> set = new HashSet<>();
        for (int[] point : points) {
            set.add(point[0] + "," + point[1]);
        }
        int minArea = Integer.MAX_VALUE;
        for (int i = 0; i < points.length; i++) {
            for (int j = i + 1; j < points.length; j++) {
                int x1 = points[i][0], y1 = points[i][1];
                int x2 = points[j][0], y2 = points[j][1];

                // Check if the points can form a diagonal of a rectangle
                if (x1 != x2 && y1 != y2) {
                    if (set.contains(x1 + "," + y2) && set.contains(x2 + "," + y1)) {
                        int area = Math.abs((x2 - x1) * (y2 - y1));
                        minArea = Math.min(minArea, area);
                    }
                }
            }
        }
        return (minArea == Integer.MAX_VALUE) ? 0 : minArea;
    }

    List<TreeNode> tnList = new ArrayList<>();

    public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
        tnList = new ArrayList<>();
        Set<Integer> set = new HashSet<>(Arrays.stream(to_delete).boxed().toList());

        delNodesBFS(root, set);
        return tnList;
    }

    public void delNodesBFS(TreeNode root, Set<Integer> set) {
        if (root == null) return;
        int val = root.val;
        if (set.contains(val)) {
            delNodesBFS(root.left, set);
            delNodesBFS(root.right, set);
        } else {
            root.left = delNodesBFS2(root.left, set);
            root.right = delNodesBFS2(root.right, set);
            tnList.add(root);
        }
    }

    public TreeNode delNodesBFS2(TreeNode root, Set<Integer> set) {
        if (root == null) return null;
        int val = root.val;
        if (set.contains(val)) {
            delNodesBFS(root.left, set);
            delNodesBFS(root.right, set);
            return null;
        } else {
            root.left = delNodesBFS2(root.left, set);
            root.right = delNodesBFS2(root.right, set);
            return root;
        }
    }


    class StockPrice {
        PriorityQueue<int[]> minQueue = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        PriorityQueue<int[]> maxQueue = new PriorityQueue<>(Comparator.comparingInt(a -> -a[0]));
        HashMap<Integer, Integer> hm = new HashMap<>();
        int currentTime;
        int currentPrice;

        public StockPrice() {
            currentTime = 0;
            currentPrice = 0;
        }

        public void update(int timestamp, int price) {
            int[] record = new int[]{price, timestamp};
            if (currentTime <= timestamp) {
                currentTime = timestamp;
                currentPrice = price;
            }
            hm.put(timestamp, price);
            minQueue.offer(record);
            maxQueue.offer(record);
        }

        public int current() {
            return currentPrice;
        }

        public int maximum() {
            int output = maxQueue.peek()[0];
            while (output != hm.get(maxQueue.peek()[1])) {
                maxQueue.poll();
                output = maxQueue.peek()[0];
            }
            return output;
        }

        public int minimum() {
            int output = minQueue.peek()[0];
            while (output != hm.get(minQueue.peek()[1])) {
                minQueue.poll();
                output = minQueue.peek()[0];
            }
            return output;
        }
    }


    public int[][] highFive(int[][] items) {
        Arrays.sort(items, Comparator.comparing((int[] a) -> a[0]).thenComparingInt(a -> -a[1]));
        List<int[]> list = new ArrayList<>();
        int priorStudent = -1;
        int index = 0;
        int n = items.length;
        while (index < n) {
            while (items[index][0] == priorStudent) {
                index++;
            }
            priorStudent = items[index][0];
            int currentScore = 0;
            for (int i = 0; i < 5; i++) {
                currentScore += items[index][1];
                index++;
            }
            currentScore = currentScore / 5;
            list.add(new int[]{priorStudent, currentScore});
        }
        int[][] output = new int[list.size()][];
        for (int i = 0; i < list.size(); i++) {
            output[i] = list.get(i);
        }
        return output;
    }


    public static int minKnightMoves(int x, int y) {
        // Normalize target to first quadrant to reduce search space
        x = Math.abs(x);
        y = Math.abs(y);

        // Define moves of a knight
        int[][] directions = {
                {2, 1}, {1, 2}, {-1, 2}, {-2, 1},
                {-2, -1}, {-1, -2}, {1, -2}, {2, -1}
        };

        // Priority queue with heuristic comparator (Manhattan distance)
        int finalX = x;
        int finalY = y;
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) ->
                Integer.compare(a[0] + Math.abs(a[1] - finalX) + Math.abs(a[2] - finalY),
                        b[0] + Math.abs(b[1] - finalX) + Math.abs(b[2] - finalY))
        );
        Set<String> visited = new HashSet<>();
        pq.offer(new int[]{0, 0, 0});
        while (!pq.isEmpty()) {
            int[] current = pq.poll();
            int moves = current[0];
            int currentX = current[1];
            int currentY = current[2];
            String position = currentX + "," + currentY;
            if (visited.contains(position)) continue;
            visited.add(position);
            if (currentX == x && currentY == y) return moves;
            for (int[] direction : directions) {
                int nextX = currentX + direction[0];
                int nextY = currentY + direction[1];
                pq.offer(new int[]{moves + 1, nextX, nextY});
            }
        }

        return -1; // Should never reach here for a valid chessboard
    }


    public long minSum(int[] nums1, int[] nums2) {
        long aSum = 0;
        long aCount = 0;
        long bSum = 0;
        long bCount = 0;
        for (int i : nums1) {
            aSum += i;
            if (i == 0) aCount++;
        }
        for (int i : nums2) {
            bSum += i;
            if (i == 0) bCount++;
        }
        if (aCount == 0 && bCount == 0) {
            return (aSum == bSum) ? aSum : -1;
        }
        long aMin = aSum + aCount;
        long bMin = bSum = bCount;
        if (aMin == bMin) return aMin;
        if (aCount == 0 && aMin < bMin) return -1;
        if (bCount == 0 && bMin < aMin) return -1;
        return Math.min(aMin, bMin);
    }

    public int maxNonDecreasingLength(int[] nums1, int[] nums2) {
        int aLength = 1; // Length of non-decreasing subarray ending in nums1
        int bLength = 1; // Length of non-decreasing subarray ending in nums2
        int n = nums1.length;
        int output = 1;  // Maximum length of non-decreasing subarray found
        for (int i = 1; i < n; i++) {
            int newALength = 1; // Length if taking nums1[i]
            int newBLength = 1; // Length if taking nums2[i]
            if (nums1[i - 1] <= nums1[i]) {
                newALength = Math.max(newALength, aLength + 1);
            }
            if (nums2[i - 1] <= nums1[i]) {
                newALength = Math.max(newALength, bLength + 1);
            }
            if (nums1[i - 1] <= nums2[i]) {
                newBLength = Math.max(newBLength, aLength + 1);
            }
            if (nums2[i - 1] <= nums2[i]) {
                newBLength = Math.max(newBLength, bLength + 1);
            }
            aLength = newALength;
            bLength = newBLength;
            output = Math.max(output, Math.max(aLength, bLength));
        }
        return output;
    }


    public int lengthOfLongestSubstringTwoDistinct(String s) {
        char[] cArray = s.toCharArray();
        Deque<Character> dq = new ArrayDeque<>();
        HashMap<Character, Integer> hm = new HashMap<>();
        int output = 0;
        for (Character c : cArray) {
            hm.merge(c, 1, Integer::sum);
            dq.addLast(c);
            if (hm.keySet().size() > 2) {
                while (hm.keySet().size() > 2) {
                    Character removeC = dq.pollFirst();
                    hm.merge(removeC, -1, Integer::sum);
                    if (hm.get(removeC) == 0) hm.remove(removeC);
                }
            }
            output = Math.max(output, dq.size());
        }
        return output;
    }

    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        char[] cArray = s.toCharArray();
        Deque<Character> dq = new ArrayDeque<>();
        HashMap<Character, Integer> hm = new HashMap<>();
        int output = 0;
        for (Character c : cArray) {
            hm.merge(c, 1, Integer::sum);
            dq.addLast(c);
            if (hm.keySet().size() > k) {
                while (hm.keySet().size() > k) {
                    Character removeC = dq.pollFirst();
                    hm.merge(removeC, -1, Integer::sum);
                    if (hm.get(removeC) == 0) hm.remove(removeC);
                }
            }
            output = Math.max(output, dq.size());
        }
        return output;
    }

    public int findMaxConsecutiveOnes(int[] nums) {
        int priorCount = 0;
        int currentCount = 0;
        int output = 0;
        for (int i : nums) {
            if (i == 1) {
                priorCount++;
                currentCount++;
            } else {
                output = Math.max(output, priorCount);
                priorCount = ++currentCount;
                currentCount = 0;
            }
        }
        output = Math.max(output, Math.max(priorCount, currentCount));
        return output;
    }

    public static double probabilityOfHeads(double[] prob, int target) {
        int n = prob.length;
        double[][] dp = new double[n + 1][target + 1];
        dp[n][0] = 1;
        for (int y = n - 1; y >= 0; y--) {
            double currentProb = prob[y];
            for (int x = 0; x <= target; x++) {
                if (x == 0) {
                    dp[y][x] = (1 - currentProb) * dp[y + 1][0];
                } else {
                    dp[y][x] = currentProb * dp[y + 1][x - 1] + (1 - currentProb) * dp[y + 1][x];
                }
            }
        }
        return dp[0][target];
    }


    public int setMask(int number, int position) {
        return (number | (1 << position));
    }

    public boolean checkMask(int number, int position) {
        return (number & (1 << position)) != 0;
    }

    public int manhattanDistance(int[] p1, int[] p2) {
        return Math.abs(p1[0] - p2[0]) + Math.abs(p1[1] - p2[1]);
    }


    public int assignBikes(int[][] workers, int[][] bikes) {
        int numWorkers = workers.length;
        int numBikes = bikes.length;
        // Number of permutation of bike selections is 2^numBikes
        int bikePermutations = 1 << numBikes;
        Integer[][] dpMatrix = new Integer[numWorkers][bikePermutations];
        return assignBikes(workers, bikes, dpMatrix, 0, 0);
    }

    public int assignBikes(int[][] workers, int[][] bikes, Integer[][] dpMatrix, int workerIndex, int bikeMask) {
        //Base case
        if (workerIndex == workers.length) return 0;
        //Cache case
        if (dpMatrix[workerIndex][bikeMask] != null) return dpMatrix[workerIndex][bikeMask];
        //Bitmask
        int minDistances = Integer.MAX_VALUE;
        for (int i = 0; i < bikes.length; i++) {
            //Check if the current bike is available
            if (!checkMask(bikeMask, i)) {
                int distance = manhattanDistance(workers[workerIndex], bikes[i]);
                int nextMask = setMask(bikeMask, i);
                int remainingDistances = assignBikes(workers, bikes, dpMatrix, workerIndex + 1, nextMask);
                minDistances = Math.min(minDistances, distance + remainingDistances);
            }
        }
        return dpMatrix[workerIndex][bikeMask] = minDistances;
    }


    public int maxA(int n) {
        int[] dp = new int[n + 1];
        int[] ctrlA = new int[n + 1];
        int[] ctrlC = new int[n + 1];
        int[] ctrlV = new int[n + 1];
        for (int i = 1; i < n; i++) {
            ctrlA[i] = dp[i - 1];
            ctrlC[i] = ctrlA[i - 1];
            ctrlV[i] = ctrlC[i - 1];
            int currentCount = Math.max(dp[i - 1] + 1, dp[i - 1] + ctrlV[i]);
            dp[i] = currentCount;
        }
        return dp[n];
    }

    int maxRequest;

    public int maximumRequests(int n, int[][] requests) {
        maxRequest = Integer.MIN_VALUE;
        List<int[]> requestList = new ArrayList<>();
        int[] buildingBalance = new int[n];
        int nonTransfer = 0;
        for (int[] request : requests) {
            if (request[0] == request[1]) {
                nonTransfer++;
            } else {
                requestList.add(request);
            }
        }
        maximumRequests(requestList, 0, 0, buildingBalance);
        return maxRequest + nonTransfer;
    }

    public void maximumRequests(List<int[]> requestList, int requestIndex, int currentCount, int[] buildingBalance) {
        int n = requestList.size();
        if (requestIndex == n) {
            for (int i : buildingBalance) if (i != 0) return;
            maxRequest = Math.max(maxRequest, currentCount);
            return;
        }
        //Early exit: Remaining requests including this one
        int remainingRequests = n - requestIndex;
        if (currentCount + (n - requestIndex) < maxRequest) return;
        //Backtrack
        int[] currentRequest = requestList.get(requestIndex);
        buildingBalance[currentRequest[0]]--;
        buildingBalance[currentRequest[1]]++;
        maximumRequests(requestList, requestIndex + 1, currentCount + 1, buildingBalance);
        buildingBalance[currentRequest[0]]++;
        buildingBalance[currentRequest[1]]--;
        maximumRequests(requestList, requestIndex + 1, currentCount, buildingBalance);
    }

    public static Integer safeMax(Integer a, Integer b) {
        if (a == null && b == null) {
            return null;
        } else if (a == null) {
            return b;
        } else if (b == null) {
            return a;
        } else {
            return Math.max(a, b);
        }
    }

    public static int tallestBillboard(int[] rods) {
        int n = rods.length;
        int maxOffset = 2500;
        int[][] dp = new int[n + 1][2 * maxOffset + 1];
        for (int[] d : dp) Arrays.fill(d, -1);
        dp[n][maxOffset] = 0;
        for (int i = n - 1; i >= 0; i--) {
            int rod = rods[i];
            for (int diff = -maxOffset; diff <= maxOffset; diff++) {
                if (dp[i + 1][diff + maxOffset] == -1) continue;
                // Do not use the current rod
                if (dp[i][diff + maxOffset] == -1) dp[i][diff + maxOffset] = 0;
                dp[i][diff + maxOffset] = Math.max(dp[i][diff + maxOffset], dp[i + 1][diff + maxOffset]);
                // Add the rod to the left side
                if (diff + rod + maxOffset <= 2 * maxOffset) {
                    if (dp[i][diff + rod + maxOffset] == -1) dp[i][diff + rod + maxOffset] = 0;
                    dp[i][diff + rod + maxOffset] = Math.max(dp[i][diff + rod + maxOffset], dp[i + 1][diff + maxOffset]);
                }
                // Add the rod to the right side
                if (diff - rod + maxOffset >= 0) {
                    if (dp[i][diff - rod + maxOffset] == -1) dp[i][diff - rod + maxOffset] = 0;
                    dp[i][diff - rod + maxOffset] = Math.max(dp[i][diff - rod + maxOffset], dp[i + 1][diff + maxOffset] + rod);
                }
            }
        }
        return dp[0][maxOffset] == -1 ? 0 : dp[0][maxOffset];
    }


    public static int minimumDifference(int[] nums) {
        int n = nums.length;
        int totalSum = 0;
        for (int i : nums) totalSum += Math.abs(i);
        int halfSum = totalSum / 2;

        boolean[] dp = new boolean[halfSum + 1];
        dp[0] = true;

        for (int num : nums) {
            for (int j = halfSum; j >= num; j--) {
                dp[j] = dp[j] || dp[j - num];
            }
        }

        for (int i = halfSum; i >= 0; i--) {
            if (dp[i]) {
                int sum1 = i;
                int sum2 = totalSum - i;
                return Math.abs(sum2 - sum1);
            }
        }

        return 0;
    }


    public int minCost(int[][] costs) {
        int n = costs.length;
        int[][] dp = new int[n][3];
        dp[n - 1][0] = costs[n - 1][0];
        dp[n - 1][1] = costs[n - 1][1];
        dp[n - 1][2] = costs[n - 1][2];
        for (int i = n - 2; i >= 0; i--) {
            dp[i][0] = costs[i][0] + Math.min(dp[i + 1][1], dp[i + 1][2]);
            dp[i][1] = costs[i][1] + Math.min(dp[i + 1][0], dp[i + 1][2]);
            dp[i][2] = costs[i][2] + Math.min(dp[i + 1][0], dp[i + 1][1]);
        }
        int output = Math.min(dp[0][0], Math.min(dp[0][1], dp[0][2]));
        return output;
    }

    public int minCostII(int[][] costs) {
        int n = costs.length;
        int m = costs[0].length;
        for (int i = n - 2; i >= 0; i--) {
            for (int j = 0; j < m; j++) {
                int currentCost = costs[i][j];
                int minCost = Integer.MAX_VALUE;
                for (int k = 0; k < n; k++) {
                    if (j == k) continue;
                    minCost = Math.min(minCost, currentCost + costs[i + 1][k]);
                }
                costs[i][j] = minCost;
            }
        }
        int output = Integer.MAX_VALUE;
        for (int i : costs[0]) {
            output = Math.min(output, i);
        }
        return output;
    }


    public int countVowelPermutation(int n) {
        long[] dp = new long[]{1, 1, 1, 1, 1};
        long mod = (long) (1e9 + 7);
        for (int i = 1; i < n; i++) {
            long[] dpNext = new long[5];
            dpNext[0] = (dp[1] + dp[2] + dp[4]) % mod;
            dpNext[1] = (dp[0] + dp[2]) % mod;
            dpNext[2] = (dp[1] + dp[3]) % mod;
            dpNext[3] = (dp[2]) % mod;
            dpNext[4] = (dp[2] + dp[3]) % mod;
            dp = dpNext;
        }
        long output = 0;
        for (long l : dp) output += l;
        return (int) (output % mod);
    }

    public int numWays(int n, int k) {
        if (n == 0) return 0;
        if (n == 1) return k;
        if (k == 1) return (n <= 2) ? 1 : 0;
        int special = k;
        int normal = k * (k - 1);
        for (int i = 3; i <= n; i++) {
            int priorSpecial = special;
            special = normal;
            normal = (priorSpecial + normal) * (k - 1);
        }
        return normal + special;
    }


    public int numberOfWays(int numPeople) {
        int mod = (int) 1e9 + 7;
        int[] dp = new int[numPeople / 2 + 1];
        dp[0] = 1;
        for (int i = 1; i <= numPeople / 2; i++) {
            dp[i] = 0;
            for (int j = 0; j < i; j++) {
                dp[i] = (int) ((dp[i] + (long) dp[j] * dp[i - 1 - j]) % mod);
            }
        }
        return dp[numPeople / 2];
    }


    public int numBusesToDestination(int[][] routes, int source, int target) {
        int n = routes.length;
        boolean[] exploredRoutes = new boolean[n];
        Set<Integer> set = new HashSet<>();
        set.add(source);
        boolean flag = true;
        int counter = 0;
        while (flag) {
            counter++;
            List<Integer> routeList = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (exploredRoutes[i]) continue;
                int[] route = routes[i];
                for (int r : route) {
                    if (set.contains(r)) {
                        routeList.add(i);
                        exploredRoutes[i] = true;
                        break;
                    }
                }
            }
            for (Integer I : routeList) {
                int[] route = routes[I];
                for (int r : route) {
                    set.add(r);
                    if (r == target) {
                        return counter;
                    }
                }
            }
            if (routeList.isEmpty()) flag = false;
            routeList.clear();
        }
        return -1;
    }

    public int minMeetingRooms(int[][] intervals) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
        for (int[] interval : intervals) {
            int start = interval[0];
            int end = interval[1];
            if (pq.isEmpty()) pq.offer(end);
            else {
                int peek = pq.peek();
                if (peek <= start) {
                    pq.poll();
                    pq.offer(end);
                } else {
                    pq.offer(end);
                }
            }
        }
        return pq.size();
    }


    public int longestStrChain(String[] words) {
        Arrays.sort(words, Comparator.comparingInt(String::length));
        HashMap<String, Integer> hm = new HashMap<>();
        int maxLength = 1;
        for (String word : words) {
            int currentLength = 1;
            for (int i = 0; i < word.length(); i++) {
                String predecessor = word.substring(0, i) + word.substring(i + 1);
                if (hm.containsKey(predecessor)) {
                    currentLength = Math.max(currentLength, hm.get(predecessor) + 1);
                }
            }
            hm.put(word, currentLength);
            maxLength = Math.max(maxLength, currentLength);
        }
        return maxLength;
    }


    public int longestPalindromeSubseq(String s) {
        int n = s.length();
        char[] sChar = s.toCharArray();
        int[][] dp = new int[n + 1][n + 1];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (i == j) {
                    dp[i][j] = 1;
                } else {
                    boolean match = sChar[i] == sChar[j];
                    if (match) {
                        dp[i][j] = dp[i + 1][j - 1] + 2;
                    }
                    dp[i][j] = Math.max(dp[i][j], Math.max(dp[i][j - 1], dp[i + 1][j]));
                }
            }
        }
        return dp[0][n - 1];
    }

    public static long maximumBooks(int[] books) {
        int n = books.length;
        long output = 0;
        if (n == 1) return books[0];
        long[] dp = new long[n];
        dp[0] = books[0];
        Deque<Long> q = new ArrayDeque<>();
        q.offer((long) books[0]);
        for (int i = 1; i < n; i++) {
            long book = books[i];
            long priorBook = q.peekLast();
            if (priorBook == book) continue;
            if (priorBook < book) {
                dp[i] = dp[i - 1] + book;
            } else {
                dp[i] = book;
                while (!q.isEmpty()) {
                    book--;
                    book = Math.max(book, 0);
                    priorBook = q.poll();
                    book = Math.min(book, priorBook);
                    dp[i] += book;
                    if (book == 0) q.clear();
                }
            }
            q.offerLast((long) books[i]);
            output = Math.max(output, dp[i]);
        }
        return output;
    }

    public boolean canAttendMeetings(int[][] intervals) {
        Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
        int currentTime = 0;
        for (int[] interval : intervals) {
            int a = interval[0];
            int b = interval[1];
            if (currentTime > a) return false;
            currentTime = b;
        }
        return true;
    }

    public void wiggleSort(int[] nums) {
        Arrays.sort(nums);
        Deque<Integer> dq = new ArrayDeque<>(Arrays.stream(nums).boxed().toList());
        boolean flag = true;
        int index = 0;
        while (!dq.isEmpty()) {
            if (flag) {
                nums[index] = dq.pollFirst();
                flag = false;
            } else {
                nums[index] = dq.pollLast();
                flag = true;
            }
            index++;
        }
        return;
    }

    public int findCircleNum(int[][] isConnected) {
        int n = isConnected.length;
        int[] mask = new int[1];
        int counter = 0;
        for (int i = 0; i < n; i++) {
            if ((mask[0] & (1 << i)) != 0) {
                counter++;
                provinceDFS(isConnected, mask, i);
            }

        }
        return counter;
    }

    public void provinceDFS(int[][] graph, int[] mask, int currentPos) {
        mask[0] = (mask[0] | (1 << currentPos));
        int[] currentNode = graph[currentPos];
        for (int i = 0; i < currentNode.length; i++) {
            if (currentNode[i] == 1 && (mask[0] & (1 << i)) != 0) provinceDFS(graph, mask, i);
        }
    }

    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        if (root == null) return null;
        if (root.val <= p.val) {
            return inorderSuccessor(root.right, p);
        } else {
            TreeNode tn = inorderSuccessor(root.left, p);
            return (tn == null) ? root : tn;
        }
    }

    public String largestNumber(int[] nums) {
        String[] strNums = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            strNums[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(strNums, new Comparator<String>() {
            @Override
            public int compare(String a, String b) {
                String order1 = a + b;
                String order2 = b + a;
                return order2.compareTo(order1);
            }
        });
        if (strNums[0].equals("0")) {
            return "0";
        }
        StringBuilder sb = new StringBuilder();
        for (String str : strNums) {
            sb.append(str);
        }
        return sb.toString();
    }

    public static List<List<Integer>> verticalOrder(TreeNode root) {
        List<List<Integer>> output = new ArrayList<>();
        if (root == null) return output;
        HashMap<Integer, List<Integer>> hm = new HashMap<>();
        verticalOrder(root, hm, 0);
        List<Integer> keyList = new ArrayList<>(hm.keySet());
        Collections.sort(keyList);
        for (Integer key : keyList) {
            output.add(hm.get(key));
        }
        return output;
    }

    public static void verticalOrder(TreeNode root, HashMap<Integer, List<Integer>> hm, int index) {
        if (root == null) return;
        int value = root.val;
        System.out.println(value);
        hm.computeIfAbsent(index, k -> new ArrayList<>()).add(value);
        verticalOrder(root.left, hm, index - 1);
        verticalOrder(root.right, hm, index + 1);

    }

    public static TreeNode buildTree(Integer[] array) {
        if (array == null || array.length == 0 || array[0] == null) {
            return null;
        }

        TreeNode root = new TreeNode(array[0]);
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int i = 1;

        while (i < array.length) {
            TreeNode current = queue.poll();

            if (array[i] != null) {
                current.left = new TreeNode(array[i]);
                queue.add(current.left);
            }
            i++;

            if (i < array.length && array[i] != null) {
                current.right = new TreeNode(array[i]);
                queue.add(current.right);
            }
            i++;
        }

        return root;
    }


    class SparseVector {
        List<int[]> sv;

        SparseVector(int[] nums) {
            sv = new ArrayList<>();
            for (int i = 0; i < nums.length; i++) {
                if (nums[i] != 0) {
                    sv.add(new int[]{i, nums[i]});
                }
            }
        }

        // Return the dotProduct of two sparse vectors
        public int dotProduct(SparseVector vec) {
            int output = 0;
            List<int[]> vecSV = vec.sv;
            int a = 0;
            int b = 0;
            while (a < this.sv.size() && b < vecSV.size()) {
                int aIndex = this.sv.get(a)[0];
                int bIndex = vecSV.get(b)[0];
                if (aIndex == bIndex) {
                    output += this.sv.get(a)[1] * vecSV.get(b)[1];
                    a++;
                    b++;
                } else if (aIndex < bIndex) {
                    a++;
                } else {
                    b++;
                }
            }
            return output;
        }
    }

    public int numKLenSubstrNoRepeats(String s, int k) {
        int n = s.length();
        if (k == 1) return n;
        int output = 0;
        Deque<Character> dq = new ArrayDeque<>();
        HashMap<Character, Integer> hm = new HashMap<>();
        for (Character c : s.toCharArray()) {
            if (dq.size() >= k) {
                Character poll = dq.pollFirst();
                if (hm.get(poll) == 1) hm.remove(poll);
                else hm.merge(poll, -1, Integer::sum);
            }
            dq.addLast(c);
            hm.merge(c, 1, Integer::sum);
            if (hm.keySet().size() == k) output++;
        }
        return output;
    }


//    public class Solution extends Relation {
//        public int findCelebrity(int n) {
//            if (n == 1) return 0;
//            int candidate = 0;
//            for (int i = 1; i < n; i++) {
//                if (knows(candidate, i)) {
//                    candidate = i;
//                }
//            }
//            for (int i = 0; i < n; i++) {
//                if (i != candidate) {
//                    if (knows(candidate, i) || !knows(i, candidate)) {
//                        return -1;
//                    }
//                }
//            }
//            return candidate;
//        }
//    }

//    class Solution {
//        public void printLinkedListInReverse(ImmutableListNode head) {
//            if (head == null) return;
//            printLinkedListInReverse(head.getNext());
//            head.printValue();
//        }
//    }

    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
        int n = startTime.length;
        if (n == 1) return profit[0];
        int[][] dp = new int[n + 1][];
        dp[n] = new int[]{Integer.MAX_VALUE, Integer.MAX_VALUE, 0};
        for (int i = 0; i < n; i++) {
            dp[i] = new int[]{startTime[i], endTime[i], profit[i]};
        }
        Arrays.sort(dp, Comparator.comparingInt(a -> a[0]));
        for (int i = n - 2; i >= 0; i--) {
            int[] job = dp[i];
            int end = job[1];
            //Binary search
            int maxProfit = -1;
            int left = i + 1;
            int right = n;
            while (left <= right) {
                int middle = left + (right - left) / 2;
                int middleJobStart = dp[middle][0];
                if (middleJobStart < end) {
                    left = middle + 1;
                } else {
                    maxProfit = Math.max(maxProfit, dp[middle][2]);
                    right = middle - 1;
                }
            }
            job[2] += maxProfit;
            job[2] = Math.max(job[2], dp[i + 1][2]);
        }
        return dp[0][2];
    }


    public int climbStairs(int n) {
        if (n == 1) return 1;
        if (n == 2) return 2;
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }


    public int numberOfArithmeticSlices(int[] nums) {
        int n = nums.length;
        if (n <= 2) return 0;
        int[] dp = new int[n];
        int[] differences = new int[n];
        for (int i = n - 3; i >= 0; i--) {
            int a = nums[i];
            int b = nums[i + 1];
            int difference = b - a;
            differences[i] = difference;
            if (differences[i + 1] == difference) {
                dp[i] = dp[i + 1] + 1;
            }
        }
        return Arrays.stream(dp).sum();
    }

    int minPaintCost;

    public int paintWalls(int[] cost, int[] time) {
        int n = cost.length;
        if (n == 1) {
            return cost[0];
        }
        int totalTime = 0;
        int[][] jobs = new int[n][];
        for (int i = 0; i < n; i++) {
            jobs[i] = new int[]{time[i], cost[i]};
            totalTime += time[i];
        }
        Arrays.sort(jobs, Comparator.comparingInt(a -> a[0]));
        int maxTime = n >> 1;
        if ((maxTime & 1) == 1) maxTime++;
        minPaintCost = Integer.MAX_VALUE;
        paintWalls(jobs, 0, 0, maxTime, 0);
        return minPaintCost;
    }

    public void paintWalls(int[][] jobs, int jobIndex, int currentTime, int maxTime, int currentCost) {
        if (jobIndex == jobs.length) return;
        if (currentCost > minPaintCost) return;
        if (currentTime >= maxTime) {
            minPaintCost = Math.min(minPaintCost, currentCost);
            return;
        }
        for (int i = jobIndex; i < jobs.length; i++) {
            int[] job = jobs[i];
            paintWalls(jobs, jobIndex + 1, currentTime + job[0], maxTime, currentCost + job[1]);
        }
        return;
    }


    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode a = head;
        ListNode b = head.next;
        while (b != null) {
            if (a.val == b.val) {
                b = b.next;
                a.next = b;
            } else {
                a = b;
                b = a.next;
            }
        }
        return head;
    }


    public int maxJumps(int[] arr, int d) {
        int n = arr.length;
        Integer[] dp = new Integer[n];
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        for (int i = 0; i < n; i++) {
            pq.offer(new int[]{arr[i], i});
        }
        int output = -1;
        while (!pq.isEmpty()) {
            int[] current = pq.poll();
            int currentIndex = current[1];
            int currentHeight = current[0];
            dp[currentIndex] = 1;
            for (int i = 1; i <= d; i++) {
                if (i >= n || dp[currentIndex + i] == null || arr[currentIndex + i] >= arr[currentIndex]) break;
                dp[currentIndex] = Math.max(dp[currentIndex], dp[currentIndex + i] + 1);
            }
            for (int i = 1; i <= d; i++) {
                if (i >= 0 || dp[currentIndex - i] == null || arr[currentIndex + i] >= arr[currentIndex]) break;
                dp[currentIndex] = Math.max(dp[currentIndex], dp[currentIndex - i] + 1);
            }
            output = Math.max(output, dp[currentIndex]);
        }
        return output;
    }

    public static int maxSubarrayLength(int[] nums) {
        int n = nums.length;
        if (n < 2) return 0;
        if (n == 2) {
            if (nums[0] > nums[1]) return 2;
            return 0;
        }
        int output = 0;
        int[] minFromRight = new int[n];
        minFromRight[n - 1] = nums[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            minFromRight[i] = Math.min(minFromRight[i + 1], nums[i]);
        }
        int j = 0;
        for (int i = 0; i < n && j < n; i++) {
            //If overlap
            j = Math.max(j, i);
            //Otherwise
            while (j < n && nums[i] > minFromRight[j]) {
                j++;
            }
            output = Math.max(output, j - i);
        }
        return output;
    }


    public int earliestAcq(int[][] logs, int n) {
        int[] parentArray = new int[n];
        for (int i = 0; i < n; i++) parentArray[i] = i;
        Arrays.sort(logs, Comparator.comparingInt(a -> a[0]));
        for (int[] log : logs) {
            int time = log[0];
            int a = log[1];
            int b = log[2];
            if (unionJoin(parentArray, a, b)) --n;
            if (n == 1) return time;
        }
        return -1;
    }

    public int findParent(int[] parentArray, int currentNode) {
        //Base case, if the current node's parent is itself
        if (parentArray[currentNode] == currentNode) return parentArray[currentNode];
        else {
            //The parent array may not be updated. By recursively calling itself, we update the parent array
            return parentArray[currentNode] = findParent(parentArray, parentArray[currentNode]);
        }
    }

    public boolean unionJoin(int[] parentArray, int nodeA, int nodeB) {
        int parentA = findParent(parentArray, nodeA);
        int parentB = findParent(parentArray, nodeB);
        if (parentA == parentB) return false;
        if (parentA > parentB) parentArray[parentA] = parentB;
        else parentArray[parentB] = parentA;
        return true;
    }

    public boolean canMeasureWater(int x, int y, int target) {
        if (x + y < target) return false;
        Boolean[][] dp = new Boolean[x + 1][y + 1];
        return waterMeasureDFS(0, 0, x, y, target, dp);
    }

    public boolean waterMeasureDFS(int x, int y, int xMax, int yMax, int target, Boolean[][] dp) {
        if (x == target || y == target || (x + y) == target) return true;
        if (x > xMax) return waterMeasureDFS(xMax, y, xMax, yMax, target, dp);
        if (y > yMax) return waterMeasureDFS(x, yMax, xMax, yMax, target, dp);
        if (dp[x][y] != null) return dp[x][y];
        dp[x][y] = dp[x][y] || waterMeasureDFS(xMax, y, xMax, yMax, target, dp);
        dp[x][y] = dp[x][y] || waterMeasureDFS(x, yMax, xMax, yMax, target, dp);
        dp[x][y] = dp[x][y] || waterMeasureDFS(0, y, xMax, yMax, target, dp);
        dp[x][y] = dp[x][y] || waterMeasureDFS(x, 0, xMax, yMax, target, dp);
        dp[x][y] = dp[x][y] || waterMeasureDFS(x + y, Math.max(0, y - (xMax - x)), xMax, yMax, target, dp);
        dp[x][y] = dp[x][y] || waterMeasureDFS(Math.max(0, x - (yMax - y)), x + y, xMax, yMax, target, dp);
        return dp[x][y];
    }


    public static int numSquares(int n) {
        if (n <= 0) return 0;
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j * j <= i; j++) {
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }
        }
        return dp[n];
    }

    public int countSubstrings(String s) {
        int n = s.length();
        char[] sChar = s.toCharArray();
        if (n == 1) return 1;
        if (n == 2) {
            if (sChar[0] == sChar[1]) return 3;
            return 2;
        }
        boolean[][] dp = new boolean[n + 1][n + 1];
        int output = 0;
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (sChar[i] == sChar[j]) {
                    if (i == j || (i + 1) == j) {
                        dp[i][j] = true;
                        output++;
                    } else {
                        if (dp[i + 1][j - 1]) {
                            dp[i][j] = true;
                            output++;
                        } else {
                            dp[i][j] = false;
                        }
                    }
                }
            }
        }
        return output;
    }


    public boolean isValidPalindrome(String s, int k) {
        int n = s.length();
        char[] sChar = s.toCharArray();
        if (n == 1) return true;
        if (n == 2) {
            if (sChar[0] == sChar[1]) return true;
            if (k >= 1) return true;
            return false;
        }
        int[][] dp = new int[n + 1][n + 1];
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                boolean match = sChar[i] == sChar[j];
                if (j == (i + 1)) {
                    if (match) dp[i][j] = 0;
                    else dp[i][j] = 1;
                } else {
                    if (match) dp[i][j] = dp[i + 1][j - 1];
                    else {
                        dp[i][j] = Math.min(dp[i + 1][j] + 1, dp[i][j - 1] + 1);
                    }
                }
            }
        }
        return (dp[0][n - 1] <= k) ? true : false;
    }

    public int numberWays(List<List<Integer>> hats) {
        int n = hats.size();
        int maxHats = 40;
        long[] dp = new long[1 << n];
        dp[0] = 1;
        List<List<Integer>> hatToPeople = new ArrayList<>();
        for (int i = 0; i <= maxHats; i++) {
            hatToPeople.add(new ArrayList<>());
        }
        for (int i = 0; i < n; i++) {
            for (int hat : hats.get(i)) {
                hatToPeople.get(hat).add(i);
            }
        }
        for (int hat = 1; hat <= maxHats; hat++) {
            long[] newDp = dp.clone();
            for (int mask = 0; mask < (1 << n); mask++) {
                if (dp[mask] > 0) {
                    for (int person : hatToPeople.get(hat)) {
                        if ((mask & (1 << person)) == 0) {
                            newDp[mask | (1 << person)] = (newDp[mask | (1 << person)] + dp[mask]) % 1_000_000_007;
                        }
                    }
                }
            }
            dp = newDp;
        }
        return (int) dp[(1 << n) - 1];
    }

    public static int longestRepeatingSubstring(String s) {
        int n = s.length();
        int[][] dp = new int[n + 1][n + 1];
        int maxLength = 0;

        // Iterate over the DP table
        for (int i = 1; i <= n; i++) {
            for (int j = i + 1; j <= n; j++) {
                if (s.charAt(i - 1) == s.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                    maxLength = Math.max(maxLength, dp[i][j]);
                }
            }
        }
        return maxLength;
    }


    public boolean predictTheWinner(int[] nums) {
        int n = nums.length;
        if (n <= 2) return true;
        int[][] dp = new int[n + 1][n + 1];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (i == j) dp[i][j] = nums[i];
                else {
                    int a = nums[i];
                    int b = nums[j];
                    int res;
                    if (i + 1 == j) {
                        res = Math.abs(a - b);
                    } else {
                        res = Math.max(a - dp[i + 1][j], b - dp[i][j - 1]);
                    }
                    dp[i][j] = res;
                }
            }
        }
        return (dp[0][n - 1] >= 0);
    }

    public int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
        HashMap<Integer, Integer> hm = new HashMap<>();
        for (int[] interval : firstList) {
            int a = interval[0];
            int b = interval[1];
            hm.merge(a, 1, Integer::sum);
            hm.merge(b, -1, Integer::sum);
        }
        for (int[] interval : secondList) {
            int a = interval[0];
            int b = interval[1];
            hm.merge(a, 1, Integer::sum);
            hm.merge(b, -1, Integer::sum);
        }
        List<Integer> keyList = new ArrayList<>(hm.keySet());
        Collections.sort(keyList);
        boolean activeInterval = false;
        int priorSum = 0;
        int start = 0;
        int sum = 0;
        List<int[]> outputList = new ArrayList<>();
        for (Integer key : keyList) {
            sum = priorSum + hm.get(key);
            if (priorSum == 1) {
                if (sum == 1) {
                    outputList.add(new int[]{key, key});
                } else if (sum == 2) {
                    start = key;
                }
            }
            if (priorSum == 2) {
                if (sum < 2) {
                    outputList.add(new int[]{start, key});
                }
            }
            if (priorSum == 0) {
                if (sum == 2) {
                    start = key;
                }
            }
            priorSum = sum;
        }
        int n = outputList.size();
        int[][] output = new int[n][];
        for (int i = 0; i < n; i++) {
            output[i] = outputList.get(i);
        }
        return output;
    }

    int minPrice;

    public int shoppingOffers(List<Integer> price, List<List<Integer>> special, List<Integer> needs) {
        minPrice = Integer.MAX_VALUE;
        shoppingOffers(price, special, needs, 0, 0);
        return minPrice;
    }

    public void shoppingOffers(List<Integer> price, List<List<Integer>> special, List<Integer> needs, int specialIndex, int currentPrice) {
        if (currentPrice >= minPrice) return;
        int n = price.size();
        if (specialIndex == special.size()) {
            for (int i = 0; i < n; i++) {
                currentPrice += needs.get(i) * price.get(i);
            }
            minPrice = Math.min(minPrice, currentPrice);
            return;
        }
        shoppingOffers(price, special, needs, specialIndex + 1, currentPrice);
        List<Integer> currentSpecial = special.get(specialIndex);
        boolean valid = true;
        for (int j = 0; j < n; j++) {
            if (needs.get(j) < currentSpecial.get(j)) {
                valid = false;
                break;
            }
        }
        if (valid) {
            for (int j = 0; j < n; j++) {
                needs.set(j, needs.get(j) - currentSpecial.get(j));
            }
            shoppingOffers(price, special, needs, specialIndex, currentPrice + currentSpecial.get(n));
            for (int j = 0; j < n; j++) {
                needs.set(j, needs.get(j) + currentSpecial.get(j));
            }
        }
    }


    class LRUCache {
        private final int capacity;
        private final LinkedHashMap<Integer, Integer> cache;

        public LRUCache(int capacity) {
            this.capacity = capacity;
            this.cache = new LinkedHashMap<Integer, Integer>(capacity, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
                    return size() > capacity;
                }
            };
        }

        public int get(int key) {
            return cache.getOrDefault(key, -1);
        }

        public void put(int key, int value) {
            cache.put(key, value);
        }

    }


//    int islandCounter;
//    int yMax;
//    int xMax;
//    char[][] islandGrid;
//    boolean[][] visited;
//
//    public int numIslands(char[][] grid) {
//        islandGrid = grid;
//        islandCounter = 0;
//        visited = new boolean[yMax+1][xMax+1];
//        yMax = grid.length;
//        xMax = grid[0].length;
//        for (int y = 0; y < yMax; y++) {
//            for (int x = 0; x < xMax; x++) {
//                if (visited[y][x]) continue;
//                if (islandGrid[y][x] == '1') {
//                    islandDFS(y, x);
//                }
//                visited[y][x] = true;
//            }
//        }
//        return islandCounter;
//    }
//
//    public void islandDFS(int y, int x) {
//        if (y < 0 || x < 0 || y == yMax || x == xMax) return;
//        if (visited[y][x]) return;
//        visited[y][x] = true;
//        if (islandGrid[y][x] == '1') {
//            islandDFS(y+1,x);
//            islandDFS(y,x+1);
//            islandDFS(y-1,x);
//            islandDFS(y,x-1);
//        }
//    }


    public int minimumDeleteSum(String s1, String s2) {
        int iMax = s1.length();
        int jMax = s2.length();
        char[] c1 = s1.toCharArray();
        char[] c2 = s2.toCharArray();
        int[][] dp = new int[iMax+1][jMax+1];
        for (int i = iMax - 1; i >= 0; i--) {
            dp[i][jMax] = (int) c1[i] + dp[i+1][jMax];
        }
        for (int j = jMax - 1; j >= 0; j--) {
            dp[iMax][j] = (int) c2[j] + dp[iMax][j+1];
        }
        for (int i = iMax - 1; i >= 0; i--) {
            for (int j = jMax - 1; j >= 0; j--) {
                if (c1[i] != c2[j]) {
                    dp[i][j] = Math.min((int) c1[i] + dp[i+1][j], (int) c2[j] + dp[i][j+1]);
                } else {
                    dp[i][j] = dp[i+1][j+1];
                }
            }
        }
        return dp[0][0];
    }



    public double minimumAverage(int[] nums) {
        Arrays.sort(nums);
        PriorityQueue<Double> pq = new PriorityQueue<>();
        int l = 0;
        int r = nums.length-1;
        while (l < r) {
            pq.offer(((double) nums[l] + nums[r]));
            l++;
            r--;
        }
        return (pq.poll()/2d);
    }


    public int minimumArea(int[][] grid) {
        int n = grid.length;
        int m = grid[0].length;
        int xMin = Integer.MAX_VALUE;
        int xMax = Integer.MIN_VALUE;
        int yMin = Integer.MAX_VALUE;
        int yMax = Integer.MIN_VALUE;
        boolean flag = false;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (grid[i][j] == 1) {
                    xMin = Math.min(xMin, i);
                    xMax = Math.max(xMax, i);
                    yMin = Math.min(yMin, j);
                    yMax = Math.max(yMax, j);
                    flag = true;
                }
            }
        }
        if (!flag) {
            return 0;
        }
        return (xMax - xMin + 1) * (yMax - yMin + 1);
    }




    public int threeSumSmaller(int[] nums, int target) {
        int count = 0;
        Arrays.sort(nums);
        int len = nums.length;
        for(int i=0; i<len-2; i++) {
            int left = i+1, right = len-1;
            while(left < right) {
                if(nums[i] + nums[left] + nums[right] < target) {
                    count += right-left;
                    left++;
                } else {
                    right--;
                }
            }
        }

        return count;
    }

    public void rotate(int[][] matrix) {
        int n = matrix.length;
        if (n == 1) return;
        //Transpose
        for (int i = 0; i < n; i++) {
            for (int j = i+1; j < n; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = tmp;
            }
        }
        //Reverse each row
        for (int[] row : matrix) {
            int r = n-1;
            int l = 0;
            while (l < r) {
                int tmp = row[l];
                row[l] = row[r];
                row[r] = tmp;
                l++;
                r--;
            }
        }
    }
    int guess(int i) {return i;}

    public int guessNumber(int n) {
        int left = 0;
        int right = n;
        int mid = -1;
        while (left <= right) {
            mid = left + (right - left)/2;
            int result = guess(mid);
            if (result == 0) return mid;
            else if (result == 1) {
                left = mid + 1;
            } else {
                right = mid-1;
            }
        }
        return -1;
    }



    public static int maxElement(int n, int maxSum, int k) {
        // Write your code here
        int leftLength = k + 1;
        int rightLength = n - k;

        int left = 1;
        int right = maxSum;
        int mid = -1;
        int output = 1;
        while (left <= right) {
            mid = left + ((right - left)/2);
            long currentSum = checkSum(mid, leftLength) + checkSum(mid, rightLength) - mid;
            if (currentSum > maxSum) {
                right = mid - 1;
            } else {
                output = mid;
                left = mid + 1;
            }
        }
        return output;

    }

    public static long checkSum(long upperBound, int length) {
        if (length == 1) return upperBound;
        if (upperBound < length) {
            return upperBound * (upperBound + 1) / 2 + (length - upperBound);
        }

        long lowerBound = upperBound - length + 1;
        long sum = (upperBound * (upperBound + 1) / 2) - (lowerBound * (lowerBound - 1) / 2);
        return sum;
    }


    public int convertArray(int[] nums) {
        int n = nums.length;
        if (n == 1) return 0;
        PriorityQueue<Integer> increasing = new PriorityQueue<>(Collections.reverseOrder());
        PriorityQueue<Integer> decreasing = new PriorityQueue<>();
        int increasingCost = 0;
        int decreasingCost = 0;
        for (int i : nums) {
            if (!increasing.isEmpty() && increasing.peek() > i) {
                increasingCost += increasing.poll()-i;
                increasing.offer(i);
            }
            if (!decreasing.isEmpty() && decreasing.peek() < i) {
                decreasingCost += i - decreasing.poll();
                decreasing.offer(i);
            }
            increasing.offer(i);
            decreasing.offer(i);
        }
        return Math.min(increasingCost, decreasingCost);
    }



    public int minFlipsMonoIncr(String s) {
        int n = s.length();
        if (n == 1) return 0;
        int[][] dp = new int[n+1][2];
        char[] sChar = s.toCharArray();
        for (int i = n - 1; i >= 0; i--) {
            char c = sChar[i];
            if (c == '1') {
                dp[i][1] = dp[i+1][1];
                dp[i][0] = Math.min(dp[i+1][0], dp[i+1][1]) + 1;
            } else {
                dp[i][1] = dp[i+1][1] + 1;
                dp[i][0] = Math.min(dp[i+1][0], dp[i+1][1]);
            }
        }
        return Math.min(dp[0][1], dp[0][0]);
    }

    public int minStickers(String[] stickers, String target) {
        Arrays.sort(stickers, Comparator.comparingInt(a -> -a.length()));
        List<int[]> stickerList = new ArrayList<>();
        for (String sticker : stickers) {
            int[] currentSticker = new int[26];
            for (char c : sticker.toCharArray()) {
                currentSticker[c - 'a']++;
            }
            boolean dominated = false;
            for (int[] existingSticker : stickerList) {
                dominated = true;
                for (int i = 0; i < 26; i++) {
                    if (existingSticker[i] < currentSticker[i]) {
                        dominated = false;
                        break;
                    }
                }
                if (dominated) break;
            }
            if (!dominated) stickerList.add(currentSticker);
        }

        int sLength = target.length();
        char[] targetChar = target.toCharArray();

        // DP preparation
        HashMap<Integer, Integer> dp = new HashMap<>();
        dp.put(0, 0);
        int dpMax = 1 << sLength;
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
        pq.add(new int[]{0,0});
        // DP
        while (!pq.isEmpty()) {
            int[] a = pq.poll();
            int count = a[1];
            int mask = a[0];
            int currentStickerCount = dp.get(mask);
            if (currentStickerCount < count) continue;
            for (int[] sticker : stickerList) {
                int nextMask = mask;
                int[] stickerCopy = Arrays.copyOf(sticker, 26);
                for (int j = 0; j < sLength; j++) {
                    if ((nextMask & (1 << j)) == 0 && stickerCopy[targetChar[j] - 'a'] > 0) {
                        stickerCopy[targetChar[j] - 'a']--;
                        nextMask |= (1 << j);
                    }
                }
                if (nextMask != mask && dp.getOrDefault(nextMask, Integer.MAX_VALUE) > currentStickerCount + 1) {
                    dp.put(nextMask, Math.min(dp.getOrDefault(nextMask, Integer.MAX_VALUE), currentStickerCount + 1));
                    pq.offer(new int[]{nextMask, currentStickerCount + 1});
                }
            }
        }
        return dp.getOrDefault(dpMax - 1, -1);
    }



    public List<String> wordBreak(String s, List<String> wordDict) {
        Set<String> set = new HashSet<>(wordDict);
        int n = s.length();
        HashMap<Integer, List<StringBuilder>> dp = new HashMap<>();
        dp.put(n, new ArrayList<>(Arrays.asList(new StringBuilder(""))));
        for (int i = n - 1; i >= 0; i--) {
            if (!dp.containsKey(i + 1)) continue;
            List<StringBuilder> priorStringBuilders = dp.get(i + 1);
            for (int j = i; j >= 0; j--) {
                String subString = s.substring(j, i + 1);
                if (set.contains(subString)) {
                    for (StringBuilder priorStringBuilder : priorStringBuilders) {
                        StringBuilder newStringBuilder = new StringBuilder(subString).append(" ").append(priorStringBuilder);
                        dp.computeIfAbsent(j, k -> new ArrayList<>()).add(newStringBuilder);
                    }
                }
            }
        }
        List<String> result = new ArrayList<>();
        for (StringBuilder sb : dp.getOrDefault(0, new ArrayList<>())) {
            result.add(sb.toString().trim());
        }
        return result;
    }



    public int minimumTime(int n, int[][] relations, int[] time) {
        HashMap<Integer, List<Integer>> graph = new HashMap<>();
        int[] childCount = new int[n];
        for (int[] relation : relations) {
            int a = relation[0] - 1;
            int b = relation[1] - 1;
            childCount[b]++;
            graph.computeIfAbsent(a, k -> new ArrayList<>()).add(b);
        }
        int[] cumTime = new int[n];
        Deque<Integer> dq = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            if (childCount[i] == 0) dq.addLast(i);
        }
        int output = 0;
        while (!dq.isEmpty()) {
            Integer currentCourse = dq.pollFirst();
            cumTime[currentCourse] += time[currentCourse];
            List<Integer> nextCourses = graph.getOrDefault(currentCourse, new ArrayList<>());
            for (Integer nextCourse : nextCourses) {
                cumTime[nextCourse] = Math.max(cumTime[nextCourse], cumTime[currentCourse]);
                childCount[nextCourse]--;
                if (childCount[nextCourse] == 0) dq.addLast(nextCourse);
            }
            output = Math.max(output, cumTime[currentCourse]);
        }
        return output;
    }

    long pCount = 0;
    char[] treeChars = null;
    List<Integer> maskList = null;
    public long countPalindromePaths(List<Integer> parent, String s) {
        pCount = 0;
        treeChars = s.toCharArray();
        maskList = new ArrayList<>();
        charTreeDFS(parent, 0);
        long output = 0;
        for (Integer mask : maskList) {
            if (Integer.bitCount(mask) <= 1) output++;
        }
        return output;
    }
    public List<Integer> charTreeDFS(List<Integer> parentList, int currentNode) {
        List<Integer> localMaskList = new ArrayList<>();
        int charValue = treeChars[currentNode] - 'a';
        Integer currentMask = 0 | (1 << charValue);
        localMaskList.add(currentMask);
        for (int i = 0; i < parentList.size(); i++) {
            if (parentList.get(i) == currentNode) {
                List<Integer> nextMasks = charTreeDFS(parentList, i);
                for (Integer nextMask : nextMasks) {
                    for (Integer localMask : localMaskList) {
                        maskList.add(nextMask & localMask);
                    }
                }
                for (Integer nextMask : nextMasks) {
                     localMaskList.add(nextMask & currentMask);
                }
            }
        }
        return localMaskList;
    }




    public String reorganizeString(String s) {
        // Calculate the frequency of each character
        int[] charFrequency = new int[26];
        for (char c : s.toCharArray()) {
            charFrequency[c - 'a']++;
        }
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> b[1] - a[1]);
        for (int i = 0; i < 26; i++) {
            int charFreq = charFrequency[i];
            if (charFreq > 0) pq.offer(new int[]{i, charFreq});
        }

        StringBuilder sb = new StringBuilder();
        int[] prev = {-1, 0}; // To store the previous character and its remaining frequency
        while (!pq.isEmpty()) {
            int[] current = pq.poll();
            sb.append((char) (current[0] + 'a'));
            current[1]--;
            if (prev[1] > 0) {
                pq.offer(prev);
            }
            prev = current;
        }
        if (sb.length() != s.length()) {
            return "";
        }
        return sb.toString();
    }





    public static void main(String[] args) {
        int[][] workers = {{0, 0}, {2, 1}};
        int[][] bikes = {{1, 2}, {3, 3}};
        int i = maxSubarrayLength(new int[]{7, 6, 5, 4, 3, 2, 1, 6, 10, 11});
        int[] zArray = computeZArray("aabqwertycqwertydaab");
        int[][] hats = stringToArray2D("[[3,4],[4,5],[5]]");
        List<List<Integer>> hatList = new ArrayList<>();
        for (int[] hat : hats) {
            List<Integer> l = new ArrayList<>();
            for (int h : hat) l.add(h);
            hatList.add(l);
        }

        String[] stickers = {"apple", "banana", "cherry", "date", "fig", "grape"};

        // Sort the array by reverse length using lambda expression
        Arrays.sort(stickers, Comparator.comparingInt((String a) -> -a.length()));

        // Print the sorted array
        System.out.println("Strings sorted by reverse length:");
        for (String str : stickers) {
            System.out.println(str);
        }


    }


    public static int[] stringToArray1D(String input) {
        String[] numberStrings = input.substring(1, input.length() - 1).split(",");
        int[] numbers = new int[numberStrings.length];
        for (int i = 0; i < numberStrings.length; i++) {
            numbers[i] = Integer.parseInt(numberStrings[i]);
        }
        return numbers;
    }

    // Method for 2D array
    public static int[][] stringToArray2D(String input) {
        // Remove outer brackets and split into individual array strings
        String[] arrayStrings = input.substring(1, input.length() - 1).split("(?<=\\]),\\[");
        // Prepare a list to hold the final arrays
        List<int[]> arraysList = new ArrayList<>();
        for (String arrayString : arrayStrings) {
            // Remove brackets from each array string and split by comma
            String[] numberStrings = arrayString.replaceAll("[\\[\\]]", "").split(",");
            int[] numbers = new int[numberStrings.length];
            for (int i = 0; i < numberStrings.length; i++) {
                numbers[i] = Integer.parseInt(numberStrings[i]);
            }
            arraysList.add(numbers);
        }
        // Convert list to array
        int[][] result = new int[arraysList.size()][];
        for (int i = 0; i < arraysList.size(); i++) {
            result[i] = arraysList.get(i);
        }
        return result;
    }


    class Interval {
        public int start;
        public int end;

        public Interval() {
        }

        public Interval(int _start, int _end) {
            start = _start;
            end = _end;
        }
    }


    public static class TreeNode {
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

        ListNode(int x) {
            val = x;
            next = null;
        }
    }

}

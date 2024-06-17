import java.util.*;
import java.util.stream.*;


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
        System.out.println(pos/n);
        System.out.println(neg/n);
        System.out.println((n - pos - neg)/n);
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
        String last = sArray[n-1];
        return last.length();
    }






    public static void main(String[] args) {
        int[] tasks = {10, 6, 6, 8, 3, 7};
        minSessions(tasks, 13);
        int[] spells = {3, 1, 2};
        int[] potions = {8, 5, 8};


        String og = abbreviationOG("daBcD", "ABCD");
        String ab = abbreviation("daBcD", "ABCD");


        int[][] game = {{100}};
        int fc = calculateMinimumHP(game);








        for (int p = 0; p < 3; p++) {
            for (int q = 0; q < 3; q++) {
                int lhs = relevantImplication(p, q);
                int rhs = relevantOr(relevantNegate(p), q);
                if (lhs == 1 && (rhs == 0 || rhs == 2) || lhs == 2 && rhs == 0) {
                    System.out.println(p);
                    System.out.println(q);
                    System.out.println("q1 relevant invalid");
                }

            }
        }
        for (int p = 0; p < 3; p++) {
            for (int q = 0; q < 3; q++) {
                int lhs = fuzzyImplication(p, q);
                int rhs = fuzzyOr(fuzzyNegate(p), q);
                if (lhs == 1 && (rhs == 0 || rhs == 2) || lhs == 2 && rhs == 0) {
                    System.out.println(p);
                    System.out.println(q);
                    System.out.println("q1 fuzzy invalid");
                }

            }
        }
        for (int p = 0; p < 3; p++) {
            for (int q = 0; q < 3; q++) {
                int rhs = fuzzyAnd(p, q);
                int lhs = fuzzyNegate(fuzzyNegate(fuzzyAnd(p, q)));
                if (lhs == 1 && (rhs == 0 || rhs == 2) || lhs == 2 && rhs == 0) {
                    System.out.println(p);
                    System.out.println(q);
                    System.out.println("q2 fuzzy invalid");
                }

            }
        }
        for (int p = 0; p < 3; p++) {
            for (int q = 0; q < 3; q++) {
                int rhs = intuitionisticAnd(p, q);
                int lhs = intuitionisticNegate(intuitionisticNegate(intuitionisticAnd(p, q)));
                if (lhs == 1 && (rhs == 0 || rhs == 2) || lhs == 2 && rhs == 0) {
                    System.out.println(p);
                    System.out.println(q);
                    System.out.println("q2 intuitionistic invalid");
                }
            }
        }

        for (int p = 0; p < 3; p++) {
            for (int q = 0; q < 3; q++) {
                for (int r = 0; r < 3; r++) {
                    int lhs = relevantAnd(p, relevantAnd(q, r));
                    int rhs = relevantAnd(relevantImplication(p, q), relevantImplication(p, r));
                    if (lhs == 1 && (rhs == 0 || rhs == 2) || lhs == 2 && rhs == 0) {
                        System.out.println(p);
                        System.out.println(q);
                        System.out.println("q3 relevant invalid");
                    }
                }

            }
        }
        for (int p = 0; p < 3; p++) {
            for (int q = 0; q < 3; q++) {
                for (int r = 0; r < 3; r++) {
                    int lhs = intuitionisticAnd(p, intuitionisticAnd(q, r));
                    int rhs = intuitionisticAnd(intuitionisticImplication(p, q), intuitionisticImplication(p, r));
                    if (lhs == 1 && (rhs == 0 || rhs == 2) || lhs == 2 && rhs == 0) {
                        System.out.println(p);
                        System.out.println(q);
                        System.out.println("q3 intui invalid");
                    }
                }

            }
        }


        int lhs = intuitionisticAnd(2, 2);
        int rhs = intuitionisticNegate(intuitionisticAnd(intuitionisticNegate(2), intuitionisticNegate(2)));
        System.out.println(lhs);
        System.out.println(rhs);
    }


    public static int fuzzyAnd(int a, int b) {
        if (a == 0) return 0;
        if (a == 1) return b;
        if (a == 2) {
            if (b == 0) return 0;
            else return 2;
        }
        return -1;
    }

    public static int intuitionisticAnd(int a, int b) {
        if (a == 0) return 0;
        if (a == 1) return b;
        if (a == 2) {
            if (b == 0) return 0;
            else return 2;
        }
        return -1;
    }

    public static int relevantAnd(int a, int b) {
        if (a == 0) return 0;
        if (a == 1) return b;
        if (a == 2) {
            if (b == 0) return 0;
            else return 2;
        }
        return -1;
    }

    public static int fuzzyOr(int a, int b) {
        if (a == 0) return b;
        if (a == 1) return 1;
        if (a == 2) {
            if (b == 1) return 1;
            else return 2;
        }
        return -1;
    }

    public static int intuitionisticOr(int a, int b) {
        if (a == 0) return b;
        if (a == 1) return 1;
        if (a == 2) {
            if (b == 1) return 1;
            else return 2;
        }
        return -1;
    }

    public static int relevantOr(int a, int b) {
        if (a == 0) return b;
        if (a == 1) return 1;
        if (a == 2) {
            if (b == 1) return 1;
            else return 2;
        }
        return -1;
    }

    public static int fuzzyImplication(int a, int b) {
        if (a == 0) return 1;
        if (a == 1) return b;
        if (a == 2) {
            if (b == 0) return 2;
            else return 1;
        }
        return -1;
    }

    public static int intuitionisticImplication(int a, int b) {
        if (a == 0) return 1;
        if (a == 1) return b;
        if (a == 2) {
            if (b == 0) return 0;
            else return 1;
        }
        return -1;
    }

    public static int relevantImplication(int a, int b) {
        if (a == 0) return 1;
        if (a == 1) {
            if (b == 1) return 1;
            else return 0;
        }
        if (a == 2) return b;
        return -1;
    }

    public static int fuzzyNegate(int a) {
        if (a == 0) return 1;
        if (a == 1) return 0;
        if (a == 2) return 2;
        return -1;
    }

    public static int intuitionisticNegate(int a) {
        if (a == 0) return 1;
        if (a == 1) return 0;
        if (a == 2) return 0;
        return -1;
    }

    public static int relevantNegate(int a) {
        if (a == 0) return 1;
        if (a == 1) return 0;
        if (a == 2) return 2;
        return -1;
    }


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

        ListNode(int x) {
            val = x;
            next = null;
        }
    }

}

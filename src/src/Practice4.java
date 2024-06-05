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
        return list.stream().mapToInt(i->i).toArray();
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
        if (index == n+1) return 1;
        List<Integer> list = hm.get(index);
        int output = 0;
        for (Integer I : list) {
            if (!set.contains(I)) {
                set.add(I);
                output += countArrangement(hm, index+1, n, set);
                set.remove(I);
            }
        }
        return output;
    }



    public static String compressedString(String word) {
        int n = word.length();
        if (n == 1) return "1"+word;
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
                .map(e->e.toString())
                .reduce((acc, e) -> acc  + e)
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
            hm.put(card,i);
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
        int[] dpArray = new int[n+1];
        int prior = 1;
        for (int i = 0; i <= n; i++) {
            dpArray[i] += prior;
            if (i+4 <= n) dpArray[i+4] = dpArray[i];
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
        long[] dpArray = new long[n+1];
        long currentSum = 0;
        for (int i = n - 1; i >= 0; i--) {
            long multiplier = i + 1;
            currentSum += H.get(i);
            dpArray[i] = Math.max(dpArray[i+1], currentSum * multiplier);
        }
        return dpArray[0];
    }



    public static int unboundedKnapsack(int k, List<Integer> arr) {
        if (arr.contains(1)) return k;
        boolean[] dpArray = new boolean[k+1];
        dpArray[0] = true;
        Collections.sort(arr, Collections.reverseOrder());
        for (Integer I : arr) {
            for (int i = 0; i <= k - I; i++) {
                if (dpArray[i]) {
                    if (i == k-I) return k;
                    dpArray[i+I] = true;
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





    public static String abbreviation(String a, String b) {
        char[] yCharArray = b.toCharArray();
        char[] xCharArray = a.toCharArray();
        int yMax = b.length();
        int xMax = a.length();
        boolean[][] dpMatrix = new boolean[yMax+1][xMax+1];
        boolean[] last = dpMatrix[yMax];
        last[xMax] = true;
        for (int i = xMax - 1; i >= 0; i--) {
            if (Character.isUpperCase(xCharArray[i])) last[i] = false;
            else last[i] = last[i+1];
        }
        dpMatrix[yMax] = last;
        for (int y = yMax-1; y >= 0; y--) {
            for (int x = xMax-1; x >= 0; x--) {
                char yChar = yCharArray[y];
                char xChar = xCharArray[x];
                if (yChar == xChar) {
                    dpMatrix[y][x] = dpMatrix[y+1][x+1];
                } else if (Character.toUpperCase(xChar) == yChar) {
                    dpMatrix[y][x] = dpMatrix[y+1][x+1] || dpMatrix[y][x+1];
                } else {
                    dpMatrix[y][x] = dpMatrix[y][x+1];
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
                    if (indexes.get(i) - indexes.get(i-1) <= indexDiff) return true;
                }
            }
            return false;
        }
        else {
            int keyListSize = keyList.size();
            for (int i = 0; i < keyListSize-1; i++) {
                int lowerKey = keyList.get(i);
                for (int j = i; j < keyListSize; j++) {
                    if (j == i) {
                        List<Integer> indexes = new ArrayList<>(indexMap.get(lowerKey));
                        for (int k = 1; k < indexes.size(); k++) {
                            if (indexes.get(k) - indexes.get(k-1) <= indexDiff) return true;
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
        boolean[] flagArray = new boolean[k+1];
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




    public static void main(String[] args) {
        int i = redJohn(5);
        mandragora(Arrays.asList(3,2,5));
        int ii = unboundedKnapsack(12, Arrays.asList(1, 6, 9));
        int iii = unboundedKnapsack(9, Arrays.asList(3, 4, 4, 4, 8));
        int iiii = unboundedKnapsack(11, Arrays.asList(3 ,7, 9));
        int iiiii = unboundedKnapsack(11, Arrays.asList(3 ,7, 9));
        int iiiiii = unboundedKnapsack(11, Arrays.asList(3 ,7, 9));
        int[] prices = {1, 2, 100};
        stockmax(new ArrayList<>(Arrays.stream(prices).boxed().toList()));
        String s = compressedString("abcde");



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
                int rhs = intuitionisticAnd(p,q);
                int lhs = intuitionisticNegate(intuitionisticNegate(intuitionisticAnd(p,q)));
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
                    int lhs = relevantAnd(p, relevantAnd(q,r));
                    int rhs = relevantAnd(relevantImplication(p,q),relevantImplication(p,r));
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
                    int lhs = intuitionisticAnd(p, intuitionisticAnd(q,r));
                    int rhs = intuitionisticAnd(intuitionisticImplication(p,q),intuitionisticImplication(p,r));
                    if (lhs == 1 && (rhs == 0 || rhs == 2) || lhs == 2 && rhs == 0) {
                        System.out.println(p);
                        System.out.println(q);
                        System.out.println("q3 intui invalid");
                    }
                }

            }
        }


        int lhs = intuitionisticAnd(2,2);
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


}

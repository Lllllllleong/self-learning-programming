import java.math.*;
import java.sql.*;
import java.util.*;


public class Practice5 {


    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int n = numCourses;
        List<Integer>[] graph = new List[numCourses];
        for (int i = 0; i < numCourses; i++) {
            graph[i] = new ArrayList<>();
        }
        int[] childCount = new int[numCourses];
        for (int[] prereq : prerequisites) {
            int a = prereq[0];
            int b = prereq[1];
            graph[b].add(a);
            childCount[a]++;
        }
        Deque<Integer> dq = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            if (childCount[i] == 0) dq.addLast(i);
        }
        while (!dq.isEmpty()) {
            int currentCourse = dq.pollFirst();
            for (Integer nextCourse : graph[currentCourse]) {
                if (--childCount[nextCourse] == 0) dq.addLast(nextCourse);
            }
        }
        for (int child : childCount) if (child != 0) return false;
        return true;
    }


    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int n = numCourses;
        List<Integer> outputList = new ArrayList<>();
        List<Integer>[] graph = new List[numCourses];
        for (int i = 0; i < numCourses; i++) {
            graph[i] = new ArrayList<>();
        }
        int[] childCount = new int[numCourses];
        for (int[] prereq : prerequisites) {
            int a = prereq[0];
            int b = prereq[1];
            graph[b].add(a);
            childCount[a]++;
        }
        Deque<Integer> dq = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            if (childCount[i] == 0) dq.addLast(i);
        }
        while (!dq.isEmpty()) {
            int currentCourse = dq.pollFirst();
            outputList.add(currentCourse);
            for (Integer nextCourse : graph[currentCourse]) {
                if (--childCount[nextCourse] == 0) dq.addLast(nextCourse);
            }
        }
        if (outputList.size() != n) return (new int[0]);
        int[] output = new int[n];
        for (int i = 0; i < n; i++) {
            output[i] = outputList.get(i);
        }
        return output;
    }


    public List<String> findItinerary(List<List<String>> tickets) {
        HashMap<String, PriorityQueue<String>> graph = new HashMap<>();
        for (var ticket : tickets) {
            String from = ticket.get(0);
            String to = ticket.get(1);
            graph.computeIfAbsent(from, k -> new PriorityQueue<>()).add(to);
        }
        List<String> output = new ArrayList<>();
        flightDFS(graph, output, "JFK");
        return output;
    }

    public void flightDFS(HashMap<String, PriorityQueue<String>> graph,
                          List<String> output,
                          String currentPos) {
        if (currentPos == null) return;
        var pq = graph.getOrDefault(currentPos, new PriorityQueue<>());
        while (!pq.isEmpty()) {
            flightDFS(graph, output, pq.poll());
        }
        output.add(0, currentPos);
    }


    public int fib(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;
        long[][] result = {{1, 0}, {0, 1}};
        long[][] fibMatrix = {{1, 1}, {1, 0}};
        n -= 1;
        while (n > 0) {
            if (n % 2 == 1) result = multiplyMatrices(result, fibMatrix);
            fibMatrix = multiplyMatrices(fibMatrix, fibMatrix);
            n /= 2;
        }
        return (int) result[0][0];
    }

    public long[][] multiplyMatrices(long[][] a, long[][] b) {
        return new long[][]{
                {a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]},
                {a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]}
        };
    }


    public int minimumSemesters(int n, int[][] relations) {
        List<Integer>[] graph = new List[n];
        for (int i = 0; i < n; i++) {
            graph[i] = new ArrayList<>();
        }
        int[] childCount = new int[n];
        for (int[] relation : relations) {
            int a = --relation[0];
            int b = --relation[1];
            childCount[b]++;
            graph[a].add(b);
        }
        int counter = 0;
        Deque<Integer> dq = new ArrayDeque<>();
        for (int i = 0; i < n; i++) if (childCount[i] == 0) dq.addLast(i);
        while (!dq.isEmpty()) {
            counter++;
            int size = dq.size();
            for (int i = 0; i < size; i++) {
                int currentCourse = dq.pollFirst();
                List<Integer> nextCourses = graph[currentCourse];
                for (Integer nextCourse : nextCourses) {
                    if (--childCount[nextCourse] == 0) dq.addLast(nextCourse);
                }
            }
        }
        for (int i : childCount) if (i != 0) return -1;
        return counter;
    }


//    public Node cloneGraph(Node node) {
//        if (node == null) return null;
//        int root = node.val;
//        HashMap<Integer, Node> nodeGraph = new HashMap<>();
//        cloneGraph(nodeGraph, node);
//        return nodeGraph.get(root);
//    }

//    public void cloneGraph(HashMap<Integer, Node> nodeGraph, Node node) {
//        if (node == null) return;
//        int value = node.val;
//        if (nodeGraph.containsKey(value)) return;
//        nodeGraph.put(value, new Node(value));
//        List<Node> neighbours = node.neighbors;
//        for (Node neighbour : neighbours) {
//            cloneGraph(nodeGraph, neighbour);
//            nodeGraph.get(value).neighbors.add(nodeGraph.get(neighbour.val));
//        }
//    }

    public int countComponents(int n, int[][] edges) {
        if (n == 1) return 1;
        int[] parent = new int[n];
        //Start off assuming all components are independent singleton
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
        //Every time we connect a component to another, we minus 1 from our assumption
        //If they have the same parent -> they are already connected -> don't need to decrement;
        int output = n;
        for (int[] edge : edges) {
            int parentA = findParent(parent, edge[0]);
            int parentB = findParent(parent, edge[1]);
            if (parentA != parentB) {
                parent[parentB] = parentA;
                output--;
            }

        }
        return output;
    }

    public int findParent(int[] parent, int node) {
        int currentParent = parent[node];
        if (currentParent != node) return parent[node] = findParent(parent, currentParent);
        return currentParent;
    }


    public int[] findRedundantConnection(int[][] edges) {
        int n = edges.length;
        int[] parent = new int[n];
        //Start off assuming all components are independent singleton
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
        //Every time we connect a component to another, we reassign one of the parents.
        //If they have the same parent -> they are already connected -> it is a redundant connection
        int output = n;
        for (int[] edge : edges) {
            int parentA = findParent(parent, edge[0]);
            int parentB = findParent(parent, edge[1]);
            if (parentA != parentB) {
                parent[parentB] = parentA;
            } else {
                return edge;
            }
        }
        return null;
    }


    public int longestIncreasingPath(int[][] matrix) {
        int yMax = matrix.length;
        int xMax = matrix[0].length;
        int[][] dp = new int[yMax][xMax];
        int[][] parentCount = new int[yMax][xMax];
        int[][] directions = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        Deque<int[]> dq = new ArrayDeque<>();
        for (int y = 0; y < yMax; y++) {
            for (int x = 0; x < xMax; x++) {
                for (int[] direction : directions) {
                    int nextY = y + direction[0];
                    int nextX = x + direction[1];
                    if (0 <= nextY && nextY < yMax && 0 <= nextX && nextX < xMax && matrix[y][x] < matrix[nextY][nextX]) {
                        parentCount[y][x]++;
                    }
                }
                if (parentCount[y][x] == 0) {
                    dq.addLast(new int[]{y, x});
                    dp[y][x] = 1;
                }
            }
        }
        int output = 0;
        while (!dq.isEmpty()) {
            int[] position = dq.pollFirst();
            int y = position[0];
            int x = position[1];
            for (int[] direction : directions) {
                int nextY = y + direction[0];
                int nextX = x + direction[1];
                if (0 <= nextY && nextY < yMax && 0 <= nextX && nextX < xMax && matrix[y][x] > matrix[nextY][nextX]) {
                    if (--parentCount[nextY][nextX] == 0) {
                        dq.addLast(new int[]{nextY, nextX});
                        dp[nextY][nextX] = Math.max(dp[nextY][nextX], dp[y][x] + 1);
                    }
                }
            }
            output = Math.max(output, dp[y][x]);
        }
        return output;
    }


    public int maxCoins(int[] nums) {
        int n = nums.length;
        int[] balloons = new int[n + 2];
        for (int i = 0; i < n; i++) {
            balloons[i + 1] = nums[i];
        }
        balloons[0] = 1;
        balloons[n + 1] = 1;
        n = n + 2;
        long[][] dp = new long[n][n];
        for (int i = n - 3; i >= 0; i--) {
            for (int j = i + 2; j < n; j++) {
                long currentBase = balloons[i] * balloons[j];
                long currentMax = 0;
                for (int k = i + 1; k < j; k++) {
                    long currentScore = currentBase * balloons[k] + dp[i][k] + dp[k][j];
                    currentMax = Math.max(currentMax, currentScore);
                }
                dp[i][j] = currentMax;
            }
        }
        return (int) dp[0][n - 1];
    }

    public boolean isMatch(String s, String p) {
        int sLength = s.length();
        int pLength = p.length();
        char[] sChar = s.toCharArray();
        char[] pChar = p.toCharArray();
        boolean[][] dp = new boolean[sLength + 1][pLength + 1];
        dp[sLength][pLength] = true;
        for (int i = sLength - 1; i >= 0; i--) {
            char sC = sChar[i];
            for (int j = 0; j < pLength; j++) {
                char pC = pChar[j];
                if (sC == pC || pC == '?') {
                    dp[i][j] = dp[i + 1][j + 1];
                } else if (pC == '*') {
                    dp[i][j] = dp[i + 1][j] || dp[i][j + 1] || dp[i + 1][j + 1];
                } else {
                    dp[i][j] = false;
                }
            }
        }
        return dp[0][0];
    }


    public String longestPalindrome(String s) {
        int n = s.length();
        if (n == 0) return "";
        boolean[][] dp = new boolean[n][n];
        String output = "";
        for (int i = 0; i < n; i++) {
            dp[i][i] = true;
            output = s.substring(i, i + 1);
        }
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    if (j - i == 1 || dp[i + 1][j - 1]) {
                        dp[i][j] = true;
                        if (j - i + 1 > output.length()) {
                            output = s.substring(i, j + 1);
                        }
                    }
                }
            }
        }
        return output;
    }

    public int minimumSwaps(int[] nums) {
        int n = nums.length;
        if (n == 1) return 0;
        int minValue = Integer.MAX_VALUE;
        int maxValue = Integer.MIN_VALUE;
        int minIndex = -1;
        int maxIndex = -1;
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            if (num >= maxValue) {
                maxValue = num;
                maxIndex = i;
            }
            if (num < minValue) {
                minValue = num;
                minIndex = i;
            }
        }
        int output = ((n - 1) - maxIndex) + (minIndex);
        if (minIndex < maxIndex) output--;
        return output;
    }


    public List<String> mostVisitedPattern(String[] username, int[] timestamp, String[] website) {
        int n = timestamp.length;
        HashMap<String, Integer> hm = new HashMap<>();
        HashMap<String, List<String>> userHistory = new HashMap<>();
        int[][] data = new int[n][2];
        for (int i = 0; i < n; i++) {
            int[] datum = new int[]{i, timestamp[i]};
            data[i] = datum;
        }
        Arrays.sort(data, Comparator.comparingInt(a -> a[1]));
        for (int[] datum : data) {
            String userName = username[datum[0]];
            String userWebsite = website[datum[0]];
            userHistory.computeIfAbsent(userName, k -> new ArrayList<>()).add(userWebsite);
            List<String> history = userHistory.getOrDefault(userName, new ArrayList<>());
            history.add(userWebsite);
            if (history.size() >= 3) {
                String s = "";
                for (int i = userHistory.size() - 4; i < userHistory.size(); i++) {
                    s += userHistory.get(i) + " ";
                }
                s = s.trim();
                hm.merge(s, 1, Integer::sum);
            }
            userHistory.put(userName, history);
        }
        for (String k : hm.keySet()) System.out.println(k);
        String key = Collections.max(hm.entrySet(), Map.Entry.comparingByValue()).getKey();
        String[] keySplit = key.split(" ");
        List<String> output = new ArrayList<>();
        for (String s : keySplit) output.add(s);
        return output;
    }


    public NodeCopy copyRandomBinaryTree(Node root) {
        if (root == null) return null;
        HashMap<Node, NodeCopy> nodeMap = new HashMap<>();
        return copyRandomBinaryTree(nodeMap, root);
    }

    private NodeCopy copyRandomBinaryTree(HashMap<Node, NodeCopy> nodeMap, Node root) {
        if (root == null) return null;
        if (nodeMap.containsKey(root)) return nodeMap.get(root);
        NodeCopy copy = new NodeCopy(root.val);
        nodeMap.put(root, copy);
        copy.left = copyRandomBinaryTree(nodeMap, root.left);
        copy.right = copyRandomBinaryTree(nodeMap, root.right);
        copy.random = copyRandomBinaryTree(nodeMap, root.random);
        return copy;
    }

    public int minOperations(int[] nums) {
        HashMap<Integer, Integer> hm = new HashMap<>();
        for (int i : nums) hm.merge(i, 1, Integer::sum);
        int output = 0;
        for (Integer value : hm.values()) {
            if (value <= 1) return -1;
            output += value / 3;
            value = value % 3;
            if (value > 0) output++;
        }
        return output;
    }

    public String pushDominoes(String dominoes) {
        int n = dominoes.length();
        if (n == 1) return dominoes;
        char[] dominoChars = dominoes.toCharArray();
        StringBuilder sb = new StringBuilder();
        char firstChar = 'a';
        int currentLength = 0;
        for (int i = 0; i <= n; i++) {
            if (i == n) {
                for (int j = 0; j < currentLength; j++) sb.append(firstChar);
            } else {
                char dominoChar = dominoChars[i];
                switch (dominoChar) {
                    case '.' -> {
                        if (currentLength == 0) {
                            firstChar = dominoChar;
                        }
                        currentLength++;
                    }
                    case 'L' -> {
                        if (currentLength == 0) {
                            sb.append(dominoChar);
                        } else {
                            currentLength++;
                            if (firstChar == '.') {
                                for (int j = 0; j < currentLength; j++) sb.append(dominoChar);
                            } else {
                                int count = currentLength / 2;
                                boolean odd = (currentLength % 2 != 0);
                                for (int j = 0; j < count; j++) sb.append('R');
                                if (odd) sb.append('.');
                                for (int j = 0; j < count; j++) sb.append('L');
                            }
                            currentLength = 0;
                        }
                    }
                    case 'R' -> {
                        if (currentLength == 0) {
                            firstChar = dominoChar;
                            currentLength++;
                        } else {
                            if (firstChar == 'R') {
                                for (int j = 0; j < currentLength; j++) sb.append(dominoChar);
                                currentLength = 1;
                            } else {
                                for (int j = 0; j < currentLength; j++) sb.append('.');
                                currentLength = 1;
                                firstChar = dominoChar;
                            }
                        }
                    }
                }
            }
        }
        return sb.toString();
    }


//    public static int numberOfWays(String s, String t, long k) {
//        String zString = t + '$' + s.substring(1) + s.substring(0, s.length()-1);
//        int[] zArray = computeZArray(zString);
//        int moves = 0;
//        for (int i = s.length() + 1; i < s.length() * 2; i++) {
//            if (zArray[i] >= s.length()) {
//                moves++;
//            }
//        }
//
//        if (moves == 0 && zArray[s.length()] < s.length()) {
//            return 0;
//        }
//
//        BigInteger mod = BigInteger.valueOf((long) (1e9 + 7));
//        BigInteger a = BigInteger.valueOf(s.length() - 1)
//                .modPow(BigInteger.valueOf(k), mod)
//                .add(BigInteger.valueOf(k % 2 == 0 ? s.length() - 1 : 1 - s.length()))
//                .multiply(BigInteger.valueOf(s.length()).modInverse(mod))
//                .mod(mod);
//
//        BigInteger b = a.add(BigInteger.valueOf(k % 2 == 0 ? -1 : 1));
//
//        BigInteger total = zArray[s.length()] >= s.length() ? a : BigInteger.ZERO;
//        total = total.add(b.multiply(BigInteger.valueOf(moves)));
//
//        return total.mod(mod).intValue();
//    }


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

    int MOD = 1_000_000_007;

    public int checkRecord(int n) {
        long[] piMatrix = new long[]{1, 1, 0, 1, 0, 0};
        long[][] transitionMatrix = new long[][]{
                {1, 1, 0, 1, 0, 0},
                {1, 0, 1, 1, 0, 0},
                {1, 0, 0, 1, 0, 0},
                {0, 0, 0, 1, 1, 0},
                {0, 0, 0, 1, 0, 1},
                {0, 0, 0, 1, 0, 0}
        };
        transitionMatrix = matrixPower(transitionMatrix, n - 1);
        long[] outputMatrix = new long[6];
        for (int j = 0; j < 6; j++) {
            for (int k = 0; k < 6; k++) {
                outputMatrix[j] = (outputMatrix[j] + piMatrix[k] * transitionMatrix[k][j]) % MOD;
            }
        }
        long output = 0;
        for (long l : outputMatrix) output += l;
        return (int) (output % MOD);
    }


    public int minCameraCover(TreeNode root) {
        int[] result = treeCameraDFS(root);
        return Math.min(result[1], result[2]);
    }

    public int[] treeCameraDFS(TreeNode node) {
        if (node == null) {
            return new int[]{0, 0, 10000};
        }
        int[] left = treeCameraDFS(node.left);
        int[] right = treeCameraDFS(node.right);
        int notCovered = left[1] + right[1];
        int coveredByChild = Math.min(left[2] + Math.min(right[1], right[2]), right[2] + Math.min(left[1], left[2]));
        int cameraHere = 1 + Math.min(left[0], Math.min(left[1], left[2])) + Math.min(right[0], Math.min(right[1], right[2]));
        return new int[]{notCovered, coveredByChild, cameraHere};
    }


    public int maxRotateFunction(int[] nums) {
        int n = nums.length;
        if (n == 1) return 0;
        long sum = 0l;
        long rotateSum = 0;
        for (int i = 0; i < n; i++) {
            sum += nums[i];
            rotateSum += i * nums[i];
        }
        int k = 0;
        long output = rotateSum;
        while (k < n) {
            k++;
            rotateSum = rotateSum + sum - (n * (nums[n - k]));
            output = Math.max(output, rotateSum);
        }
        return (int) output;
    }


    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        List<Integer> output = new ArrayList<>();
        if (n == 1) {
            output.add(0);
            return output;
        }
        List<Integer>[] graph = new List[n];
        for (int i = 0; i < n; i++) {
            graph[i] = new ArrayList<>();
        }

        int[] adjList = new int[n];
        for (int[] edge : edges) {
            int a = edge[0];
            int b = edge[1];
            graph[a].add(b);
            graph[b].add(a);
            adjList[a]++;
            adjList[b]++;
        }
        Deque<Integer> dq = new ArrayDeque<>();
        for (int i = 0; i < n; i++) if (adjList[i] == 1) dq.addLast(i);
        while (!dq.isEmpty()) {
            output = new ArrayList<>();
            int size = dq.size();
            for (int i = 0; i < size; i++) {
                int currentNode = dq.pollFirst();
                output.add(currentNode);
                for (int neighbor : graph[currentNode]) {
                    if (--adjList[neighbor] == 1) {
                        dq.addLast(neighbor);
                    }
                }
            }
        }

        return output;
    }


    public int maxStarSum(int[] vals, int[][] edges, int k) {
        int n = vals.length;
        List<Integer>[] graph = new List[n];
        for (int i = 0; i < n; i++) {
            graph[i] = new ArrayList<>();
        }
        for (int[] edge : edges) {
            int a = edge[0];
            int b = edge[1];
            graph[a].add(b);
            graph[b].add(a);
        }
        int maxSum = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());
            for (int neighbor : graph[i]) {
                pq.offer(vals[neighbor]);
            }
            int currentSum = vals[i];
            for (int j = 0; j < k && !pq.isEmpty(); j++) {
                int topValue = pq.poll();
                if (topValue > 0) {
                    currentSum += topValue;
                } else {
                    break;
                }
            }
            maxSum = Math.max(maxSum, currentSum);
        }
        return maxSum;
    }


    public int longestCycle(int[] edges) {
        int n = edges.length;
        int[] degree = new int[n];
        BitSet mask = new BitSet();
        for (int edge : edges) if (edge != -1) degree[edge]++;
        Deque<Integer> dq = new ArrayDeque<>();
        for (int i = 0; i < n; i++) if (degree[i] == 0) dq.addLast(i);
        while (!dq.isEmpty()) {
            int currentNode = dq.pollFirst();
            mask.set(currentNode);
            if (edges[currentNode] == -1) continue;
            if (--degree[edges[currentNode]] == 0) dq.addLast(edges[currentNode]);
        }
        int output = -1;
        for (int i = 0; i < n; i++) {
            if (!mask.get(i)) {
                mask.set(i);
                int nextNode = edges[i];
                int count = 1;
                while (nextNode != i) {
                    mask.set(nextNode);
                    count++;
                    nextNode = edges[nextNode];
                }
                output = Math.max(output, count);
            }
        }
        return output;
    }

    public int minimumTimeRequired(int[] jobs, int k) {
        int left = 0;
        int right = 0;
        for (int job : jobs) {
            left = Math.max(left, job);
            right += job;
        }
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (canFinish(jobs, k, mid)) {
                right = mid; // Try for a smaller maximum time
            } else {
                left = mid + 1; // Try for a larger maximum time
            }
        }
        return left;
    }

    private boolean canFinish(int[] jobs, int k, int maxWork) {
        int[] bins = new int[k];
        return backtrack(jobs, 0, bins, maxWork);
    }

    private boolean backtrack(int[] jobs, int jobIndex, int[] bins, int maxWork) {
        if (jobIndex == jobs.length) {
            return true;
        }
        int currentJob = jobs[jobIndex];
        for (int i = 0; i < bins.length; i++) {
            if (bins[i] + currentJob <= maxWork) {
                bins[i] += currentJob;
                if (backtrack(jobs, jobIndex + 1, bins, maxWork)) {
                    return true;
                }
                bins[i] -= currentJob;
            }
            if (bins[i] == 0) {
                break;
            }
        }
        return false;
    }

    int maxCardinality = 0;

    public int minKBitFlips(int[] nums, int k) {
        int n = nums.length;
        BitSet mask = new BitSet(n + k + 1);
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (nums[i] == 1) mask.set(i);
        }
        for (int i = 0; i < n; i++) {
            if (!mask.get(i)) {
                mask.flip(i, i + k);
                count++;
            }
        }
        int cardinality = mask.cardinality();
        if (cardinality == n) return count;
        return -1;
    }

    public int maxLength(List<String> arr) {
        List<Integer> maskList = new ArrayList<>();
        for (String s : arr) {
            char[] sChar = s.toCharArray();
            int mask = 0;
            boolean add = true;
            for (char c : sChar) {
                int charInt = c - 'a';
                if ((mask & (1 << charInt)) != 0) {
                    add = false;
                    break;
                }
                mask |= (1 << charInt);
            }
            if (add) maskList.add(mask);
        }
        maxLength(maskList, 0, 0);
        return maxCardinality;
    }

    public void maxLength(List<Integer> words, int index, int mask) {
        if (index == words.size()) {
            maxCardinality = Math.max(maxCardinality, Integer.bitCount(mask));
            return;
        }
        for (int i = index; i < words.size(); i++) {
            int word = words.get(i);
            if ((word & mask) == 0) {
                int nextMask = word | mask;
                maxLength(words, i + 1, nextMask);
            }
        }
        maxCardinality = Math.max(maxCardinality, Integer.bitCount(mask));
    }


    public List<Integer> findAllPeople(int n, int[][] meetings, int firstPerson) {
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        parent[firstPerson] = 0;
        Arrays.sort(meetings, Comparator.comparingInt(a -> a[2]));
        int index = 0;
        int m = meetings.length;
        while (index < m) {
            int currentTime = meetings[index][2];
            List<int[]> currentTimeMeetings = new ArrayList<>();
            while (index < m && meetings[index][2] == currentTime) {
                currentTimeMeetings.add(meetings[index]);
                index++;
            }
            // Union for all the meetings in the current time
            for (int[] meeting : currentTimeMeetings) {
                unionMeeting(parent, meeting[0], meeting[1]);
            }
            // If the people do not connect to 0, or if there was a meeting with 0, and
            // the people do not connect to the root of 0,
            // then the meeting was essentially pointless.
            // So we reverse it, and pretend it never happened
            for (int[] meeting : currentTimeMeetings) {
                if (findMeeting(parent, meeting[0]) != findMeeting(parent, 0)) {
                    parent[meeting[0]] = meeting[0];
                    parent[meeting[1]] = meeting[1];
                }
            }
        }
        List<Integer> output = new ArrayList<>();
        for (int j = 0; j < n; j++) {
            if (findMeeting(parent, j) == findMeeting(parent, 0)) {
                output.add(j);
            }
        }

        return output;
    }

    public int findMeeting(int[] parent, int node) {
        if (node == parent[node]) return node;
        return parent[node] = findMeeting(parent, parent[node]);
    }

    public void unionMeeting(int[] parent, int a, int b) {
        int parentA = findMeeting(parent, a);
        int parentB = findMeeting(parent, b);
        if (parentA != parentB) {
            parent[parentB] = parentA;
        }
    }


    public List<String> findAllRecipes(String[] recipes,
                                       List<List<String>> ingredients,
                                       String[] supplies) {
        HashMap<String, List<String>> graphHM = new HashMap<>();
        HashMap<String, Integer> degreeHM = new HashMap<>();
        List<String> output = new ArrayList<>();
        Deque<String> dq = new ArrayDeque<>(Arrays.asList(supplies));
        int n = recipes.length;
        for (int i = 0; i < n; i++) {
            String recipe = recipes[i];
            var ingredientList = ingredients.get(i);
            for (String ingredient : ingredientList) {
                degreeHM.merge(recipe, 1, Integer::sum);
                graphHM.computeIfAbsent(ingredient, k -> new ArrayList<>()).add(recipe);
            }
        }
        while (!dq.isEmpty()) {
            String currentFood = dq.pollFirst();
            var nextFoods = graphHM.getOrDefault(currentFood, new ArrayList<>());
            for (String nextFood : nextFoods) {
                if (degreeHM.merge(nextFood, -1, Integer::sum) == 0) {
                    dq.addLast(nextFood);
                    output.add(nextFood);
                }
            }
        }
        return output;
    }

    public long maximumImportance(int n, int[][] roads) {
        long[] degree = new long[n];
        for (int[] road : roads) {
            degree[road[0]]++;
            degree[road[1]]++;
        }
        Arrays.sort(degree);
        long output = 0;
        for (int i = 0; i < n; i++) output += (i + 1) * degree[i];
        return output;
    }


    public String[] reorderLogFiles(String[] logs) {
        List<String> digitLogs = new ArrayList<>();
        List<String> letterLogs = new ArrayList<>();
        for (String log : logs) {
            String[] parts = log.split(" ", 2);
            if (Character.isDigit(parts[1].charAt(0))) {
                digitLogs.add(log);
            } else {
                letterLogs.add(log);
            }
        }
        Comparator<String> letterComparator = new Comparator<String>() {
            @Override
            public int compare(String log1, String log2) {
                String[] split1 = log1.split(" ", 2);
                String[] split2 = log2.split(" ", 2);
                int cmp = split1[1].compareTo(split2[1]);
                if (cmp != 0) {
                    return cmp;
                }
                return split1[0].compareTo(split2[0]);
            }
        };
        Collections.sort(letterLogs, letterComparator);
        List<String> result = new ArrayList<>(letterLogs);
        result.addAll(digitLogs);
        return result.toArray(new String[0]);
    }

    public String mostCommonWord(String paragraph, String[] banned) {
        Set<String> set = new HashSet<>();
        paragraph = paragraph.toLowerCase().trim().replaceAll("[\\\\p{Punct}]", "");
        for (String s : banned) set.add(s);
        HashMap<String, Integer> hm = new HashMap<>();
        String[] words = paragraph.split(" ");
        for (String word : words) {
            if (!set.contains(word)) hm.merge(word, 1, Integer::sum);
        }
        return Collections.max(hm.entrySet(), Map.Entry.comparingByValue()).getKey();
    }


    public List<Integer> partitionLabels(String s) {
        int n = s.length();
        char[] sChar = s.toCharArray();
        int[] charIndex = new int[26];
        for (int i = 0; i < n; i++) charIndex[sChar[i] - 'a'] = i;
        List<Integer> output = new ArrayList<>();
        int start = 0;
        while (start < n) {
            int left = start;
            int right = charIndex[sChar[start] - 'a'];
            while (left < right) {
                left++;
                right = Math.max(right, charIndex[sChar[left] - 'a']);
            }
            output.add(right - start + 1);
            start = right + 1;
        }
        return output;
    }


    int timeCounter = 0;

    public List<List<Integer>> criticalConnections(int n, List<List<Integer>> connections) {
        List<List<Integer>> graph = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList<>());
        }
        for (List<Integer> connection : connections) {
            graph.get(connection.get(0)).add(connection.get(1));
            graph.get(connection.get(1)).add(connection.get(0));
        }
        int[] rank = new int[n];
        int[] low = new int[n];
        Arrays.fill(rank, -1);
        List<List<Integer>> output = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            // Node i is not visited yet
            if (rank[i] == -1) {
                tarjanDFS(i, -1, graph, rank, low, output);
            }
        }
        return output;
    }

    private void tarjanDFS(int currentNode,
                           int parent,
                           List<List<Integer>> graph,
                           int[] rank,
                           int[] low,
                           List<List<Integer>> output) {
        ++timeCounter;
        rank[currentNode] = timeCounter;
        low[currentNode] = timeCounter;
        for (int nextNode : graph.get(currentNode)) {
            if (nextNode == parent) continue;
            // If next node is not visited yet
            if (rank[nextNode] == -1) {
                tarjanDFS(nextNode, currentNode, graph, rank, low, output);
                // Update low[u] considering the subtree rooted with v
                low[currentNode] = Math.min(low[currentNode], low[nextNode]);
                // If the lowest vertex reachable from subtree under the next node is
                // below the currentNode in DFS tree, then this is a critical connection
                if (low[nextNode] > rank[currentNode]) {
                    output.add(Arrays.asList(currentNode, nextNode));
                }
            } else {
                // Update low[currentNode] considering the back edge
                low[currentNode] = Math.min(low[currentNode], rank[nextNode]);
            }
        }
    }


    int[][] gameGrid;
    int xBound;
    int yBound;
    int[][] memo;

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        gameGrid = obstacleGrid;
        xBound = gameGrid[0].length;
        yBound = gameGrid.length;
        memo = new int[yBound][xBound];
        for (int i = 0; i < yBound; i++) {
            for (int j = 0; j < xBound; j++) {
                memo[i][j] = -1;
            }
        }
        return uniquePath(0, 0);
    }

    public int uniquePath(int y, int x) {
        if (x >= xBound || y >= yBound) return 0;
        if (gameGrid[y][x] == 1) return 0;
        if (x == xBound - 1 && y == yBound - 1) {
            return 1;
        }
        if (memo[y][x] != -1) return memo[y][x];
        int down = uniquePath(y + 1, x);
        int right = uniquePath(y, x + 1);
        memo[y][x] = down + right;
        return memo[y][x];
    }


    public int findMaxLength(int[] nums) {
        int n = nums.length;
        int[] firstIndex = new int[2 * n + 1];
        Arrays.fill(firstIndex, Integer.MIN_VALUE);
        int output = 0;
        int sum = n;
        firstIndex[n] = -1;
        for (int i = 0; i < n; i++) {
            sum = sum + (nums[i] * 2 - 1);
            if (firstIndex[sum] != Integer.MIN_VALUE) {
                output = Math.max(output, i - firstIndex[sum]);
            } else {
                firstIndex[sum] = i;
            }
        }
        return output;
    }


    public long[][] multiply(long[][] a, long[][] b) {
        int n = a.length;
        long[][] result = new long[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    result[i][j] = (result[i][j] + a[i][k] * b[k][j]) % MOD;
                }
            }
        }
        return result;
    }

    public long[][] matrixPower(long[][] a, int n) {
        int size = a.length;
        long[][] result = new long[size][size];
        for (int i = 0; i < size; i++) {
            result[i][i] = 1;
        }
        long[][] base = a;
        while (n > 0) {
            if ((n & 1) == 1) {
                result = multiply(result, base);
            }
            base = multiply(base, base);
            n >>= 1;
        }
        return result;
    }


    public int numberOfWays(int startPos, int endPos, int k) {
        int MOD = 1_000_000_007;
        int distanceDifference = Math.abs(endPos - startPos);
        int maxDistanceDifference = k;
        if (distanceDifference > maxDistanceDifference) return 0;
        long[][] dp = new long[k + 1][maxDistanceDifference + 1];
        dp[0][0] = 1;
        for (int i = 1; i <= k; i++) {
            for (int j = 0; j <= maxDistanceDifference; j++) {
                if (j == 0) {
                    dp[i][j] = (dp[i - 1][j + 1] * 2) % MOD;
                } else if (j == maxDistanceDifference) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j + 1]) % MOD;
                }
            }
        }
        return (int) dp[k][distanceDifference];
    }

    public int[] findBuildings(int[] heights) {
        int n = heights.length;
        int minHeight = Integer.MIN_VALUE;
        int index = n - 1;
        for (int i = n - 1; i >= 0; i--) {
            int height = heights[i];
            if (height > minHeight) {
                minHeight = height;
                heights[index--] = i;
            }
        }
        return Arrays.copyOfRange(heights, index + 1, heights.length);
    }

    public int maximumProfit(int[] present, int[] future, int budget) {
        int n = present.length;
        List<int[]> stockList = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int currentPrice = present[i];
            int futurePrice = future[i];
            if (futurePrice > currentPrice) {
                stockList.add(new int[]{currentPrice, futurePrice});
            }
        }
        n = stockList.size();
        if (n == 0) {
            return 0;
        }
        int[] dp = new int[budget + 1];
        for (int i = 0; i < n; i++) {
            int cost = stockList.get(i)[0];
            int profit = stockList.get(i)[1] - cost;
            for (int j = budget; j >= cost; j--) {
                dp[j] = Math.max(dp[j], dp[j - cost] + profit);
            }
        }
        return dp[budget];
    }


    class ArrayReader {
        int get(int i) {
            return i;
        }

        ;
    }

    public int search(ArrayReader reader, int target) {
        int left = 0;
        int right = 10000;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (reader.get(mid) == target) return mid;
            else if (reader.get(mid) < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }

    int treeDiameter = 0;

    public int diameterOfBinaryTree(TreeNode root) {
        diameterOfTree(root);
        return treeDiameter - 1;
    }

    public int diameterOfTree(TreeNode root) {
        if (root == null) return 0;
        int left = diameterOfTree(root.left);
        int right = diameterOfTree(root.right);
        treeDiameter = Math.max(treeDiameter, left + right + 1);
        return Math.max(left + 1, right + 1);
    }


    public int maxNumberOfFamilies(int n, int[][] reservedSeats) {
        HashMap<Integer, Integer> hm = new HashMap<>();
        for (int[] reserved : reservedSeats) {
            int row = reserved[0] - 1;
            int col = reserved[1] - 1;
            hm.put(row, (hm.getOrDefault(row, 0) | (1 << col)));
        }
        int output = 0;
        int leftMask = Integer.parseInt("0111100000", 2);
        int middleMask = Integer.parseInt("0001111000", 2);
        int rightMask = Integer.parseInt("0000011110", 2);
        for (var entry : hm.entrySet()) {
            int count = 0;
            int mask = entry.getValue();
            if ((leftMask & mask) == 0) count++;
            if ((rightMask & mask) == 0) count++;
            if (count == 0 && (middleMask & mask) == 0) count++;
            output += count;
        }
        return (2 * (n - hm.size()) + output);
    }


    public int oddEvenJumps(int[] arr) {
        // boolean[] = {Even, Odd}
        TreeMap<Integer, boolean[]> tm = new TreeMap<>();
        int n = arr.length;
        int last = arr[n - 1];
        tm.put(last, new boolean[]{true, true});
        int output = 1;
        for (int i = n - 2; i >= 0; i--) {
            int currentPosition = arr[i];
            boolean even = false;
            boolean odd = false;
            Integer ceilingKey = tm.ceilingKey(currentPosition);
            Integer floorKey = tm.floorKey(currentPosition);
            if (ceilingKey != null) odd = tm.get(ceilingKey)[0];
            if (floorKey != null) even = tm.get(floorKey)[1];
            if (odd) output++;
            tm.put(currentPosition, new boolean[]{even, odd});
        }
        return output;
    }


    public int mergeStones(int[] stones, int k) {
        int n = stones.length;
        if ((n - 1) % (k - 1) != 0) return -1;
        int[] prefixSum = new int[n + 1];
        for (int i = 0; i < n; i++) {
            prefixSum[i + 1] = prefixSum[i] + stones[i];
        }
        int[][] dp = new int[n][n];
        for (int len = k; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                dp[i][j] = Integer.MAX_VALUE;
                for (int m = i; m < j; m += k - 1) {
                    dp[i][j] = Math.min(dp[i][j], dp[i][m] + dp[m + 1][j]);
                }
                if ((j - i) % (k - 1) == 0) {
                    dp[i][j] += prefixSum[j + 1] - prefixSum[i];
                }
            }
        }

        return dp[0][n - 1];
    }


    public long wonderfulSubstrings(String word) {
        Map<Integer, Integer> prefixMasks = new HashMap<>();
        prefixMasks.put(0, 1);
        long wSubstrings = 0;
        int currentMask = 0;
        for (char c : word.toCharArray()) {
            int charIndex = c - 'a';
            currentMask ^= (1 << charIndex);
            wSubstrings += prefixMasks.getOrDefault(currentMask, 0);
            for (int i = 0; i < 10; i++) {
                int maskWithOneBitFlipped = currentMask ^ (1 << i);
                wSubstrings += prefixMasks.getOrDefault(maskWithOneBitFlipped, 0);
            }
            prefixMasks.put(currentMask, prefixMasks.getOrDefault(currentMask, 0) + 1);
        }
        return wSubstrings;
    }

    public int longestNiceSubarray(int[] nums) {
        int n = nums.length;
        int left = 0;
        int mask = 0;
        int output = 0;
        for (int right = 0; right < n; right++) {
            int num = nums[right];
            while ((mask & num) != 0) mask ^= nums[left++];
            mask |= num;
            output = Math.max(output, right - left + 1);
        }
        return output;
    }

    HashMap<Integer, Integer> minDayMap = new HashMap<>();

    public int minDays(int n) {
        if (n <= 2) return n;
        if (minDayMap.containsKey(n)) return minDayMap.get(n);
        minDayMap.put(n, 1 + Math.min(n % 2 + minDays(n / 2), n % 3 + minDays(n / 3)));
        return minDayMap.get(n);
    }

    public int minimumCoins(int[] prices) {
        int n = prices.length;
        int[] dp = new int[n + 2];
        for (int i = n - 1; i >= 0; i--) {
            int currentFruit = prices[i];
            dp[i] = currentFruit + Math.min(dp[i + 1], dp[i + 2]);
        }
        return dp[0];
    }


    public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
        if (maxChoosableInteger >= desiredTotal) return true;
        if ((maxChoosableInteger * (maxChoosableInteger + 1)) / 2 < desiredTotal) return false;
        Boolean[] dp = new Boolean[1 << maxChoosableInteger];
        return canIWin(dp, 0, maxChoosableInteger, desiredTotal);
    }

    private boolean canIWin(Boolean[] dp, int mask, int maxChoose, int currentSum) {
        if (dp[mask] != null) return dp[mask];
        for (int i = 1; i <= maxChoose; i++) {
            if ((mask & (1 << (i - 1))) == 0) {
                if (currentSum - i <= 0 || !canIWin(dp, mask | (1 << (i - 1)), maxChoose, currentSum - i)) {
                    return dp[mask] = true;
                }
            }
        }
        return dp[mask] = false;
    }


    public int numOfSubarrays(int[] arr) {
        int MOD = 1_000_000_007;
        int oddSumCount = 0;
        int evenSumCount = 1;
        int currentPrefixSum = 0;
        int result = 0;
        for (int num : arr) {
            currentPrefixSum += num;
            if ((currentPrefixSum & 1) == 0) {
                evenSumCount++;
                result = (result + oddSumCount) % MOD;
            } else {
                oddSumCount++;
                result = (result + evenSumCount) % MOD;
            }
        }
        return result;
    }


    public void reverseWords(char[] s) {
        reverse(s, 0, s.length - 1);
        int start = 0;
        for (int end = 0; end <= s.length; end++) {
            if (end == s.length || s[end] == ' ') {
                reverse(s, start, end - 1);
                start = end + 1;
            }
        }
    }

    private void reverse(char[] s, int left, int right) {
        while (left < right) {
            char temp = s[left];
            s[left] = s[right];
            s[right] = temp;
            left++;
            right--;
        }
    }


    public int strangePrinter(String s) {
        if (s.length() == 1) return 1;
        StringBuilder sb = new StringBuilder();
        char prior = '.';
        for (char c : s.toCharArray()) {
            if (c != prior) sb.append(c);
            prior = c;
        }
        char[] sChar = sb.toString().toCharArray();
        int n = sChar.length;
        int[][] dp = new int[n][n];
        for (int i = 0; i < n; i++) dp[i][i] = 1;
        for (int length = 2; length <= n; length++) {
            for (int i = 0; i <= n - length; i++) {
                int j = i + length - 1;
                dp[i][j] = length;
                for (int k = i; k < j; k++) {
                    int currentSplit = dp[i][k] + dp[k + 1][j];
                    if (sChar[k] == sChar[j]) {
                        currentSplit--;
                    }
                    dp[i][j] = Math.min(dp[i][j], currentSplit);
                }
            }
        }
        return dp[0][n - 1];
    }

    public int maxA(int n) {
        int[] dp = new int[n + 1 + 5];
        for (int i = 0; i < n; i++) {
            dp[i] = Math.max(dp[i], dp[i - 1] + 1);
            int current = dp[i];
            dp[i + 3] = Math.max(dp[i + 3], current * 2);
            dp[i + 4] = Math.max(dp[i + 4], current * 3);
            dp[i + 5] = Math.max(dp[i + 5], current * 4);
        }
        return dp[n];
    }

    public int longestValidParentheses(String s) {
        int n = s.length();
        char[] sChar = s.toCharArray();
        int output = 0;
        Integer[] indexDP = new Integer[n];
        int currentCount = 0;
        for (int i = 0; i < n; i++) {
            char c = sChar[i];
            if (c == '(') {
                if (indexDP[currentCount] == null) {
                    indexDP[currentCount] = i;
                }
                currentCount++;
            } else {
                indexDP[currentCount] = null;
                currentCount--;
                if (currentCount >= 0) {
                    int length = i - indexDP[currentCount] + 1;
                    output = Math.max(output, length);
                } else {
                    currentCount = 0;
                }
            }
        }
        return output;
    }

    public int numberOfArrays(String s, int k) {
        int n = s.length();
        int MOD = 1_000_000_007;
        char[] sChar = s.toCharArray();
        long[] dp = new long[n + 1];
        dp[n] = 1;
        for (int i = n - 1; i >= 0; i--) {
            if (sChar[i] == '0') continue;
            long number = 0;
            for (int j = i; j < n; j++) {
                number = number * 10 + (sChar[j] - '0');
                if (number > k) break;
                dp[i] = (dp[i] + dp[j + 1]) % MOD;
            }
        }

        return (int) dp[0];
    }

    public boolean winnerSquareGame(int n) {
        // dp[i] == Can a player win, starting at position i
        // DP: From position i, can I force a player into a position where they cannot win?
        boolean[] dp = new boolean[n + 1];
        for (int i = 1; i <= n; i++) {
            boolean flag = false;
            int k = 1;
            int position = i - (k * k);
            while (position >= 0) {
                if (dp[position] == false) {
                    flag = true;
                    break;
                }
                k++;
                position = i - (k * k);
            }
            dp[i] = flag;
        }
        return dp[n];
    }

    public long minCost(int[] nums, int[] costs) {
        int n = nums.length;
        long[] dp = new long[n];
        Arrays.fill(dp, Long.MAX_VALUE);
        dp[0] = 0;
        Deque<Integer> downStack = new ArrayDeque<>();
        Deque<Integer> upStack = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            long l = nums[i];
            while (!downStack.isEmpty() && l >= nums[downStack.peekLast()]) {
                int downIndex = downStack.pollLast();
                dp[i] = Math.min(dp[i], dp[downIndex] + costs[i]);
            }
            while (!upStack.isEmpty() && l < nums[upStack.peekLast()]) {
                int upIndex = upStack.pollLast();
                dp[i] = Math.min(dp[i], dp[upIndex] + costs[i]);
            }
            downStack.addLast(i);
            upStack.addLast(i);
        }
        return dp[n - 1];
    }


    public int minDifficulty(int[] jobDifficulty, int d) {
        int n = jobDifficulty.length;
        int[][] dp = new int[d + 1][n + 1];
        for (int[] p : dp) Arrays.fill(p, Integer.MAX_VALUE);
        dp[0][n] = 0;
        for (int day = 1; day <= d; day++) {
            for (int i = 0; i < n; i++) {
                int currentDifficulty = jobDifficulty[i];
                for (int j = i; j < n; j++) {
                    currentDifficulty = Math.max(currentDifficulty, jobDifficulty[j]);
                    if (dp[day - 1][j + 1] != Integer.MAX_VALUE) {
                        dp[day][i] = Math.min(dp[day][i], currentDifficulty + dp[day - 1][j + 1]);
                    }
                }
            }
        }
        int output = dp[d][0];
        return (output == Integer.MAX_VALUE) ? -1 : output;
    }

    public int numWays(String[] words, String target) {
        int MOD = 1_000_000_007;
        int t = target.length();
        int n = words[0].length();
        int[][] charFrequency = new int[n][26];
        for (String word : words) {
            char[] wordChar = word.toCharArray();
            for (int i = 0; i < wordChar.length; i++) charFrequency[i][wordChar[i] - 'a']++;
        }
        long[] dp = new long[t + 1];
        dp[t] = 1;
        char[] targetChar = target.toCharArray();
        for (int i = n - 1; i >= 0; i--) {
            for (int j = 0; j < t; j++) {
                dp[j] = (dp[j] + charFrequency[i][targetChar[j] - 'a'] * dp[j + 1]) % MOD;
            }
        }
        return (int) dp[0];
    }

    public boolean checkRecord(String s) {
        int aCount = 0;
        int lCount = 0;
        for (char c : s.toCharArray()) {
            if (c == 'P') {
                lCount = 0;
            } else if (c == 'A') {
                aCount++;
                lCount = 0;
                if (aCount == 2) return false;
            } else {
                lCount++;
                if (lCount == 3) return false;
            }
        }
        return true;
    }

    public boolean checkRecord2(String s) {
        int aCount = 0;
        int lCount = 0;
        for (char c : s.toCharArray()) {
            if (c == 'P') {
                lCount = 0;
            } else if (c == 'A') {
                aCount++;
                lCount = 0;
                if (aCount == 2) return false;
            } else {
                lCount++;
                if (lCount == 3) return false;
            }
        }
        return true;
    }


    public int networkDelayTime(int[][] times, int n, int k) {
        int[][] graph = new int[n][n];
        for (int[] g : graph) Arrays.fill(g, -1);
        for (int[] time : times) {
            int a = time[0] - 1;
            int b = time[1] - 1;
            int c = time[2];
            graph[a][b] = c;
        }
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
        pq.offer(new int[]{k - 1, 0});
        int[] minTime = new int[n];
        Arrays.fill(minTime, Integer.MAX_VALUE);
        while (!pq.isEmpty()) {
            int[] current = pq.poll();
            int to = current[0];
            int time = current[1];
            if (minTime[to] <= time) continue;
            minTime[to] = time;
            int[] nextTo = graph[to];
            for (int i = 0; i < n; i++) {
                int next = nextTo[i];
                if (next != -1 && minTime[i] > time + next) {
                    pq.offer(new int[]{i, next});
                }
            }
        }
        int output = Integer.MIN_VALUE;
        for (int i : minTime) {
            if (i == Integer.MIN_VALUE) return -1;
            output = Math.max(output, i);
        }
        return output;
    }

    public int maxDistance(List<List<Integer>> arrays) {
        int output = 0;
        List<Integer> first = arrays.get(0);
        int min = first.get(0);
        int max = first.get(first.size() - 1);
        int n = arrays.size();
        for (int i = 1; i < n; i++) {
            var v = arrays.get(i);
            int currentMin = v.get(0);
            int currentMax = v.get(v.size() - 1);
            output = Math.max(output, currentMax - min);
            output = Math.max(output, max - currentMin);
            max = Math.max(max, currentMax);
            min = Math.min(min, currentMin);
        }
        return output;
    }


    private static final Character[][] STROBOS = new Character[][]{
            {'0', '0'},
            {'1', '1'},
            {'6', '9'},
            {'8', '8'},
            {'9', '6'}
    };
    private static final Character[] STROBO_SINGLETON = new Character[]{'0', '1', '8'};

    public List<String> findStrobogrammatic(int n) {
        return findStrobogrammatic(n, n);
    }

    private List<String> findStrobogrammatic(int currentLength, int targetLength) {
        if (currentLength == 0) {
            return new ArrayList<>(Arrays.asList(""));
        }
        if (currentLength == 1) {
            return new ArrayList<>(Arrays.asList("0", "1", "8"));
        }

        List<String> prev = findStrobogrammatic(currentLength - 2, targetLength);
        List<String> result = new ArrayList<>();

        for (String s : prev) {
            for (Character[] pair : STROBOS) {
                if (currentLength == targetLength && pair[0] == '0') {
                    continue; // Skip leading zeros
                }
                result.add(pair[0] + s + pair[1]);
            }
        }

        return result;
    }



    private int cherryDPGet(int y, int x1, int x2, int[][][] dp) {
        int cherryYMax = dp.length;
        int cherryXMax = dp[0].length;
        if (y >= cherryYMax) return 0;
        if (y < 0 || x1 < 0 || x2 < 0 || x1 >= cherryXMax || x2 >= cherryXMax) return Integer.MIN_VALUE;
        return dp[y][x1][x2];
    }

    public int cherryPickup(int[][] grid) {
        int cherryYMax = grid.length;
        int cherryXMax = grid[0].length;
        int[][][] dp = new int[cherryYMax][cherryXMax][cherryXMax];
        for (int y = cherryYMax - 1; y >= 0; y--) {
            for (int x1 = 0; x1 < cherryXMax; x1++) {
                for (int x2 = x1; x2 < cherryXMax; x2++) {
                    dp[y][x1][x2] = grid[y][x1] + (x1 == x2 ? 0 : grid[y][x2]);
                    int nextMax = 0;
                    for (int i = x1 - 1; i <= x1 + 1; i++) {
                        for (int j = Math.max(x2 - 1, i); j <= x2 + 1; j++) {
                            nextMax = Math.max(nextMax, cherryDPGet(y + 1, i, j, dp));
                        }
                    }
                    dp[y][x1][x2] += nextMax;
                }
            }
        }
        return dp[0][0][cherryXMax - 1];
    }

    int maxTeamScore = 0;

    public int bestTeamScore(int[] scores, int[] ages) {
        int n = scores.length;
        int[][] players = new int[n][2];
        for (int i = 0; i < n; i++) {
            players[i][0] = scores[i];
            players[i][1] = ages[i];
        }
        Arrays.sort(players, Comparator.comparingInt((int[] a) -> a[1]).thenComparingInt(a -> a[0]));
        int[] dp = new int[n];
        int result = 0;
        for (int i = 0; i < n; i++) {
            dp[i] = players[i][0];
            for (int j = 0; j < i; j++) {
                if (players[j][0] <= players[i][0]) {
                    dp[i] = Math.max(dp[i], dp[j] + players[i][0]);
                }
            }
            result = Math.max(result, dp[i]);
        }
        return result;
    }

    public int maximizeSweetness(int[] sweetness, int k) {
        int left = Arrays.stream(sweetness).min().getAsInt();
        int right = Arrays.stream(sweetness).sum();
        int output = 0;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            int splitResult = canSplit(sweetness, mid);
            if (splitResult >= k + 1) {
                output = mid; // valid split, try for larger minimum sweetness
                left = mid + 1;
            } else {
                right = mid - 1; // not enough splits, reduce the sweetness floor
            }
        }

        return output; // the maximum minimum sweetness
    }

    public int canSplit(int[] sweetness, int sweetFloor) {
        int splitCounter = 0;
        int sweetCounter = 0;
        for (int i : sweetness) {
            sweetCounter += i;
            if (sweetCounter >= sweetFloor) {
                sweetCounter = 0;
                splitCounter++;
            }
        }
        return splitCounter;
    }






    /**
     * Main Method
     *
     *
     *
     *
     *
     */
    public static void main(String[] args) {
        Practice5 practice5 = new Practice5();
        var v = practice5.findStrobogrammatic(2);
    }

    public static List<List<String>> convertToListOfLists(String[][] array) {
        List<List<String>> listOfLists = new ArrayList<>();

        for (String[] subArray : array) {
            listOfLists.add(Arrays.asList(subArray));
        }

        return listOfLists;
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


    public class Node {
        int val;
        Node left;
        Node right;
        Node random;

        Node() {
        }

        Node(int val) {
            this.val = val;
        }

        Node(int val, Node left, Node right, Node random) {
            this.val = val;
            this.left = left;
            this.right = right;
            this.random = random;
        }
    }

    public class NodeCopy {
        int val;
        NodeCopy left;
        NodeCopy right;
        NodeCopy random;

        NodeCopy() {
        }

        NodeCopy(int val) {
            this.val = val;
        }

        NodeCopy(int val, NodeCopy left, NodeCopy right, NodeCopy random) {
            this.val = val;
            this.left = left;
            this.right = right;
            this.random = random;
        }
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

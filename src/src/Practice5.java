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
            rotateSum = rotateSum + sum - (n * (nums[n-k]));
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
        BitSet mask = new BitSet(n+k+1);
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (nums[i] == 1) mask.set(i);
        }
        for (int i = 0; i < n; i++) {
            if (!mask.get(i)) {
                mask.flip(i, i+k);
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
        for (int i = 0; i < n; i++) output += (i+1) * degree[i];
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


    int timeCounter= 0;
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



    public int numberOfWays(int startPos, int endPos, int k) {
        int MOD = 1_000_000_007;
        int distanceDifference = Math.abs(endPos - startPos);
        int maxDistanceDifference = k;
        if (distanceDifference > maxDistanceDifference) return 0;
        long[][] dp = new long[k+1][maxDistanceDifference + 1];
        dp[0][0] = 1;
        for (int i = 1; i <= k; i++) {
            for (int j = 0; j <= maxDistanceDifference; j++) {
                if (j == 0) {
                    dp[i][j] = (dp[i-1][j+1] * 2) % MOD;
                } else if (j == maxDistanceDifference) {
                    dp[i][j] = dp[i-1][j-1];
                } else {
                    dp[i][j] = (dp[i-1][j-1] + dp[i-1][j+1]) % MOD;
                }
            }
        }
        return (int) dp[k][distanceDifference];
    }


    /**
     * Main Method
     *
     *
     *
     *
     *
     *
     *
     *
     *
     */
    public static void main(String[] args) {

        String[][] flights = {{"JFK", "KUL"}, {"JFK", "NRT"}, {"NRT", "JFK"}};
        var flightList = convertToListOfLists(flights);
        long startTime, endTime;

        // Using the + operator
        startTime = System.nanoTime();
        for (int i = 0; i < 100000; i++) {
            String result = "Hello" + "World" + "!";
        }
        endTime = System.nanoTime();
        long durationPlus = endTime - startTime;

        // Using StringBuilder.append
        startTime = System.nanoTime();
        for (int i = 0; i < 100000; i++) {
            StringBuilder sb = new StringBuilder();
            sb.append("Hello").append("World").append("!");
            String result = sb.toString();
        }
        endTime = System.nanoTime();
        long durationStringBuilder = endTime - startTime;

        System.out.println("Time using + operator: " + durationPlus + " ns");
        System.out.println("Time using StringBuilder.append: " + durationStringBuilder + " ns");


        int e;
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

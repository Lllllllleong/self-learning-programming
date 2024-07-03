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


    /**
     * Main Method
     */
    public static void main(String[] args) {
        String[][] flights = {{"JFK", "KUL"}, {"JFK", "NRT"}, {"NRT", "JFK"}};
        var flightList = convertToListOfLists(flights);


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


    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        List<Integer> output = new ArrayList<>();
        List<Integer>[] graph = new List[n];
        Arrays.fill(graph, new ArrayList<>());
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
            output = new ArrayList<>(dq);
            int size = dq.size();
            for (int i = 0; i < size; i++) {
                int currentNode = dq.pollFirst();
                var nextNodes = graph[currentNode];
                for (Integer nextNode : nextNodes) {
                    if (--adjList[nextNode] == 1) dq.addLast(nextNode);
                }
            }
        }
        return output;
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


    public Node cloneGraph(Node node) {
        if (node == null) return null;
        int root = node.val;
        HashMap<Integer, Node> nodeGraph = new HashMap<>();
        cloneGraph(nodeGraph, node);
        return nodeGraph.get(root);
    }

    public void cloneGraph(HashMap<Integer, Node> nodeGraph, Node node) {
        if (node == null) return;
        int value = node.val;
        if (nodeGraph.containsKey(value)) return;
        nodeGraph.put(value, new Node(value));
        List<Node> neighbours = node.neighbors;
        for (Node neighbour : neighbours) {
            cloneGraph(nodeGraph, neighbour);
            nodeGraph.get(value).neighbors.add(nodeGraph.get(neighbour.val));
        }
    }

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
        int[][] directions = {{-1,0},{0,1},{1,0},{0,-1}};
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
                    dq.addLast(new int[]{y,x});
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
        for(int i = 0; i < n; i++){
            balloons[i+1] = nums[i];
        }
        balloons[0] = 1;
        balloons[n+1] =1;
        n = n+2;
        long[][] dp = new long[n][n];
        for (int i = n - 3; i >= 0; i--) {
            for (int j = i+2; j < n; j++) {
                long currentBase = balloons[i] * balloons[j];
                long currentMax = 0;
                for (int k = i+1; k < j; k++) {
                    long currentScore = currentBase * balloons[k] + dp[i][k] + dp[k][j];
                    currentMax = Math.max(currentMax, currentScore);
                }
                dp[i][j] = currentMax;
            }
        }
        return (int) dp[0][n-1];
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
                    dp[i][j] = dp[i + 1][j] || dp[i][j + 1] || dp[i+1][j+1];
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
        int output = ((n-1) - maxIndex) + (minIndex);
        if (minIndex < maxIndex) output--;
        return output;
    }







    class Node {
        public int val;
        public List<Node> neighbors;

        public Node() {
            val = 0;
            neighbors = new ArrayList<Node>();
        }

        public Node(int _val) {
            val = _val;
            neighbors = new ArrayList<Node>();
        }

        public Node(int _val, ArrayList<Node> _neighbors) {
            val = _val;
            neighbors = _neighbors;
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
        Practice4.TreeNode left;
        Practice4.TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, Practice4.TreeNode left, Practice4.TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }


    public class ListNode {
        int val;
        Practice4.ListNode next;

        ListNode(int x) {
            val = x;
            next = null;
        }
    }


}

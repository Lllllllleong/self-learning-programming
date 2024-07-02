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

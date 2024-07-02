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


//    public List<String> findItinerary(List<List<String>> tickets) {
//        HashMap<String, PriorityQueue<String>> graph = new HashMap<>();
//        for (var ticket : tickets) {
//            String from = ticket.get(0);
//            String to = ticket.get(1);
//            graph.computeIfAbsent(from, k -> new PriorityQueue<>()).add(to);
//        }
//        List<String> output = new ArrayList<>();
//        flightDFS(graph, output, "JFK");
//        return output;
//    }
//
//    public void flightDFS(HashMap<String, PriorityQueue<String>> graph,
//                          List<String> output,
//                          String currentPos) {
//        if (currentPos == null) return;
//        var pq = graph.getOrDefault(currentPos, new PriorityQueue<>());
//        while (!pq.isEmpty()) {
//            flightDFS(graph, output, pq.poll());
//        }
//        output.add(0, currentPos);
//    }


    public static List<String> findItinerary(List<List<String>> tickets) {
        return new AbstractList<String>() {

            private LinkedList<String> resList;

            private void onload() {
                Map<String, PriorityQueue<String>> flights = new HashMap<String, PriorityQueue<String>>();
                resList = new LinkedList<String>();
                for (List<String> ticket : tickets) {
                    final String source = ticket.get(0);
                    final String destination = ticket.get(1);
                    if (!flights.containsKey(source)) {
                        flights.put(source, new PriorityQueue<String>());
                    }
                    flights.get(source).add(destination);
                }
                dfs("JFK", flights);
            }

            private void dfs(String departure, Map<String, PriorityQueue<String>> flights) {
                PriorityQueue<String> arrivals = flights.get(departure);
                while (null != arrivals && !arrivals.isEmpty()) {
                    dfs(arrivals.poll(), flights);
                }
                resList.addFirst(departure);
            }

            private void init() {
                if (null == resList) {
                    onload();
                    System.gc();
                }
            }

            @Override
            public String get(int index) {
                init();
                return resList.get(index);
            }

            @Override
            public int size() {
                init();
                return resList.size();
            }

        };
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

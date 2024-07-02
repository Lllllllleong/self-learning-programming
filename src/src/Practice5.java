import java.math.*;
import java.sql.*;
import java.util.*;


public class Practice5 {


    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int n = numCourses;
        int[] childCount = new int[n];
        boolean[][] graph = new boolean[n][n];
        for (var prereq : prerequisites) {
            int a = prereq[0];
            int b = prereq[1];
            graph[b][a] = true;
            childCount[a]++;
        }
        Deque<Integer> dq = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            if (childCount[i] == 0) dq.addLast(i);
        }
        while (!dq.isEmpty()) {
            int currentCourse = dq.pollFirst();
            for (int i = 0; i < n; i++) {
                if (graph[currentCourse][i]) {
                    if (--childCount[i] == 0) dq.addLast(i);
                }
            }
        }
        for (int child : childCount) if (child != 0) return false;
        return true;
    }




    /**
     * Main Method
     *
     *
     *
     *
     *
     *
     */
    public static void main(String[] args) {







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

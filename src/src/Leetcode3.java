import com.sun.security.jgss.GSSUtil;

import java.util.*;

public class Leetcode3 {

    public List<Integer> diffWaysToCompute(String expression) {
        int n = expression.length();
        List<Integer> l = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            char c = expression.charAt(i);
            if (c == '*' || c == '+' || c == '-') {
                List<Integer> leftCombinations = diffWaysToCompute(expression.substring(0, i));
                List<Integer> rightCombinations = diffWaysToCompute(expression.substring(i + 1));
                for (Integer I : leftCombinations) {
                    for (Integer II : rightCombinations) {
                        switch (c) {
                            case '*' -> l.add(I * II);
                            case '-' -> l.add(I - II);
                            case '+' -> l.add(I + II);
                        }
                    }
                }
            }
        }
        //Base case: If l is empty, then expression is an integer
        if (l.size() == 0) {
            l.add(Integer.valueOf(expression));
        }
        return l;
    }


    public boolean checkValidString(String s) {
        int minClosing = 0;
        int maxClosing = 0;
        for (char c : s.toCharArray()) {
            switch (c) {
                case '(' -> {
                    minClosing++;
                    maxClosing++;
                }
                case ')' -> {
                    minClosing--;
                    maxClosing--;
                }
                case '*' -> {
                    minClosing--;
                    maxClosing++;
                }
            }
            minClosing = Math.max(minClosing, 0);
            if (maxClosing < 0) return false;
        }
        return (minClosing == 0);
    }


    public TreeNode deleteNode(TreeNode root, int key) {
        if (root == null) return root;
        if (root.val == key) {
            TreeNode mergedNode = mergeNode(root.left, root.right);
            return mergedNode;
        } else {
            if (root.val < key) {
                root.right = deleteNode(root.right, key);
            } else {
                root.left = deleteNode(root.left, key);
            }
            return root;
        }
    }

    public TreeNode mergeNode(TreeNode a, TreeNode b) {
        if (a == null) return b;
        if (b == null) return a;
        if (a.right == null) {
            a.right = b;
            return a;
        } else if (b.left == null) {
            b.left = a;
            return b;
        } else {
            b.left = mergeNode(a, b.left);
            return b;
        }
    }

    //DP approach
    public boolean canPartition(int[] nums) {
        int n = nums.length;
        if (n == 1) return false;
        int sum = 0;
        for (int i : nums) sum += i;
        if (sum % 2 != 0) return false;
        int halfSum = sum / 2;
        boolean[] dpBool = new boolean[halfSum + 1];
        dpBool[0] = true;
        for (int i : nums) {
            for (int j = halfSum; j >= i; j--) {
                if (dpBool[j - i]) {
                    dpBool[j] = true;
                }
            }
        }
        return dpBool[halfSum];
    }

    public int lastStoneWeightII(int[] stones) {
        int n = stones.length;
        if (n == 1) return stones[0];
        int sum = 0;
        for (int i : stones) sum += i;
        int halfSum = sum / 2;
        boolean[] dpBool = new boolean[halfSum + 1];
        dpBool[0] = true;
        for (int stone : stones) {
            for (int j = halfSum; j >= stone; j--) {
                if (dpBool[j - stone]) {
                    dpBool[j] = true;
                }
            }
        }
        int countDown = halfSum;
        while (countDown > 0 && dpBool[countDown] != false) {
            countDown--;
        }
        return (sum - (2 * countDown));
    }


    public int[][] insert(int[][] intervals, int[] newInterval) {
        List<int[]> output = new ArrayList<>();
        List<int[]> input = new ArrayList<>();
        int n = intervals.length;
        if (n == 0) {
            int[][] out = new int[1][2];
            out[0] = newInterval;
            return out;
        }
        for (var i : intervals) input.add(i);
        input.add(newInterval);
        Collections.sort(input, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                int aStart = a[0];
                int bStart = b[0];
                return (aStart - bStart);
            }
        });
        int[] first = input.get(0);
        for (int i = 1; i < input.size(); i++) {
            System.out.println("i, currentstart, currentend");
            int[] current = input.get(i);
            int currentStart = current[0];
            int currentEnd = current[1];
            System.out.println(i);
            System.out.println(currentStart);
            System.out.println(currentEnd);
            if (first[0] <= currentStart && currentStart <= first[1]) {
                first[1] = Math.max(first[1], currentEnd);
            } else {
                output.add(first);
                first = current;
            }
            if (i == input.size() - 1) output.add(first);
        }
        int[][] out = new int[output.size()][2];
        int counter = 0;
        for (int[] i : output) {
            out[counter] = i;
            counter++;
        }
        return out;
    }


    public int scoreOfStudents(String s, int[] answers) {
        int out = 0;
        String reverseS = "";
        for (char c : s.toCharArray()) reverseS = c + reverseS;
        int[] pointArray = new int[1001];
        int addFirstLeftToRight = addFirstEval(s);
        int addFirstRightToLeft = addFirstEval(reverseS);
        pointArray[addFirstLeftToRight] = 2;
        pointArray[addFirstRightToLeft] = 2;
        int leftToRight = leftToRight(s);
        int rightToLeft = leftToRight(reverseS);
        pointArray[leftToRight] = 2;
        pointArray[rightToLeft] = 2;

        int fivePoints = multiFirstEval(s);
        pointArray[fivePoints] = 5;
        for (int i : answers) out += pointArray[i];
        return out;
    }

    public int multiFirstEval(String s) {
        Deque<Integer> dq = new ArrayDeque<>();
        dq.add(Integer.valueOf(String.valueOf(s.charAt(0))));
        int n = s.length();
        for (int i = 1; i < n; i = i + 2) {
            char operator = s.charAt(i);
            Integer number = Integer.valueOf(String.valueOf(s.charAt(i + 1)));
            if (operator == '*') {
                Integer prior = dq.pollLast();
                number = number * prior;
                dq.addLast(number);
            } else {
                dq.addLast(number);
            }
        }
        int out = 0;
        while (!dq.isEmpty()) {
            out += dq.poll();
        }
        return out;
    }

    public int addFirstEval(String s) {
        Deque<Integer> dq = new ArrayDeque<>();
        dq.add(Integer.valueOf(String.valueOf(s.charAt(0))));
        int n = s.length();
        for (int i = 1; i < n; i = i + 2) {
            char operator = s.charAt(i);
            Integer number = Integer.valueOf(String.valueOf(s.charAt(i + 1)));
            if (operator == '+') {
                Integer prior = dq.pollLast();
                number = number + prior;
                dq.addLast(number);
            } else {
                dq.addLast(number);
            }
        }
        int out = dq.poll();
        while (!dq.isEmpty()) {
            out = out * dq.poll();
        }
        return out;
    }


    public int leftToRight(String s) {
        Deque<Integer> dq = new ArrayDeque<>();
        dq.add(Integer.valueOf(String.valueOf(s.charAt(0))));
        int n = s.length();
        for (int i = 1; i < n; i = i + 2) {
            char operator = s.charAt(i);
            Integer number = Integer.valueOf(String.valueOf(s.charAt(i + 1)));
            Integer prior = dq.pollLast();
            if (operator == '+') {
                number = number + prior;
                dq.addLast(number);
            } else {
                number = number * prior;
                dq.addLast(number);
            }
        }
        return dq.poll();
    }


    Integer[][][] dpArray3D;

    public int findMaxForm(String[] strs, int m, int n) {
        //m zeros
        //n ones
        int sLength = strs.length;
        dpArray3D = new Integer[sLength][m][n];
        return maxForm(strs, m, n, 0);
    }

    public int maxForm(String[] sArray, int xZeros, int xOnes, int index) {
        if (index == sArray.length) return 0;
        if (xZeros == 0 && xOnes == 0) return 0;
        if (dpArray3D[index][xZeros][xOnes] != null) return dpArray3D[index][xZeros][xOnes];
        String s = sArray[index];
        int zeroCount = 0, oneCount = 0;
        for (char c : s.toCharArray()) {
            switch (c) {
                case '0' -> zeroCount++;
                default -> oneCount++;
            }
        }
        dpArray3D[index][xZeros][xOnes] = maxForm(sArray, xZeros, xOnes, index+1);
        if ((xZeros-zeroCount) >= 0 && (xOnes-oneCount) >= 0) {
            dpArray3D[index][xZeros][xOnes]
                    = Math.max(dpArray3D[index][xZeros][xOnes], maxForm(sArray, xZeros-zeroCount, xOnes-oneCount, index+1));
        }
        return dpArray3D[index][xZeros][xOnes];
    }


    public static void main(String[] args) {
        int i = Integer.MAX_VALUE;
        System.out.println(i);
        System.out.println(Integer.MAX_VALUE);

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

//class Solution extends SolBase {
//    public int rand10() {
//        int i = 0;
//        for (int j = 0; j < 10; j++) {
//            i += rand7();
//        }
//        i = i % 10;
//        if (i == 0) i = 10;
//        return i;
//    }
//}





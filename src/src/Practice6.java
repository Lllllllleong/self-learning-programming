import java.util.*;

public class Practice6 {

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

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    public static void main(String[] args) {
        Practice5 practice5 = new Practice5();

    }
    class Solution {
        public int minOperations(int[] nums) {
            int n = nums.length;
            int zero = 0;
            int one = 0;
            for (int i = n - 1; i >= 0; i--) {
                int num = nums[i];
                if (num == 1) {
                    zero = Math.min(zero + 2, one + 1);
                } else {
                    one = Math.min(one + 2, zero + 1);
                }
            }
            one = Math.min(one, zero + 1);
            return one;
        }

    public int minOperations(int n) {
        int output = 0;
        char[] binaryString = Integer.toBinaryString(n).toCharArray();
        int bsLength = binaryString.length;
        char[] bsChar = new char[bsLength+1];
        bsChar[0] = '0';
        for (int i = 0; i < bsLength; i++) {
            bsChar[i+1] = binaryString[i];
        }
        for (int i = bsChar.length - 1; i >= 0; i--) {
            if (bsChar[i] == '1') {
                output++;
                if (i - 1 >= 0 && bsChar[i-1] == '1') {
                    while (i - 1 >= 0 && bsChar[i-1] == '1') i--;
                    bsChar[i-1] = '1';
                }
            }
        }
        return output;
    }

    public int maxPalindromes(String s, int k) {
        int n = s.length();
        char[] sChar = s.toCharArray();
        boolean[] palindromeDP = new boolean[n];
        int[] dp = new int[n+1];
        for (int i = n - 1; i >= 0; i--) {
            char c = sChar[i];
            for (int j = n - 1; j >= i; j--) {
                char d = sChar[j];
                boolean flag = (j - i <= 2) ? true : palindromeDP[j-1];
                if (c == d && flag) {
                    palindromeDP[j] = true;
                    if (j - i + 1 >= k) dp[i] = Math.max(dp[i], dp [j+1] + 1);
                } else {
                    palindromeDP[j] = false;
                }
            }
            dp[i] = Math.max(dp[i], dp[i+1]);
        }
        return dp[0];
    }




    public int countGoodStrings(int low, int high, int zero, int one) {
        int mod = 1000000007;
        int[] dp = new int[high + 2];
        dp[zero]++;
        dp[one]++;
        int output = 0;
        for (int i = 0; i <= high; i++) {
            int l = dp[i];
            if (l == 0) continue;
            if (i + zero <= high) {
                dp[i + zero] += l;
                dp[i+zero] %= mod;
            }
            if (i + one <= high) {
                dp[i + one] += l;
                dp[i+one] %= mod;
            }
            if (i >= low) output = (output + dp[i]) % mod;
        }
        return output;
    }

    public List<Integer> goodIndices(int[] nums, int k) {
        int n = nums.length;
        List<Integer> output = new ArrayList<>();
        boolean[] dp = new boolean[n+2];
        int prior = Integer.MAX_VALUE;
        int consecCounter = 0;
        for (int i = n - 1; i >= 0; i--) {
            int num = nums[i];
            if (num > prior) {
                consecCounter = 1;
            } else {
                consecCounter++;
            }
            if (consecCounter >= k) dp[i] = true;
            prior = num;
        }
        consecCounter = 0;
        prior = Integer.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            if (num > prior) {
                consecCounter = 1;
            } else {
                consecCounter++;
            }
            if (consecCounter >= k && dp[i+2]) output.add(i+1);
            prior = num;
        }
        return output;
    }



}

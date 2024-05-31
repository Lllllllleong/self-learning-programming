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




    public static void main(String[] args) {

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

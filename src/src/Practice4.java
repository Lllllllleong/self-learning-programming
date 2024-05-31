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

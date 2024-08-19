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

    public int garbageCollection(String[] garbage, int[] travel) {
        int n = garbage.length;
        //G, P, M
        int[] cache = new int[3];
        int output = 0;
        for (int i = 0; i < n; i++) {
            char[] chars = garbage[i].toCharArray();
            for (char c : chars) {
                int j = (c == 'G') ? 0 : (c == 'P') ? 1 : 2;
                while (cache[j] < i) output += travel[cache[j]++];
                output += 1;
            }
        }
        return output;
    }


    public boolean makePalindrome(String s) {
        int n = s.length();
        char[] chars = s.toCharArray();
        int left = 0;
        int right = n-1;
        int counter = 0;
        while (left <= right) {
            if (chars[left++] != chars[right--]) counter++;
        }
        return (counter <= 2);
    }

    public int palindromePartition(String s, int k) {
        char[] chars = s.toCharArray();
        int n = chars.length;
        int[][] dp = new int[n][n];
        int[] cache = new int[n+1];
        for (int i = n - 1; i >= 0; i--) {
            char a = chars[i];
            for (int j = i+1; j < n; j++) {
                char b = chars[j];
                if (a != b) {
                    if (j == (i + 1)) dp[i][j] = 1;
                    else dp[i][j] = dp[i+1][j-1] + 1;
                } else {
                    dp[i][j] = dp[i + 1][j - 1];
                }
            }
            cache[i] = dp[i][n-1];
        }
        for (int i = 1; i < k; i++) {
            int upperBound = n - 1 - i;
            int[] currentCache = new int[n];
            for (int j = upperBound; j >= 0; j--) {
                currentCache[j] = Integer.MAX_VALUE;
                for (int l = j; l <= upperBound; l++) {
                    currentCache[j] = Math.min(currentCache[j], dp[j][l] + cache[l+1]);
                }
            }
            cache = currentCache;
        }
        return cache[0];
    }
    public int maximumSum(int[] arr) {
        int n = arr.length;
        if (n == 1) return arr[0];
        int[] cache = new int[2];
        int output = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            int num = arr[i];
            cache[0] += num;
            cache[1] += num;
            if (cache[0] != num) {
                cache[1] = Math.max(cache[1], cache[0] - num);
            }
            output = Math.max(output, Math.max(cache[0], cache[1]));
            cache[0] = Math.max(cache[0], 0);
            cache[1] = Math.max(cache[1], 0);
        }
        return output;
    }


    public int removeAlmostEqualCharacters(String word) {
        int n = word.length();
        char[] chars = word.toCharArray();
        int output = 0;
        for (int i = 1; i < n; i++) {
            int aChar = chars[i - 1] - 'a';
            int bChar = chars[i] - 'a';
            int diff = Math.abs(aChar - bChar);
            if (diff <= 1 || diff == 26) {
                i++;
                output++;
            }
        }
        return output;
    }

    public List<Integer> shortestDistanceColor(int[] colors, int[][] queries) {
        List<Integer> output = new ArrayList<>();
        int n = colors.length;
        int[][] dp = new int[n][3];
        int[] countDP = new int[3];
        Arrays.fill(countDP, -1);
        for (int i = 0; i < n; i++) {
            int color = colors[i] - 1;
            countDP[color] = 0;
            for (int j = 0; j < 3; j++) {
                dp[i][j] = countDP[j];
                if (countDP[j] >= 0) {
                    countDP[j]++;
                }
            }
        }
        Arrays.fill(countDP, -1);
        for (int i = n - 1; i >= 0; i--) {
            int color = colors[i] - 1;
            countDP[color] = 0;
            for (int j = 0; j < 3; j++) {
                if (countDP[j] >= 0) {
                    if (dp[i][j] == -1) dp[i][j] = countDP[j];
                    else dp[i][j] = Math.min(dp[i][j], countDP[j]);
                    countDP[j]++;
                }
            }
        }
        for (int[] query : queries) {
            int index = query[0];
            int c = query[1] - 1;
            output.add(dp[index][c]);
        }
        return output;
    }


    public int minCost(int maxTime, int[][] edges, int[] passingFees) {
        int n = passingFees.length;
        List<int[]>[] graph = new ArrayList[n];
        for (int i = 0; i < n; i++) {
            graph[i] = new ArrayList<>();
        }
        for (int[] edge : edges) {
            int a = edge[0];
            int b = edge[1];
            int time = edge[2];
            graph[a].add(new int[]{b, time});
            graph[b].add(new int[]{a, time});
        }
        int[][] minCost = new int[n][maxTime + 1];
        for (int i = 0; i < n; i++) {
            Arrays.fill(minCost[i], Integer.MAX_VALUE);
        }
        minCost[0][0] = passingFees[0];
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[2]));
        pq.add(new int[]{0, 0, passingFees[0]});
        while (!pq.isEmpty()) {
            int[] current = pq.poll();
            int node = current[0];
            int timeSpent = current[1];
            int cost = current[2];
            if (node == n - 1) {
                return cost;
            }
            for (int[] neighbor : graph[node]) {
                int nextNode = neighbor[0];
                int travelTime = neighbor[1];
                int nextTimeSpent = timeSpent + travelTime;
                int nextCost = cost + passingFees[nextNode];

                if (nextTimeSpent <= maxTime && nextCost < minCost[nextNode][nextTimeSpent]) {
                    minCost[nextNode][nextTimeSpent] = nextCost;
                    pq.offer(new int[]{nextNode, nextTimeSpent, nextCost});
                }
            }
        }
        return -1;
    }


    public int maxTotalReward(int[] rewardValues) {
        int n = rewardValues.length;
        if (n == 1) return rewardValues[0];
        Arrays.sort(rewardValues);
        int max = rewardValues[n - 1];
        boolean[] dp = new boolean[max + 1];
        dp[0] = true;
        int prior = -1;
        int output = 0;
        for (int i : rewardValues) {
            if (i == prior) continue;
            prior = i;
            for (int j = i - 1; j >= 0; j--) {
                if (dp[j]) {
                    int next = j + i;
                    if (next < max) dp[next] = true;
                    else output = Math.max(output, next);
                }

            }
        }
        return output;
    }

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
        char[] bsChar = new char[bsLength + 1];
        bsChar[0] = '0';
        for (int i = 0; i < bsLength; i++) {
            bsChar[i + 1] = binaryString[i];
        }
        for (int i = bsChar.length - 1; i >= 0; i--) {
            if (bsChar[i] == '1') {
                output++;
                if (i - 1 >= 0 && bsChar[i - 1] == '1') {
                    while (i - 1 >= 0 && bsChar[i - 1] == '1') i--;
                    bsChar[i - 1] = '1';
                }
            }
        }
        return output;
    }

    public int maxPalindromes(String s, int k) {
        int n = s.length();
        char[] sChar = s.toCharArray();
        boolean[] palindromeDP = new boolean[n];
        int[] dp = new int[n + 1];
        for (int i = n - 1; i >= 0; i--) {
            char c = sChar[i];
            for (int j = n - 1; j >= i; j--) {
                char d = sChar[j];
                boolean flag = (j - i <= 2) ? true : palindromeDP[j - 1];
                if (c == d && flag) {
                    palindromeDP[j] = true;
                    if (j - i + 1 >= k) dp[i] = Math.max(dp[i], dp[j + 1] + 1);
                } else {
                    palindromeDP[j] = false;
                }
            }
            dp[i] = Math.max(dp[i], dp[i + 1]);
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
                dp[i + zero] %= mod;
            }
            if (i + one <= high) {
                dp[i + one] += l;
                dp[i + one] %= mod;
            }
            if (i >= low) output = (output + dp[i]) % mod;
        }
        return output;
    }

    public List<Integer> goodIndices(int[] nums, int k) {
        int n = nums.length;
        List<Integer> output = new ArrayList<>();
        boolean[] dp = new boolean[n + 2];
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
            if (consecCounter >= k && dp[i + 2]) output.add(i + 1);
            prior = num;
        }
        return output;
    }
}




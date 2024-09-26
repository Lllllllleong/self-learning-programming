import java.sql.*;
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
        public int val;
        public Node prev;
        public Node next;
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
        int key;
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int key, int val) {
            this.key = key;
            this.val = val;
        }

        ListNode(int key, int val, ListNode next) {
            this.key = key;
            this.val = val;
            this.next = next;
        }
    }

    public static void main(String[] args) {
        Practice6 practice6 = new Practice6();
        int[] nums = new int[]{1, 3, 5, 3, 3, 7, 1, 7, 3};

        String a = "catg";
        String b = "atgcatc";


        StringBuilder sb = new StringBuilder();
        for (int i : nums) sb.append(i);
        System.out.println(sb.toString());
    }



    class MyCalendar {

        ListNode ln;
        public MyCalendar() {
            ln = new ListNode(-2, -1, null);
        }
        public boolean book(int start, int end) {
            ListNode prev = ln;
            ListNode next = ln.next;
            while (next != null && next.key < end) {
                prev = next;
                next = next.next;
            }
            if (prev.val > start) return false;
            ListNode current = new ListNode(start, end, next);
            prev.next = current;
            return true;
        }
    }


    public int countMatchingSubarrays(int[] nums, int[] pattern) {
        int n = nums.length;
        int p = pattern.length;
        int output = 0;
        StringBuilder sb = new StringBuilder();
        for (int i : pattern) sb.append(i + 1);
        String patternString = sb.toString();
        sb = new StringBuilder();
        for (int i = 1; i < n; i++) {
            int previous = nums[i-1];
            int current = nums[i];
            int currentPatternChar = Integer.compare(current, previous) + 1;
            if (sb.length() >= p) {
                sb.delete(0,1);
            }
            sb.append(currentPatternChar);
            if (sb.length() == p) {
                String currentString = sb.toString();
                System.out.println(currentString);
                if (patternString.equals(currentString)) output++;
            }
        }
        return output;
    }

    public long maximumSubarraySum(int[] nums, int k) {
        int n = nums.length;
        long output = Long.MIN_VALUE;
        long prefixSum = 0;
        HashMap<Integer, Long> hm = new HashMap<>();
        for (int i = 0; i < n; i++) {
            int currentNumber = nums[i];
            hm.put(currentNumber, Math.min(hm.getOrDefault(currentNumber, Long.MAX_VALUE), prefixSum));
            prefixSum += currentNumber;
            if (hm.containsKey(currentNumber - k)) {
                output = Math.max(output, prefixSum - hm.get(currentNumber - k));
            }
            if (hm.containsKey(currentNumber + k)) {
                output = Math.max(output, prefixSum - hm.get(currentNumber + k));
            }
        }
        return (output == Long.MIN_VALUE) ? 0 : output;
    }



    int maxSubtree = 0;
    public int maximumSubtreeSize(int[][] edges, int[] colors) {
        int n = colors.length;
        maxSubtree = 0;
        List<Integer>[] graph = new ArrayList[n];
        for (int i = 0; i < n; i++) graph[i] = new ArrayList<>();
        for (int[] edge : edges) {
            int a = edge[0];
            int b = edge[1];
            graph[a].add(b);
            graph[b].add(a);
        }
        maxSubtreeDFS(graph, -1, 0, colors, new int[n]);
        return maxSubtree;

    }

    public void maxSubtreeDFS(List<Integer>[] graph,
                              int parent,
                              int currentNode,
                              int[] colours,
                              int[] subtreeSize) {
        List<Integer> adjList = graph[currentNode];
        for (int i : adjList) {
            if (i != parent) maxSubtreeDFS(graph, currentNode, i, colours, subtreeSize);
        }
        boolean flag = true;
        int currentColour = colours[currentNode];
        int currentSubtreeSize = 0;
        for (int i : adjList) {
            if (i == parent) continue;
            int childColour = colours[i];
            if (currentColour != childColour) {
                flag = false;
                break;
            } else {
                currentSubtreeSize += subtreeSize[i];
            }
        }
        if (!flag) {
            colours[currentNode] = -1;
        } else {
            maxSubtree = Math.max(maxSubtree, currentSubtreeSize + 1);
        }
    }




    public int[] pourWater(int[] heights, int volume, int k) {
        int n = heights.length;
        while (volume > 0) {
            volume--;
            boolean poured = false;
            int position = k;
            for (int i = k - 1; i >= 0; i--) {
                if (heights[i] < heights[position]) {
                    position = i;
                } else if (heights[i] > heights[position]) {
                    break;
                }
            }
            if (position != k) {
                heights[position]++;
                poured = true;
            } else {
                for (int i = k + 1; i < n; i++) {
                    if (heights[i] < heights[position]) {
                        position = i;
                    } else if (heights[i] > heights[position]) {
                        break;
                    }
                }
                if (position != k) {
                    heights[position]++;
                    poured = true;
                }
            }
            if (!poured) {
                heights[k]++;
            }
        }
        return heights;
    }

    public long validSubstringCount(String word1, String word2) {
        int n = word1.length();
        long output = 0;
        int[] charFrequency = new int[26];
        for (char c : word2.toCharArray()) {
            charFrequency[c - 'a']++;
        }
        int left = 0, right = 0;
        int[] windowCount = new int[26];
        char[] chars1 = word1.toCharArray();
        int required = word2.length();
        while (right < n) {
            char rightChar = chars1[right];
            if (charFrequency[rightChar - 'a'] > 0) {
                windowCount[rightChar - 'a']++;
                if (windowCount[rightChar - 'a'] <= charFrequency[rightChar - 'a']) {
                    required--;
                }
            }
            while (required == 0) {
                output += n - right;
                char leftChar = chars1[left];
                if (charFrequency[leftChar - 'a'] > 0) {
                    windowCount[leftChar - 'a']--;
                    if (windowCount[leftChar - 'a'] < charFrequency[leftChar - 'a']) {
                        required++;
                    }
                }
                left++;
            }
            right++;
        }
        return output;
    }


    public int[] toArray(Node node) {
        List<Integer> list = new ArrayList<>();
        while (node.prev != null) node = node.prev;
        list.add(node.val);
        while (node.next != null) {
            node = node.next;
            list.add(node.val);
        }
        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    public int winningPlayerCount(int n, int[][] pick) {
        int[][] cache = new int[n][11];
        for (int[] currentPick : pick) {
            int player = currentPick[0];
            int colour = currentPick[1];
            cache[player][colour]++;
        }
        int output = 0;
        for (int i = 0; i < n; i++) {
            int[] currentPlayerPicks = cache[i];
            for (int currentPick : currentPlayerPicks) {
                if (currentPick > i) {
                    output++;
                    break;
                }
            }
        }
        return output;
    }


    public long maxScore(int[] a, int[] b) {
        int n = a.length;
        int m = b.length;
        long[] dp = new long[m + 1];
        for (int i = n - 1; i >= 0; i--) {
            long currentMultiplier = a[i];
            int mStart = (m-1) - ((n-1) - i);
            long[] dpNext = new long[m + 1];
            dpNext[mStart + 1] = Long.MIN_VALUE;
            for (int j = mStart; j >= 0; j--) {
                dpNext[j] = Math.max(dpNext[j + 1], currentMultiplier * b[j] + dp[j + 1]);
            }
            dp = dpNext;
        }
        return dp[0];
    }


    public long findMaximumScore(List<Integer> nums) {
        long output = 0;
        long max = nums.get(0);
        long priorIndex = 0;
        int n = nums.size();
        for (int i = 1; i < n; i++) {
            long current = nums.get(i);
            if (current > max || i == n - 1) {
                output += (i - priorIndex) * max;
                max = current;
                priorIndex = i;
            }
        }

        return output;
    }

    public int[] findDiagonalOrder(int[][] mat) {
        if (mat == null || mat.length == 0) return new int[0];
        int yMax = mat.length;
        int xMax = mat[0].length;
        int[] output = new int[yMax * xMax];
        int index = 0;
        for (int d = 0; d < yMax + xMax - 1; d++) {
            if (d % 2 == 0) {
                int y = Math.min(d, yMax - 1);
                int x = d - y;
                while (y >= 0 && x < xMax) {
                    output[index++] = mat[y--][x++];
                }
            } else {
                int x = Math.min(d, xMax - 1);
                int y = d - x;
                while (x >= 0 && y < yMax) {
                    output[index++] = mat[y++][x--];
                }
            }
        }
        return output;
    }

    public int countPairs(int[] nums) {
        int[] frequency = new int[1000001];
        int output = 0;
        for (int i : nums) {
            output += frequency[i];
            String numberStr = String.format("%07d", i);
            char[] digits = numberStr.toCharArray();
            for (int j = 0; j < digits.length; j++) {
                for (int k = j + 1; k < digits.length; k++) {
                    swap(digits, j, k);
                    String swappedString = new String(digits);

                    int swappedNumber = Integer.parseInt(swappedString);

                    if (swappedNumber != i && swappedNumber < frequency.length) {
                        output += frequency[swappedNumber];
                    }
                    swap(digits, j, k);
                }
            }
            frequency[i]++;
        }
        return output;
    }

    private void swap(char[] arr, int i, int j) {
        char temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }



    public int maximumLength(int[] nums, int k) {
        int n = nums.length;
        int[][] dp = new int[n][k];
        int output = 0;
        for (int i = 0; i < n; i++) {
            int a = nums[i];
            for (int j = i+1; j < n; j++) {
                int b = nums[j];
                int mod = (a + b) % k;
                dp[j][mod] = Math.max(dp[j][mod], dp[i][mod] + 1);
                output = Math.max(output, dp[j][mod]);
            }
        }
        return output + 1;
    }




    public int peopleAwareOfSecret(int n, int delay, int forget) {
        int MOD = 1_000_000_007;
        long[] dp = new long[n + 1];
        dp[1] = 1;
        long currentSharers = 0;
        for (int i = 1; i <= n; i++) {
            currentSharers = (currentSharers + dp[i]) % MOD;
            if (i + delay <= n) {
                dp[i + delay] = (dp[i + delay] + currentSharers) % MOD;
            }
            if (i + forget <= n) {
                dp[i + forget] = (dp[i + forget] - currentSharers + MOD) % MOD;
            }
        }
        long result = 0;
        for (int i = n - forget + 1; i <= n; i++) {
            result = (result + dp[i]) % MOD;
        }
        return (int) result;
    }


    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<List<Integer>, List<String>> hm = new HashMap<>();
        for (String s : strs) {
            Integer[] charFrequency = new Integer[26];
            Arrays.fill(charFrequency, 0);
            for (char c : s.toCharArray())
                charFrequency[c - 'a']++;
            List<Integer> key = Arrays.asList(charFrequency);
            hm.computeIfAbsent(key, k -> new ArrayList<>()).add(s);
        }
        return new ArrayList<>(hm.values());
    }

    public boolean isAnagram(String s, String t) {
        int[] sCharFrequency = new int[26];
        int[] tCharFrequency = new int[26];
        for (char c : s.toCharArray()) {
            sCharFrequency[c - 'a']++;
        }
        for (char c : t.toCharArray()) {
            tCharFrequency[c - 'a']++;
        }
        for (int i = 0; i < 26; i++) {
            if (sCharFrequency[i] != tCharFrequency[i]) return false;
        }
        return true;
    }


    public List<Integer> majorityElement(int[] nums) {
        List<Integer> output = new ArrayList<>();
        int a = Integer.MAX_VALUE;
        int b = Integer.MAX_VALUE;
        int aFrequency = 0;
        int bFrequency = 0;
        for (int i : nums) {
            if (i == a) {
                aFrequency++;
            } else if (i == b) {
                bFrequency++;
            } else if (aFrequency == 0) {
                aFrequency++;
                a = i;
            } else if (bFrequency == 0) {
                bFrequency++;
                b = i;
            } else {
                aFrequency--;
                bFrequency--;
            }
        }
        aFrequency = 0;
        bFrequency = 0;
        for (int i : nums) {
            if (i == a) aFrequency++;
            if (i == b) bFrequency++;
        }
        if (aFrequency > nums.length / 3) output.add(a);
        if (bFrequency > nums.length / 3) output.add(b);
        return output;
    }

    public int minimumDeletions(int[] nums) {
        int n = nums.length;
        if (n <= 3) {
            if (n == 1) return 1;
            else return 2;
        }
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        int minIndex = -1;
        int maxIndex = -1;
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            if (num < min) {
                minIndex = i;
                min = num;
            }
            if (num > max) {
                maxIndex = i;
                max = num;
            }
        }
        int lowerIndex = Math.min(minIndex, maxIndex);
        int upperIndex = Math.max(minIndex, maxIndex);
        int output = Integer.MAX_VALUE;
        output = Math.min(output, upperIndex + 1);
        output = Math.min(output, n - lowerIndex);
        output = Math.min(output, (lowerIndex + 1 + (n - upperIndex)));
        return output;
    }


    public String largestTimeFromDigits(int[] arr) {
        int maxHour = -1;
        int maxMin = -1;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (j == i) continue;
                for (int k = 0; k < 4; k++) {
                    if (k == i || k == j) continue;
                    int l = 6 - i - j - k;
                    int hour = arr[i] * 10 + arr[j];
                    int min = arr[k] * 10 + arr[l];
                    if (hour < 24 && min < 60) {
                        if (hour > maxHour || (hour == maxHour && min > maxMin)) {
                            maxHour = hour;
                            maxMin = min;
                        }
                    }
                }
            }
        }
        if (maxHour == -1) return "";
        return String.format("%02d:%02d", maxHour, maxMin);
    }

    public int validSubarrays(int[] nums) {
        int n = nums.length;
        int output = 0;
        Deque<Integer> dq = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            while (!dq.isEmpty() && nums[dq.peekLast()] > num) {
                output += (i - dq.pollLast());
            }
            dq.addLast(i);
        }
        while (!dq.isEmpty()) {
            output += (n - dq.pollFirst());
        }
        return output;
    }


    public int bagOfTokensScore(int[] tokens, int power) {
        int n = tokens.length;
        Arrays.sort(tokens);
        int maxScore = 0;
        int currentScore = 0;
        int left = 0;
        int right = n - 1;
        while (left <= right) {
            if (power >= tokens[left]) {
                power -= tokens[left];
                currentScore++;
                maxScore = Math.max(maxScore, currentScore);
                left++;
            } else {
                if (currentScore == 0) break;
                power += tokens[right];
                currentScore--;
                right--;
            }
        }
        return maxScore;
    }


    public int minSteps(int n) {
        int output = 0;
        for (int i = 2; n > 1; i++) {
            while (n % i == 0) {
                output += i;
                n /= i;
            }
        }
        return output;
    }

    public boolean containsNearbyAlmostDuplicate(int[] nums, int indexDiff, int valueDiff) {
        int n = nums.length;
        HashMap<Integer, Integer> bucketMap = new HashMap<>();
        int bucketSize = valueDiff + 1;
        int offset = Integer.MAX_VALUE;
        for (int i : nums) offset = Math.min(offset, i);
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            int bucketKey = (num - offset) / bucketSize;
            if (bucketMap.containsKey(bucketKey)) return true;
            if (bucketMap.containsKey(bucketKey - 1)
                    && Math.abs(nums[i] - bucketMap.get(bucketKey - 1)) <= valueDiff) return true;
            if (bucketMap.containsKey(bucketKey + 1)
                    && Math.abs(nums[i] - bucketMap.get(bucketKey + 1)) <= valueDiff) return true;
            bucketMap.put(bucketKey, nums[i]);
            if (i >= indexDiff) {
                bucketMap.remove(((nums[i - indexDiff]) - offset) / bucketSize);
            }
        }
        return false;
    }

    TreeNode tnOutput = null;
    int pVal = 0;
    int qVal = 0;

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        pVal = p.val;
        qVal = q.val;
        lca(root);
        return tnOutput;
    }

    public boolean lca(TreeNode root) {
        if (root == null) return false;
        boolean left = lca(root.left);
        boolean right = lca(root.right);
        boolean mid = (root.val == pVal || root.val == qVal);
        if ((left && right) || (mid && left) || (mid && right)) {
            tnOutput = root;
        }
        return (left || right || mid);
    }

    public int hIndex(int[] citations) {
        int n = citations.length;
        int[] prefixSum = new int[n + 1];
        for (int c : citations) {
            if (c >= n) {
                prefixSum[n]++;
            } else {
                prefixSum[c]++;
            }
        }
        int remainingPapers = 0;
        for (int h = n; h >= 0; h--) {
            remainingPapers += prefixSum[h];
            if (remainingPapers >= h) {
                return h;
            }
        }
        return 0;
    }

    public int[] sortArray(int[] nums) {
        int n = nums.length;
        Random rand = new Random();
        randomQuickSort(nums, 0, n - 1, rand);
        return nums;
    }

    public void randomQuickSort(int[] nums, int a, int b, Random rand) {
        if (a < b) {
            int randomPivot = randomIndex(a, b, rand);
            //Swap the pivot number with the end;
            int pivotNumber = nums[randomPivot];
            nums[randomPivot] = nums[b];
            nums[b] = pivotNumber;
            int i = a - 1;
            int j = a - 1;
            while (++j < b) {
                if (nums[j] <= pivotNumber) {
                    i++;
                    int tmp = nums[j];
                    nums[j] = nums[i];
                    nums[i] = tmp;
                }
            }
            //Swap back the pivot
            nums[b] = nums[i + 1];
            nums[i + 1] = pivotNumber;
            randomQuickSort(nums, a, i, rand);
            randomQuickSort(nums, i + 2, b, rand);
        }
    }

    public int randomIndex(int a, int b, Random rand) {
        int range = b - a + 1;
        int output = a + rand.nextInt(range);
        return output;
    }

    public int minCut(String s) {
        char[] chars = s.toCharArray();
        int n = chars.length;
        boolean[][] isPalindrome = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            int left = i;
            int right = i;
            while (left >= 0 && right < n && chars[left] == chars[right]) {
                isPalindrome[left][right] = true;
                left--;
                right++;
            }
            left = i - 1;
            right = i;
            while (left >= 0 && right < n && chars[left] == chars[right]) {
                isPalindrome[left][right] = true;
                left--;
                right++;
            }
        }
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[n] = 0;
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (isPalindrome[i][j]) {
                    dp[i] = Math.min(dp[i], dp[j + 1] + 1);
                }
            }
        }
        return dp[0] - 1;
    }

    public int longestIdealString(String s, int k) {
        int[] dp = new int[26];
        char[] sChars = s.toCharArray();
        int output = 0;
        for (char c : sChars) {
            int charIndex = c - 'a';
            int i = charIndex;
            int nextDifference = 1;
            dp[charIndex]++;
            while (nextDifference <= k && i > 0) {
                i--;
                nextDifference++;
                dp[charIndex] = Math.max(dp[charIndex], dp[i] + 1);
            }
            i = charIndex;
            nextDifference = 1;
            while (nextDifference <= k && i < 25) {
                i++;
                nextDifference++;
                dp[charIndex] = Math.max(dp[charIndex], dp[i] + 1);
            }
            output = Math.max(output, dp[charIndex]);
        }
        return output;
    }


    public int numTeams(int[] rating) {
        TreeMap<Integer, Integer> lowerTM = new TreeMap<>();
        TreeMap<Integer, Integer> higherTM = new TreeMap<>();
        int output = 0;
        for (Integer I : rating) {
            int numLower = 0;
            Integer lowerKey = lowerTM.lowerKey(I);
            while (lowerKey != null) {
                output += lowerTM.get(lowerKey);
                numLower++;
                lowerKey = lowerTM.lowerKey(lowerKey);
            }
            lowerTM.put(I, numLower);
            int numHigher = 0;
            Integer higherKey = higherTM.higherKey(I);
            while (higherKey != null) {
                output += higherTM.get(higherKey);
                numHigher++;
                higherKey = higherTM.higherKey(higherKey);
            }
            higherTM.put(I, numHigher);
        }
        return output;
    }


    public int kConcatenationMaxSum(int[] arr, int k) {
        long M = 1000000007;
        long arrSum = 0;
        long firstMax = 0;
        long firstSum = 0;
        for (int i : arr) {
            arrSum += i;
            firstSum += i;
            firstMax = Math.max(firstMax, firstSum);
            firstSum = Math.max(firstSum, 0);
        }
        if (k == 1) return (int) (firstMax % M);
        long secondMax = firstMax;
        long secondSum = firstSum;
        for (int i : arr) {
            secondSum += i;
            secondMax = Math.max(secondMax, secondSum);
            secondSum = Math.max(secondSum, 0);
        }
        if (secondMax == firstMax) return (int) (firstMax % M);
        if (arrSum <= 0 || k == 2) return (int) (secondMax % M);
        long result = ((secondMax % M) + (((k - 2) * arrSum) % M)) % M;
        return (int) result;
    }

    public int minimumOperations(List<Integer> nums) {
        int n = nums.size();
        int[] dp = new int[4];
        for (int num : nums) {
            if (num == 1) {
                dp[1]++;
            } else if (num == 2) {
                dp[2] = Math.max(dp[2], dp[1]) + 1;
            } else if (num == 3) {
                dp[3] = Math.max(dp[3], Math.max(dp[1], dp[2])) + 1;
            }
        }
        int maxLength = Math.max(dp[1], Math.max(dp[2], dp[3]));
        return n - maxLength;
    }

//    class Solution {
//        int[] original;
//        int[] current;
//        int n = 0;
//        Random r;
//        public Solution(int[] nums) {
//            n = nums.length;
//            original = nums.clone();
//            current = nums;
//            r = new Random();
//        }
//
//        public int[] reset() {
//            current = original.clone();
//            return current;
//        }
//
//        public int getRandomFromRange(int start, int end) {
//            return r.nextInt((end - start)) + start;
//        }
//        public int[] shuffle() {
//            for (int i = 0; i < n; i++) {
//                int randomIndex = getRandomFromRange(i, n);
//                int temp = current[i];
//                current[i] = current[randomIndex];
//                current[randomIndex] = temp;
//            }
//            return current;
//        }
//    }

//    class Solution {
//
//        public List<Integer> list;
//        public Random rand;
//
//        public Solution(ListNode head) {
//            list = new ArrayList<>();
//            rand = new Random();
//            list.add(head.val);
//            ListNode current = head.next;
//            while (current != null) {
//                list.add(current.val);
//                current = current.next;
//            }
//        }
//
//        public int getRandom() {
//            return list.get(rand.nextInt(list.size()));
//        }
//    }


    public long maxEnergyBoost(int[] energyDrinkA, int[] energyDrinkB) {
        int n = energyDrinkA.length;
        long[][] dp = new long[n + 2][2];
        for (int i = n - 1; i >= 0; i--) {
            dp[i][0] = Math.max(energyDrinkA[i] + dp[i + 1][0], energyDrinkA[i] + dp[i + 2][1]);
            dp[i][1] = Math.max(energyDrinkB[i] + dp[i + 1][1], energyDrinkB[i] + dp[i + 2][0]);
        }
        return Math.max(dp[0][0], dp[0][1]);
    }

    public int rob(TreeNode root) {
        int[] result = robTree(root);
        return Math.max(result[0], result[1]);
    }

    private int[] robTree(TreeNode root) {
        if (root == null) {
            return new int[]{0, 0};
        }
        int[] left = robTree(root.left);
        int[] right = robTree(root.right);
        int maxWithoutRobbingCurrent = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        int maxWithRobbingCurrent = root.val + left[0] + right[0];
        return new int[]{maxWithoutRobbingCurrent, maxWithRobbingCurrent};
    }

    public static int getMaxLen(int[] nums) {
        int maxLen = 0;
        int firstNegative = -1, zeroPosition = -1;
        int negativeCount = 0;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < 0) {
                negativeCount++;
                if (firstNegative == -1) firstNegative = i;
            }

            if (nums[i] == 0) {
                zeroPosition = i;
                firstNegative = -1;
                negativeCount = 0;
            } else {
                if (negativeCount % 2 == 0) {
                    maxLen = Math.max(maxLen, i - zeroPosition);
                } else {
                    maxLen = Math.max(maxLen, i - firstNegative);
                }
            }
        }

        return maxLen;
    }

    public int returnToBoundaryCount(int[] nums) {
        int output = 0;
        int pos = 0;
        for (int i : nums) {
            pos += i;
            if (pos == 0) output++;
        }
        return output;
    }


    public int longestString(int x, int y, int z) {
        int mn = Math.min(x, y);
        if (x == y) {
            return (4 * x + 2 * z);
        } else {
            return (2 * mn + (mn + 1) * 2 + 2 * z);
        }
    }

    public long zeroFilledSubarray(int[] nums) {
        long output = 0;
        long counter = 0;
        for (int i : nums) {
            if (i != 0) {
                counter = 0;
                continue;
            }
            counter++;
            output += counter;
        }
        return output;
    }

    public long countSubarrays(int[] nums) {
        int prior = Integer.MIN_VALUE;
        long output = 0;
        long counter = 0;
        for (int i : nums) {
            if (i <= prior) counter = 0;
            counter++;
            output += counter;
            prior = i;
        }
        return output;
    }

    public long maximumValueSum(int[] nums, int k, int[][] edges) {
        int n = nums.length;
        long[][] dp = new long[n + 1][2];
        //Base case: The last node cannot XOR by itself
        //Let dp[i][1] mean xor, and dp[i][0] mean not to xor
        dp[n][0] = 0;
        dp[n][1] = Long.MIN_VALUE;
        for (int i = n - 1; i >= 0; i--) {
            long num = nums[i];
            long numXOR = num ^ k;
            dp[i][0] = Math.max(numXOR + dp[i + 1][1], num + dp[i + 1][0]);
            dp[i][1] = Math.max(numXOR + dp[i + 1][0], num + dp[i + 1][1]);
        }
        return dp[0][0];
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
        int right = n - 1;
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
        int[] cache = new int[n + 1];
        for (int i = n - 1; i >= 0; i--) {
            char a = chars[i];
            for (int j = i + 1; j < n; j++) {
                char b = chars[j];
                if (a != b) {
                    if (j == (i + 1)) dp[i][j] = 1;
                    else dp[i][j] = dp[i + 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i + 1][j - 1];
                }
            }
            cache[i] = dp[i][n - 1];
        }
        for (int i = 1; i < k; i++) {
            int upperBound = n - 1 - i;
            int[] currentCache = new int[n];
            for (int j = upperBound; j >= 0; j--) {
                currentCache[j] = Integer.MAX_VALUE;
                for (int l = j; l <= upperBound; l++) {
                    currentCache[j] = Math.min(currentCache[j], dp[j][l] + cache[l + 1]);
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




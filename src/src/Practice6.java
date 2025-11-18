import TreeXMLExample.*;

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

        practice6.partition("aab");

        int[] p = {0, 5, 7, 8, 10, 16, 17, 18, 16, 20, 27, 29};
        int[] out = practice6.rodCutting(11, p);
        System.out.println(Arrays.toString(out));


    }

    /**
     * Main Method
     * z
     * ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
     * ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
     * ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
     * ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
     * ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
     * ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
     * ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
     * ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
     * ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
     * ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
     * ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
     */

    public int superEggDrop(int k, int n) {
        int[][] dp = new int[n + 1][k + 1];
        for (int i = 1; i <= k; i++) dp[1][i] = 1;
        for (int i = 1; i <= n; i++) dp[i][1] = i;
        for (int j = 2; j <= k; j++) {
            for (int i = 2; i <= n; i++) {
                int currentMin = n;
                for (int l = 1; l <= i; l++) {
                    int worstCase = 1 + Math.max(dp[l - 1][j - 1], dp[i - l][j]);
                    currentMin = Math.min(currentMin, worstCase);
                }
                dp[i][j] = currentMin;
            }
        }
        return dp[n][k];
    }

    public String clearDigits(String s) {
        int n = s.length();
        char[] sChar = s.toCharArray();
        int counter = 0;
        StringBuilder sb = new StringBuilder();
        for (int i = n - 1; i >= 0; i--) {
            char c = sChar[i];
            if (Character.isDigit(c)) {
                counter++;
            } else {
                if (counter == 0) sb.append(c);
                else counter--;
            }
        }
        return sb.reverse().toString();
    }


    public int minCost(int n, int[] cuts) {
        Arrays.sort(cuts);
        List<Integer> cutList = new ArrayList<>();
        cutList.add(0);
        for (int i : cuts) cutList.add(i);
        cutList.add(n);
        int m = cutList.size();
        int[][] dp = new int[m][m];
        for (int i = m - 3; i >= 0; i--) {
            for (int j = i + 2; j < m; j++) {
                int currentMin = Integer.MAX_VALUE;
                for (int k = i + 1; k < j; k++) {
                    currentMin = Math.min(currentMin, dp[i][k] + dp[k][j]);
                }
                currentMin += cutList.get(j) - cutList.get(i);
                dp[i][j] = currentMin;
            }
        }
        return dp[0][m - 1];
    }


    public int[] rodCutting(int n, int[] p) {
        // If the rod length is <= 1, return the corresponding price directly.
        // Array to store the maximum values for each length up to n.
        int[] v = new int[n + 1];
        v[1] = p[1];

        // Compute v[k] for increasing values of k, from 2 to n.
        for (int k = 2; k <= n; k++) {
            int runningMax = p[k]; // Initialize with the value of not making a cut.

            // Try all possible cuts at position y (1 <= y < k).
            for (int y = 1; y < k; y++) {
                // Left part has length y, right part has length (k - y - 1).
                if (k - y - 1 >= 0) {
                    runningMax = Math.max(runningMax, p[y] + v[k - y - 1]);
                }
            }

            // Store the maximum value for the rod length k.
            v[k] = runningMax;
        }

        // Return the maximum value for the full rod length n.
        return v;
    }

    public boolean arrayStringsAreEqual(String[] word1, String[] word2) {
        StringBuilder sb1 = new StringBuilder();
        StringBuilder sb2 = new StringBuilder();
        for (String s : word1) sb1.append(s);
        for (String s : word2) sb2.append(s);
        return sb1.toString().equals(sb2.toString());
    }

    public int canBeTypedWords(String text, String brokenLetters) {
        boolean[] charFlags = new boolean[26];
        for (char c : brokenLetters.toCharArray()) {
            charFlags[c - 'a'] = true;
        }
        String[] words = text.split(" ");
        int output = 0;
        for (String word : words) {
            boolean flag = true;
            for (char c : word.toCharArray()) {
                if (charFlags[c - 'a']) {
                    flag = false;
                    break;
                }
            }
            if (flag) output++;
        }
        return output;
    }

    public String getEncryptedString(String s, int k) {
        int n = s.length();
        StringBuilder encrypted = new StringBuilder();
        for (int i = 0; i < n; i++) {
            int newIndex = (i + k) % n;
            encrypted.append(s.charAt(newIndex));
        }

        return encrypted.toString();
    }

    public int[] resultsArray(int[][] queries, int k) {
        int n = queries.length;
        int[] output = new int[n];
        PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());
        for (int i = 0; i < n; i++) {
            int a = queries[i][0];
            int b = queries[i][1];
            int distance = Math.abs(a) + Math.abs(b);
            pq.add(distance);
            while (pq.size() > k) pq.poll();
            int out = (pq.size() < k) ? -1 : pq.peek();
            output[i] = out;
        }
        return output;
    }


    public List<Integer> stableMountains(int[] height, int threshold) {
        int n = height.length;
        List<Integer> output = new ArrayList<>();
        for (int i = 1; i < n; i++) {
            if (height[i - 1] > threshold) output.add(i);
        }
        return output;
    }


    public int[] getSneakyNumbers(int[] nums) {
        int[] output = new int[2];
        int i = 0;
        BitSet bs = new BitSet();
        for (int n : nums) {
            if (bs.get(n)) {
                output[i++] = n;
            } else {
                bs.set(n);
            }
        }
        return output;
    }


    public int countWinningSequences(String s) {
        int n = s.length();
        int[] aSequence = new int[n];
        char[] sChar = s.toCharArray();
        for (int i = 0; i < n; i++) {
            char c = sChar[i];
            if (c == 'E') aSequence[i] = 0;
            else if (c == 'F') aSequence[i] = 1;
            else aSequence[i] = 2;
        }
        int aFirstPlay = aSequence[0];
        HashMap<Integer, Long>[] dp = new HashMap[3];
        for (int i = 0; i < 3; i++) {
            dp[i] = new HashMap<>();
            int score = bobWinScore(aFirstPlay, i);
            dp[i].put(score, 1l);
        }
        for (int i = 1; i < n; i++) {
            int currentA = aSequence[i];
            HashMap<Integer, Long>[] currentDP = new HashMap[3];
            for (int b = 0; b < 3; b++) {
                currentDP[b] = new HashMap<>();
                int score = bobWinScore(currentA, b);
                for (int bb = 0; bb < 3; bb++) {
                    if (bb == b) continue;
                    for (var entry : dp[bb].entrySet()) {
                        currentDP[b].merge(entry.getKey() + score, entry.getValue(), Long::sum);
                    }
                }
            }
            dp = currentDP;
        }
        int output = 0;
        for (var hm : dp) {
            for (var entry : hm.entrySet()) {
                if (entry.getKey() > 0) {
                    output += entry.getValue();
                    output %= 1_000_000_007;
                }
            }
        }
        return output;
    }

    public int bobWinScore(int a, int b) {
        if (a == b) return 0;
        if (b == 0) return (a == 1 ? -1 : 1);
        if (b == 1) return (a == 2 ? -1 : 1);
        if (b == 2) return (a == 0 ? -1 : 1);
        return Integer.MIN_VALUE;
    }


    List<Integer> treeSizeList;

    public int kthLargestPerfectSubtree(TreeNode root, int k) {
        treeSizeList = new ArrayList<>();
        treeDFS(root);
        Collections.sort(treeSizeList, Collections.reverseOrder());
        if (k <= treeSizeList.size()) {
            return treeSizeList.get(k - 1);
        } else {
            return -1;
        }
    }

    public int treeDFS(TreeNode node) {
        if (node == null) return 0;
        int left = treeDFS(node.left);
        int right = treeDFS(node.right);
        if (node.left == null && node.right == null) {
            treeSizeList.add(1);
            return 1;
        } else if (left > 0 && right > 0 && left == right) {
            int size = left + right + 1;
            treeSizeList.add(size);
            return size;
        } else {
            return 0;
        }
    }

    public long[] findXSum(int[] nums, int k, int x) {
        int n = nums.length;
        System.out.println(n);
        long[] output = new long[n - k + 1];
        HashMap<Long, Long> hm = new HashMap<>();
        for (int i = 0; i < k - 1; i++) {
            hm.merge((long) nums[i], 1l, Long::sum);
        }
        int K = k - 1;
        int index = 0;
        while (K < n) {
            hm.merge((long) nums[K], 1l, Long::sum);
            long sum = 0;
            List<Long> keyList = new ArrayList<>(hm.keySet());
            Collections.sort(keyList, (a, b) -> {
                int valueCompare = Long.compare(hm.get(b), hm.get(a));
                if (valueCompare != 0) {
                    return valueCompare;
                } else {
                    return Long.compare(b, a);
                }
            });
            for (int i = 0; i < Math.min(keyList.size(), x); i++) {
                sum += keyList.get(i) * hm.get(keyList.get(i));
            }
            output[index] = sum;
            hm.merge((long) nums[index], -1l, Long::sum);
            if (hm.get((long) nums[index]) == 0) hm.remove(nums[index]);
            K++;
            index++;
        }
        return output;
    }

    public int[] findXSum1(int[] nums, int k, int x) {
        int n = nums.length;
        int[] output = new int[n - k + 1];
        HashMap<Integer, Integer> hm = new HashMap<>();
        for (int i = 0; i < k - 1; i++) {
            hm.merge(nums[i], 1, Integer::sum);
        }
        int K = k - 1;
        int index = 0;
        while (K < n) {
            hm.merge(nums[K], 1, Integer::sum);
            int sum = 0;
            List<Integer> keyList = new ArrayList<>(hm.keySet());
            Collections.sort(keyList, (a, b) -> {
                int valueCompare = Integer.compare(hm.get(b), hm.get(a));
                if (valueCompare != 0) {
                    return valueCompare;
                } else {
                    return Integer.compare(b, a);
                }
            });
            for (int i = 0; i < Math.min(keyList.size(), x); i++) {
                sum += keyList.get(i) * hm.get(keyList.get(i));
            }
            output[index] = sum;
            hm.merge(nums[index], -1, Integer::sum);
            K++;
            index++;
        }
        return output;
    }


    public List<List<String>> partition(String s) {
        char[] sChar = s.toCharArray();
        int n = s.length();
        boolean[][] flagDP = new boolean[n][n];
        for (int i = n - 1; i >= 0; i--) {
            flagDP[i][i] = true;
            int left = i;
            int right = i + 1;
            while (left >= 0 && right < n && sChar[left] == sChar[right]) {
                flagDP[left][right] = true;
                left--;
                right++;
            }
            left = i - 1;
            right = i + 1;
            while (left >= 0 && right < n && sChar[left] == sChar[right]) {
                flagDP[left][right] = true;
                left--;
                right++;
            }
        }
        List<List<String>>[] listDP = new List[n + 1];
        for (int i = 0; i < n + 1; i++) {
            List<List<String>> doubleList = new ArrayList<>();
            listDP[i] = doubleList;
        }
        List<String> a = new ArrayList<>();
        List<List<String>> b = new ArrayList<>();
        b.add(a);
        listDP[n] = b;
        for (int i = n - 1; i >= 0; i--) {
            StringBuilder sb = new StringBuilder();
            for (int j = i; j < n; j++) {
                sb.append(sChar[j]);
                if (flagDP[i][j]) {
                    String currentPString = sb.toString();
                    List<List<String>> doubleList = listDP[j + 1];
                    for (List<String> stringList : doubleList) {
                        stringList.add(0, currentPString);
                        listDP[i].add(stringList);
                    }
                }
            }
        }
        return listDP[0];
    }


    public int maxGoodNumber(int[] nums) {
        List<String> binaryStringList = new ArrayList<>();
        for (int i : nums) {
            binaryStringList.add(Integer.toBinaryString(i));
        }
        Collections.sort(binaryStringList, new Comparator<String>() {
            public int compare(String a, String b) {
                String ab = a + b;
                String ba = b + a;
                return ba.compareTo(ab);
            }

        });
        StringBuilder sb = new StringBuilder();
        for (String s : binaryStringList)
            sb.append(s);
        return (Integer.parseInt(sb.toString(), 2));
    }


    public int minAddToMakeValid(String s) {
        int count = 0;
        int output = 0;
        for (char c : s.toCharArray()) {
            if (c == '(') {
                count++;
            } else {
                count--;
                if (count < 0) {
                    count = 0;
                    output++;
                }
            }
        }
        output += count;
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

    public ListNode removeElements(ListNode head, int val) {
        ListNode start = new ListNode();
        start.next = head;
        ListNode current = start;
        while (current.next != null) {
            ListNode next = current.next;
            if (next.val == val) {
                current.next = next.next;
            } else {
                current = current.next;
            }
        }
        return start.next;
    }


    public int minLength(String s) {
        Deque<Character> dq = new ArrayDeque<>();
        for (Character c : s.toCharArray()) {
            switch (c) {
                case 'B' -> {
                    if (!dq.isEmpty() && dq.peekLast() == 'A') dq.pollLast();
                    else dq.addLast(c);
                }
                case 'D' -> {
                    if (!dq.isEmpty() && dq.peekLast() == 'C') dq.pollLast();
                    else dq.addLast(c);
                }
                default -> {
                    dq.addLast(c);
                }
            }
        }
        return dq.size();
    }


    public boolean areSentencesSimilar(String sentence1, String sentence2) {
        String[] words1 = sentence1.split(" ");
        String[] words2 = sentence2.split(" ");

        int prefixMatch = 0;
        while (prefixMatch < words1.length && prefixMatch < words2.length &&
                words1[prefixMatch].equals(words2[prefixMatch])) {
            prefixMatch++;
        }
        int suffixMatch = 0;
        while (suffixMatch < words1.length - prefixMatch && suffixMatch < words2.length - prefixMatch &&
                words1[words1.length - 1 - suffixMatch].equals(words2[words2.length - 1 - suffixMatch])) {
            suffixMatch++;
        }
        return prefixMatch + suffixMatch >= Math.min(words1.length, words2.length);
    }


    public boolean evaluateTree(TreeNode root) {
        boolean leafNode = (root.left == null && root.right == null);
        if (leafNode) return (root.val == 1) ? true : false;
        if (root.val == 2) return (evaluateTree(root.left) || evaluateTree(root.right));
        else return (evaluateTree(root.left) && evaluateTree(root.right));
    }

    public String[] findRelativeRanks(int[] score) {
        int n = score.length;
        List<Integer> list = new ArrayList<>(Arrays.stream(score).boxed().toList());
        Collections.sort(list, Collections.reverseOrder());
        HashMap<Integer, String> hm = new HashMap<>();
        for (int i = 0; i < n; i++) {
            Integer key = list.get(i);
            String value = "";
            if (i == 0) value = "Gold Medal";
            else if (i == 1) value = "Silver Medal";
            else if (i == 2) value = "Bronze Medal";
            else value = Integer.toString(i + 1);
            hm.put(key, value);
        }
        String[] output = new String[n];
        for (int i = 0; i < n; i++) {
            output[i] = hm.get(score[i]);
        }
        return output;
    }

    public int[] frequencySort(int[] nums) {
        int n = nums.length;
        HashMap<Integer, Integer> hm = new HashMap<>();
        List<Integer> list = new ArrayList<>();
        for (int i : nums) {
            hm.merge(i, 1, Integer::sum);
            list.add(i);
        }
        Collections.sort(list, (a, b) -> {
            int freqA = hm.get(a);
            int freqB = hm.get(b);
            if (freqA == freqB) {
                return Integer.compare(b, a);
            }
            return Integer.compare(freqA, freqB);
        });
        for (int i = 0; i < n; i++) {
            nums[i] = list.get(i);
        }
        return nums;
    }

    public int[] arrayRankTransform(int[] arr) {
        int[] arrClone = arr.clone();
        Arrays.sort(arrClone);
        HashMap<Integer, Integer> hm = new HashMap<>();
        for (int i : arrClone) {
            if (!hm.containsKey(i)) {
                hm.put(i, hm.size() + 1);
            }
        }
        for (int i = 0; i < arr.length; i++) {
            arr[i] = hm.get(arr[i]);
        }
        return arr;
    }

    public boolean threeConsecutiveOdds(int[] arr) {
        int counter = 0;
        for (int i : arr) {
            if (i % 2 != 0) {
                counter++;
                if (counter == 3) return true;
            } else {
                counter = 0;
            }
        }
        return false;
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
            int previous = nums[i - 1];
            int current = nums[i];
            int currentPatternChar = Integer.compare(current, previous) + 1;
            if (sb.length() >= p) {
                sb.delete(0, 1);
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
            int mStart = (m - 1) - ((n - 1) - i);
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
            for (int j = i + 1; j < n; j++) {
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

    public int maximumLengthSubstring(String s) {
        int output = 0;
        int n = s.length();
        int[] frequencyCount = new int[26];
        char[] sChar = s.toCharArray();
        int left = 0;
        for (int i = 0; i < n; i++) {
            char c = sChar[i];
            frequencyCount[c - 'a']++;
            while (frequencyCount[c - 'a'] > 2) {
                frequencyCount[sChar[left] - 'a']--;
                left++;
            }
            output = Math.max(output, i - left + 1);
        }
        return output;
    }

    public int finalPositionOfSnake(int n, List<String> commands) {
        int y = 0;
        int x = 0;
        for (String s : commands) {
            switch (s) {
                case "LEFT" -> x--;
                case "RIGHT" -> x++;
                case "UP" -> y--;
                default -> y++;
            }
        }
        return (y * n) + x;
    }

    public int minElement(int[] nums) {
        int minimum = Integer.MAX_VALUE;
        for (int i : nums) {
            int currentSum = 0;
            while (i > 0) {
                currentSum += (i % 10);
                i /= 10;
            }
            minimum = Math.min(minimum, currentSum);

        }
        return minimum;
    }

    public int generateKey(int num1, int num2, int num3) {
        Deque<Integer> dq = new ArrayDeque<>();
        for (int i = 0; i < 4; i++) {
            int currentDigit = Integer.MAX_VALUE;
            currentDigit = Math.min(currentDigit, (num1 % 10));
            currentDigit = Math.min(currentDigit, (num2 % 10));
            currentDigit = Math.min(currentDigit, (num3 % 10));
            num1 /= 10;
            num2 /= 10;
            num3 /= 10;
            dq.push(currentDigit);
        }
        int output = 0;
        while (!dq.isEmpty()) {
            output = output * 10 + dq.pollFirst();
        }
        return output;
    }

    public List<String> stringSequence(String target) {
        char[] chars = target.toCharArray();
        List<String> output = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        for (char c : chars) {
            sb.append('a');
            output.add(sb.toString());
            while (sb.charAt(sb.length() - 1) != c) {
                sb.setCharAt(sb.length() - 1, (char) (sb.charAt(sb.length() - 1) + 1));
                output.add(sb.toString());
            }
        }
        return output;
    }


    public int countKConstraintSubstrings(String s, int k) {
        char[] chars = s.toCharArray();
        int n = chars.length;
        int zeros = 0;
        int ones = 0;
        int left = 0;
        int output = 0;
        for (int right = 0; right < n; right++) {
            char c = chars[right];
            if (c == '0') zeros++;
            else ones++;
            while (zeros > k && ones > k) {
                if (chars[left] == '0') zeros--;
                else ones--;
                left++;
            }
            output += right - left + 1;
        }
        return output;
    }


    public int[] findingUsersActiveMinutes(int[][] logs, int k) {
        int n = logs.length;
        HashMap<Integer, HashSet<Integer>> hm = new HashMap<>();
        int[] output = new int[k];
        for (int[] log : logs) {
            int id = log[0];
            int time = log[1];
            if (!hm.containsKey(id)) hm.put(id, new HashSet<>());
            hm.get(id).add(time);
        }
        for (var v : hm.values()) {
            output[v.size() - 1]++;
        }
        return output;
    }

    public List<String> buildArray(int[] target, int n) {
        List<String> output = new ArrayList<>();
        int counter = 1;
        for (int i : target) {
            while (counter < i) {
                output.add("Push");
                output.add("Pop");
                counter++;
            }
            output.add("Push");
            counter++;
        }
        return output;
    }

    public int countDistinctIntegers(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        for (int i : nums) {
            set.add(i);
            set.add(reverseNumber(i));
        }
        return set.size();
    }

    public int reverseNumber(int i) {
        int output = 0;
        while (i != 0) {
            output *= 10;
            output += (i % 10);
            i /= 10;
        }
        return output;
    }

    private HashSet<Integer> targetNodeValues;
    private TreeNode lcaNode;

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode[] nodes) {
        targetNodeValues = new HashSet<>();
        for (TreeNode node : nodes) {
            targetNodeValues.add(node.val);
        }
        lcaNode = null;
        findLCA(root);
        return lcaNode;
    }

    private int findLCA(TreeNode root) {
        if (root == null) return 0;
        int leftCount = findLCA(root.left);
        int rightCount = findLCA(root.right);
        int currentCount = 0;
        if (targetNodeValues.contains(root.val)) {
            currentCount = 1;
        }
        int totalCount = leftCount + rightCount + currentCount;
        if (totalCount == targetNodeValues.size() && lcaNode == null) {
            lcaNode = root;
        }
        return totalCount;
    }


    public double maxAmount(String initialCurrency,
                            List<List<String>> pairs1,
                            double[] rates1,
                            List<List<String>> pairs2,
                            double[] rates2) {
        int p1 = pairs1.size(), p2 = pairs2.size();
        boolean flag1 = true, flag2 = true;
        HashMap<String, Double> hm = new HashMap<>();
        hm.put(initialCurrency, 1d);
        while (flag1) {
            flag1 = false;
            for (int i = 0; i < p1; i++) {
                var pair = pairs1.get(i);
                String currencyA = pair.get(0);
                String currencyB = pair.get(1);
                double rate = rates1[i];
                if (hm.containsKey(currencyA) &&
                        (!hm.containsKey(currencyB) || (hm.get(currencyB) < hm.get(currencyA) * rate))) {
                    flag1 = true;
                    hm.put(currencyB, hm.get(currencyA) * rate);
                } else if (hm.containsKey(currencyB) &&
                        (!hm.containsKey(currencyA) || (hm.get(currencyA) < hm.get(currencyB) * (1 / rate)))) {
                    flag1 = true;
                    hm.put(currencyA, hm.get(currencyB) * (1 / rate));
                }
            }
        }

        while (flag2) {
            flag2 = false;
            for (int i = 0; i < p2; i++) {
                var pair = pairs2.get(i);
                String currencyA = pair.get(0);
                String currencyB = pair.get(1);
                double rate = rates2[i];
                if (hm.containsKey(currencyA) &&
                        (!hm.containsKey(currencyB) || (hm.get(currencyB) < hm.get(currencyA) * rate))) {
                    flag2 = true;
                    hm.put(currencyB, hm.get(currencyA) * rate);
                } else if (hm.containsKey(currencyB) &&
                        (!hm.containsKey(currencyA) || (hm.get(currencyA) < hm.get(currencyB) * (1 / rate)))) {
                    flag2 = true;
                    hm.put(currencyA, hm.get(currencyB) * (1 / rate));
                }
            }
        }
        return hm.get(initialCurrency);
    }

    public int[] xorQueries(int[] arr, int[][] queries) {
        int n = arr.length;
        int m = queries.length;
        int[] output = new int[m];
        int currentXOR = 0;
        for (int i = 0; i < n; i++) {
            currentXOR = currentXOR ^ arr[i];
            arr[i] = currentXOR;
        }
        for (int i = 0; i < m; i++) {
            int[] query = queries[i];
            int a = query[0];
            int b = query[1];
            int left = (a == 0) ? 0 : arr[a - 1];
            int right = arr[b];
            output[i] = left ^ right;
        }
        return output;
    }

    public int maxScore(int n, int k, int[][] stayScore, int[][] travelScore) {
        int[] dp = new int[n];
        while (k-- > 0) {
            int[] currentDP = dp.clone();
            for (int i = 0; i < n; i++) {
                currentDP[i] += stayScore[k][i];
                for (int j = 0; j < n; j++) {
                    currentDP[i] = Math.max(currentDP[i], dp[j] + travelScore[i][j]);
                }
            }
            dp = currentDP;
        }
        return Arrays.stream(dp).max().getAsInt();
    }

    public boolean isPossibleToRearrange(String s, String t, int k) {
        int n = s.length();
        int substringLength = n / k;
        HashMap<String, Integer> hm = new HashMap<>(k << 2);
        for (int i = 0; i < n; i += substringLength) {
            String substringS = s.substring(i, i + substringLength);
            hm.merge(substringS, 1, Integer::sum);
        }
        for (int i = 0; i < n; i += substringLength) {
            String substringT = t.substring(i, i + substringLength);
            if (hm.getOrDefault(substringT, 0) <= 0) return false;
            else hm.merge(substringT, -1, Integer::sum);
        }
        return true;
    }

    public int deleteGreatestValue(int[][] grid) {
        int n = grid.length;
        int m = grid[0].length;
        int output = 0;
        for (int[] g : grid) Arrays.sort(g);
        for (int i = 0; i < m; i++) {
            int max = Integer.MIN_VALUE;
            for (int j = 0; j < n; j++) {
                max = Math.max(max, grid[j][i]);
            }
            output += max;
        }
        return output;
    }


    public int longestCommonPrefix(String s, String t) {
        int sLength = s.length();
        int tLength = t.length();
        char[] sChars = s.toCharArray();
        char[] tChars = t.toCharArray();
        int sIndex = 0;
        int tIndex = 0;
        boolean flag = false;
        while (sIndex < sLength && tIndex < tLength) {
            if (sChars[sIndex] != tChars[tIndex]) {
                if (flag) break;
                else {
                    flag = true;
                    sIndex++;
                }
            } else {
                sIndex++;
                tIndex++;
            }
        }
        return tIndex;
    }


    public long minCost(int[] arr, int[] brr, long k) {
        int n = arr.length;
        long output1 = 0;
        long output2 = 0;
        for (int i = 0; i < n; i++) output1 += Math.abs(arr[i] - brr[i]);
        Arrays.sort(arr);
        Arrays.sort(brr);
        for (int i = 0; i < n; i++) output1 += Math.abs(arr[i] - brr[i]);
        output2 = Math.abs(output2) + k;
        return Math.min(output1, output2);
    }


    public int minimumDeletions(String s) {
        int n = s.length();
        char[] sChar = s.toCharArray();
        int a = 0;
        int b = 0;
        for (int i = 0; i < n; i++) {
            char c = sChar[i];
            switch (c) {
                case 'a' -> {
                    b = Math.min(a, b) + 1;
                }
                case 'b' -> {
                    b = Math.min(a, b);
                    a++;
                }
            }
        }
        return Math.min(a, b);
    }

    public int numberOfSets(int n, int k) {
        int[] dp = new int[n + 1];
        Arrays.fill(dp, 1);
        int MOD = 1_000_000_007;
        while (k-- > 0) {
            int[] currentDP = new int[n + 1];
            for (int i = n - 2; i >= 0; i--) {
                currentDP[i] = currentDP[i + 1];
                for (int j = n - 1; j > i; j--) {
                    currentDP[i] += dp[j + 1];
                    currentDP[i] %= MOD;
                }
            }
            System.out.println(Arrays.toString(currentDP));
            dp = currentDP;
        }
        return dp[0];
    }


    public long getDescentPeriods(int[] prices) {
        int n = prices.length;
        long output = 0;
        long currentLength = 0;
        for (int i = 0; i < n; i++) {
            if (currentLength == 0) {
                currentLength++;
            } else {
                if (prices[i] == (prices[i - 1] - 1)) {
                    currentLength++;
                } else {
                    output += (currentLength * (currentLength + 1)) / 2;
                    currentLength = 1;
                }
            }
        }
        output += (currentLength * (currentLength + 1)) / 2;
        return output;
    }


    public long minIncrementOperations(int[] nums, int k) {
        int n = nums.length;
        long[] dp = new long[n];
        for (int i = 0; i < n; i++) {
            dp[i] = (nums[i] < k) ? (k - nums[i]) : 0;
            if (i >= 3) {
                dp[i] += Math.min(Math.min(dp[i - 1], dp[i - 2]), dp[i - 3]);
            }
        }
        return Math.min(Math.min(dp[n - 1], dp[n - 2]), dp[n - 3]);
    }

    public long minCost(int n, int[][] cost) {
        long[][] minCost = new long[3][3];
        for (int i = (n / 2) - 1; i >= 0; i--) {
            int j = (n - 1) - i;
            long[][] currentCosts = new long[3][3];
            for (int k = 0; k < 3; k++) Arrays.fill(currentCosts[k], Long.MAX_VALUE);

            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    if (l == k) continue;
                    for (int m = 0; m < 3; m++) {
                        if (m == k) continue;
                        for (int o = 0; o < 3; o++) {
                            if (o == m || o == l) continue;
                            if (minCost[m][o] != Long.MAX_VALUE) {
                                long currentCost = cost[i][k] + cost[j][l] + minCost[m][o];
                                currentCosts[k][l] = Math.min(currentCosts[k][l], currentCost);
                            }
                        }
                    }
                }
            }
            minCost = currentCosts;
        }
        long output = Long.MAX_VALUE;
        for (int k = 0; k < 3; k++) {
            for (int l = 0; l < 3; l++) {
                if (k == l) continue;
                output = Math.min(output, minCost[k][l]);
            }
        }
        return output;
    }


    public int maxSum(int[] nums, int k, int m) {
        int n = nums.length;
        long[] dp = new long[n + 1];
        dp[n] = 0;
        long[] prefixSum = new long[n];
        prefixSum[0] = nums[0];
        for (int i = 1; i < n; i++) {
            prefixSum[i] = prefixSum[i - 1] + nums[i];
        }
        int counter = 0;
        while (counter < k) {
            long[] currentDP = new long[n + 1];
            Arrays.fill(currentDP, Long.MIN_VALUE);
            int upperBound = n - (counter * m);
            int lowerBound = ((k - (counter + 1)) * m);
            for (int i = upperBound - m; i >= lowerBound; i--) {
                for (int j = i + (m - 1); j < upperBound; j++) {
                    long currentSum = prefixSum[j];
                    if (i > 0) currentSum -= prefixSum[i - 1];
                    currentDP[i] = Math.max(currentDP[i], currentSum + dp[j + 1]);
                }
                currentDP[i] = Math.max(currentDP[i], currentDP[i + 1]);
            }
            dp = currentDP;
            counter++;

        }
        return (int) dp[0];
    }


    public int maxAbsoluteSum(int[] nums) {
        int output = 0;
        int minSum = 0;
        int maxSum = 0;
        for (int i : nums) {
            minSum += i;
            maxSum += i;
            output = Math.max(output, Math.max(Math.abs(minSum), Math.abs(maxSum)));
            if (minSum > 0) minSum = 0;
            if (maxSum < 0) maxSum = 0;
        }
        return output;
    }


    public int maximumMatchingIndices(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int output = 0;
        for (int i = 0; i < n; i++) {
            int counter = 0;
            for (int j = 0; j < n; j++) {
                int index = (j + i) % n;
                if (nums1[index] == nums2[j]) counter++;
            }
            output = Math.max(output, counter);
        }
        return output;
    }


    public int[] processQueries(int[] queries, int m) {
        int n = queries.length;
        int[] output = new int[n];
        LinkedList<Integer> ll = new LinkedList<>();
        for (int i = 1; i <= m; i++) ll.addLast(i);
        for (int i = 0; i < n; i++) {
            int index = ll.indexOf(queries[i]);
            output[i] = index;
            ll.remove(index);
            ll.addFirst(queries[i]);
        }
        return output;
    }


    public long calculateScore(String s) {
        int n = s.length();
        long output = 0;
        char[] sChars = s.toCharArray();
        Deque<Integer>[] dqArray = new ArrayDeque[26];
        for (int i = 0; i < 26; i++) dqArray[i] = new ArrayDeque<Integer>();
        for (int i = 0; i < n; i++) {
            char c = sChars[i];
            int charValue = c - 'a';
            int mirrorValue = 25 - charValue;
            Deque<Integer> dq = dqArray[mirrorValue];
            if (dq.isEmpty()) {
                dqArray[charValue].addFirst(i);
            } else {
                output += (i - dq.pollFirst());
            }
        }
        return output;
    }

    public long[] unmarkedSumArray(int[] nums, int[][] queries) {
        int n = nums.length;
        int m = queries.length;
        long sum = 0;
        long[] output = new long[m];
        PriorityQueue<Integer> pq = new PriorityQueue<>(
                Comparator.comparingInt((Integer i) -> nums[i]) // Primary comparison: values in nums
                        .thenComparingInt(i -> i)         // Secondary comparison: the indices themselves
        );
        for (int i = 0; i < n; i++) {
            pq.offer(i);
            sum += nums[i];
        }
        for (int i = 0; i < m; i++) {
            int[] query = queries[i];
            int a = query[0];
            int b = query[1];
            sum -= nums[a];
            nums[a] = 0;
            while (!pq.isEmpty() && b > 0) {
                int currentIndex = pq.poll();
                if (nums[currentIndex] == 0) continue;
                else {
                    b--;
                    sum -= nums[currentIndex];
                    nums[currentIndex] = 0;
                }
            }
            output[i] = sum;
        }
        return output;
    }

    int LBSTSCount = 0;

    public int largestBSTSubtree(TreeNode root) {
        LBSTSCount = 0;
        LBSTS(root);
        return LBSTSCount;
    }

    public int[] LBSTS(TreeNode root) {
        if (root == null) {
            return new int[]{Integer.MAX_VALUE, Integer.MIN_VALUE, 0};
        }
        int[] left = LBSTS(root.left);
        int[] right = LBSTS(root.right);
        boolean flag = (left[2] == -1) || (right[2] == -1) || (root.val <= left[1]) || (root.val >= right[0]);
        if (flag) {
            return new int[]{Integer.MAX_VALUE, Integer.MIN_VALUE, -1};
        } else {
            int counter = left[2] + right[2] + 1;
            LBSTSCount = Math.max(LBSTSCount, counter);
            return new int[]{Math.min(root.val, left[0]), Math.max(root.val, right[1]), counter};
        }
    }

    public int lengthAfterTransformations(String s, int t) {
        long output = 0;
        int MOD = 1_000_000_007;
        int[] charFrequency = new int[26];
        for (char c : s.toCharArray()) {
            charFrequency[c - 'a']++;
        }
        Deque<Long> dq = new ArrayDeque<>();
        for (int i = 0; i < 26; i++) dq.offer(1L);
        int counter = 0;
        while (counter++ < t) {
            dq.addLast((dq.pollFirst() + dq.peekFirst()) % MOD);
        }
        for (int i = 0; i < 26; i++) {
            output += charFrequency[i] * dq.pollFirst();
            output %= MOD;
        }
        return (int) output;
    }


    public int maxDistinctElements(int[] nums, int k) {
        if (k >= nums.length) return nums.length;
        int counter = 0;
        int minReq = Integer.MIN_VALUE;
        Arrays.sort(nums);
        for (int i : nums) {
            int minimum = i - k;
            int maximum = i + k;
            if (maximum <= minReq) continue;
            counter++;
            minReq = Math.max(minReq + 1, minimum);
        }
        return counter;
    }

    public int maxIncreasingSubarrays(List<Integer> nums) {
        int n = nums.size();
        int output = 0;
        Deque<Integer> dq = new ArrayDeque<>();
        dq.add(0);
        int prior = Integer.MAX_VALUE;
        int counter = 0;
        for (int i = n - 1; i > 0; i--) {
            int num = nums.get(i);
            if (num < prior) {
                counter++;
            } else {
                counter = 1;
            }
            dq.addFirst(counter);
            prior = num;
        }
        prior = Integer.MIN_VALUE;
        counter = 0;
        for (int num : nums) {
            if (num > prior) {
                counter++;
            } else {
                counter = 1;
            }
            prior = num;
            output = Math.max(output, Math.min(counter, dq.pollFirst()));
        }
        return output;
    }


    public int countPaths(int n, int[][] roads) {
        int MOD = 1_000_000_007;
        int[][] graph = new int[n][n];
        int[] graphMask = new int[n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(graph[i], Integer.MAX_VALUE);
        }
        for (int[] road : roads) {
            int a = road[0];
            int b = road[1];
            int c = road[2];
            graphMask[a] |= (1 << b);
            graphMask[b] |= (1 << a);
            graph[a][b] = c;
            graph[b][a] = c;
        }
        long[] minTimes = new long[n];
        Arrays.fill(minTimes, Long.MAX_VALUE);
        minTimes[0] = 0;
        long[] ways = new long[n];
        ways[0] = 1;
        // int[]{currentNode, currentTime}
        PriorityQueue<long[]> pq = new PriorityQueue<>(Comparator.comparingLong(a -> a[1]));
        pq.add(new long[]{0, 0});
        while (!pq.isEmpty()) {
            long[] current = pq.poll();
            int currentNode = (int) current[0];
            long currentTime = current[1];
            if (currentTime > minTimes[currentNode]) {
                continue;
            }
            int currentNodeGraphMask = graphMask[currentNode];
            for (int i = 0; i < n; i++) {
                if ((currentNodeGraphMask & (1 << i)) == 0 || graph[currentNode][i] == Integer.MAX_VALUE) continue;
                long arrivalTime = currentTime + (long) graph[currentNode][i];
                if (arrivalTime < minTimes[i]) {
                    minTimes[i] = arrivalTime;
                    ways[i] = ways[currentNode];
                    pq.offer(new long[]{i, arrivalTime});
                } else if (arrivalTime == minTimes[i]) {
                    ways[i] = (ways[i] + ways[currentNode]) % MOD;
                }
            }
        }
        return (int) ways[n - 1] % MOD;
    }


    public int addMinimum(String word) {
        char[] pattern = new char[]{'a', 'b', 'c'};
        int output = 0;
        int patternIndex = 0;
        for (char c : word.toCharArray()) {
            while (c != pattern[patternIndex]) {
                output++;
                patternIndex = (patternIndex + 1) % 3;
            }
            patternIndex = (patternIndex + 1) % 3;
        }
        output += (3 - patternIndex) % 3;
        return output;
    }




    public int countPartitions(int[] nums) {
        int n = nums.length;
        long sumA = 0;
        long sumB = Arrays.stream(nums).sum();
        int output = 0;
        for (int i = 0; i < n-1; i++) {
            sumA += nums[i];
            sumB -= nums[i];
            if ((sumA - sumB) % 2 == 0) output++;

        }
        return output;
    }


    public long calculateScore(String[] instructions, int[] values) {
        int n = values.length;
        int i = 0;
        long output = 0;
        while (i >= 0
                && i < n
                && instructions[i] != null) {
            String s = instructions[i];
            instructions[i] = null;
            if (s.equals("add")) {
                output += values[i];
                i++;
            } else {
                i += values[i];
            }
        }
        return output;
    }

    public int maxContainers(int n, int w, int maxWeight) {
        int a = maxWeight/w;
        int b = n * n;
        return Math.min(a,b);
    }


//    public boolean[] pathExistenceQueries(int n, int[] nums, int maxDiff, int[][] queries) {
//        int m = queries.length;
//        boolean[] output = new boolean[m];
//        int[] minNodes = new int[n];
//        int minNode = 0;
//        for (int i = 1; i < n; i++) {
//            int currentNodeValue = nums[i];
//            if ((currentNodeValue - nums[i-1]) <= maxDiff) {
//                minNodes[i] = minNodes[i-1];
//            } else {
//                minNodes[i] = i;
//            }
//        }
//        for (int i = 0; i < m; i++) {
//            int[] query = queries[i];
//            int smaller = Math.min(query[0], query[1]);
//            int bigger = Math.max(query[0], query[1]);
//            output[i] = (smaller >= minNodes[bigger]);
//        }
//        return output;
//    }

    public int countCoveredBuildings(int n, int[][] buildings) {
        int output = 0;
        HashMap<Integer, int[]> xMap = new HashMap<>();
        HashMap<Integer, int[]> yMap = new HashMap<>();
        for (int[] building : buildings) {
            int x = building[0];
            int y = building[1];
            int[] yRange = xMap.getOrDefault(x, new int[]{y,y});
            yRange[0] = Math.min(yRange[0], y);
            yRange[1] = Math.max(yRange[1], y);
            xMap.put(x, yRange);
            int[] xRange = yMap.getOrDefault(y, new int[]{x,x});
            xRange[0] = Math.min(xRange[0], x);
            xRange[1] = Math.max(xRange[1], x);
            yMap.put(y, xRange);
        }
        for (int[] building : buildings) {
            int x = building[0];
            int y = building[1];
            int[] yRange = xMap.get(x);
            int[] xRange = yMap.get(y);
            if (yRange[0] < y && y < yRange[1] && xRange[0] < x && x < xRange[1]) output++;
        }
        return output;
    }






}




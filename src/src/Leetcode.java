import java.util.*;

public class Leetcode {
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

    class QuadNode {
        public boolean val;
        public boolean isLeaf;
        public QuadNode topLeft;
        public QuadNode topRight;
        public QuadNode bottomLeft;
        public QuadNode bottomRight;

        public QuadNode() {
        }

        public QuadNode(boolean _val, boolean _isLeaf, QuadNode _topLeft, QuadNode _topRight, QuadNode _bottomLeft, QuadNode _bottomRight) {
            val = _val;
            isLeaf = _isLeaf;
            topLeft = _topLeft;
            topRight = _topRight;
            bottomLeft = _bottomLeft;
            bottomRight = _bottomRight;
        }
    }

    class Bomb {
        int x;
        int y;
        double power;
        HashSet<Bomb> bombsWithinRange;

        public Bomb(int x, int y, int power) {
            this.x = x;
            this.y = y;
            this.power = power;
            bombsWithinRange = new HashSet<>();
        }
    }

    TreeNode currentTN;

    // Definition for a Node.
    class Node {
        public int val;
        public Node prev;
        public Node next;
        public Node child;
        public Node random;
        public Node left;
        public Node right;
        public List<Node> neighbors;

        public Node(int val) {
            this.val = val;
            this.next = null;
            this.random = null;
        }


    }

    public static boolean canSplitArray(List<Integer> nums, int m) {
        //Subarray the list by taking away the head or tail, whichever is lower
        //Repeat until only size 2
        //By doing this, we are essentially only looking for two numbers next to each other, that sum to at least m
        if (nums.size() <= 2) return true;
        for (int i = 0; i < nums.size() - 1; i++) {
            if ((nums.get(i) + nums.get(i + 1)) >= m) return true;
        }
        return false;
    }

    public static List<String> ambiguousCoordinates(String s) {

        List<String> output = new ArrayList<>();
        for (int i = 1; i < s.length(); i++) {
            String a = s.substring(0, i);
            String b = s.substring(i);
            List<String> combinationA = combinationGenerator(a);
            List<String> combinationB = combinationGenerator(b);
            if (!combinationA.isEmpty() && !combinationB.isEmpty()) {
                for (String aa : combinationA) {
                    for (String bb : combinationB) {
                        String toAdd = "(" + aa + ", " + bb + ")";
                        output.add(toAdd);
                    }
                }
            }
        }
        return output;
    }

    public static List<String> combinationGenerator(String s) {
        List<String> output = new ArrayList<>();
        char c = s.charAt(0);
        //Rejection cases
        //Reject if all zeros
        if (Integer.valueOf(s) == 0 || (s.charAt(0) == '0' && s.charAt(s.length() - 1) == '0')) {
            return output;
        }
        //If it's a single digit
        if (s.length() == 1) {
            output.add(s);
            return output;
        }
        //If last char is 0, then it must be a singleton e.g 24930
        char cc = s.charAt(s.length() - 1);
        if (cc == '0') {
            output.add(s);
            return output;
        }
        //If first char is 0, then it must be singleton of 0.__
        if (c == '0') {
            output.add("0." + s.substring(1));
            return output;
        }

        for (int i = 1; i < s.length(); i++) {
            output.add(s.substring(0, i) + "." + s.substring(i));
        }
        return output;
    }

    public boolean btreeGameWinningMove(TreeNode root, int n, int x) {
        int upperPath = upperPathScore(root, x);
        TreeNode lowerTree = dfsFind(root, x);
        int lowerLeftPath = nodeCounter(lowerTree.left);
        int lowerRightPath = nodeCounter(lowerTree.right);
        int a = upperPath + lowerLeftPath;
        int b = upperPath + lowerRightPath;
        int c = lowerLeftPath + lowerRightPath;
        int threshold = n / 2;
        return (a > threshold || b > threshold || c > threshold);
    }

    public int upperPathScore(TreeNode root, int x) {
        if (root == null) return 0;
        if (root.val == x) return 0;
        else {
            int output = 1;
            output += upperPathScore(root.left, x) + upperPathScore(root.right, x);
            return output;
        }
    }

    public int nodeCounter(TreeNode root) {
        int output = 0;
        if (root == null) return 0;
        else {
            output++;
            output += nodeCounter(root.left) + nodeCounter(root.right);
            return output;
        }
    }

    public TreeNode dfsFind(TreeNode root, int x) {
        if (root == null) {
            return null;
        }
        if (root.val == x) {
            return root;
        }
        TreeNode left = dfsFind(root.left, x);
        if (left != null) {
            return left;
        }
        TreeNode right = dfsFind(root.right, x);
        return right;
    }

    public int longestCommonSubsequence2(String text1, String text2) {
        int output = 0;
        for (int i = 0; i < text1.length(); i++) {
            for (int j = 0; j < text2.length(); j++) {
                if (text2.charAt(j) == text1.charAt(i)) {
                    int currentMax = 0;
                    int counter = i;
                    for (int k = j; k < text2.length(); k++) {
                        if (text1.charAt(counter) == text2.charAt(j)) {
                            currentMax++;

                        }
                    }
                }


            }
        }
        return output;
    }

    public boolean equationsPossible(String[] equations) {
        for (String s : equations) {
            int one = s.charAt(0) - '0';
            char c = s.charAt(1);
            int two = s.charAt(3);
            if (c == '=') {
                if (one != two) return false;
            } else {
                if (one == two) return false;
            }
        }
        return true;
    }

    public boolean checkZeroOnes(String s) {
        int counter0 = 0;
        int counter1 = 0;
        int max0 = 0;
        int max1 = 0;
        for (char c : s.toCharArray()) {
            if (c == '0') {
                counter0++;
                max0 = Math.max(counter0, max0);
                counter1 = 0;
            } else {
                counter1++;
                max1 = Math.max(counter1, max1);
                counter0 = 0;
            }
        }
        return (max1 > max0);
    }

    public int smallestEqual(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            if ((i % 10) == nums[i]) return i;
        }
        return -1;
    }

    public int countKDifference(int[] nums, int k) {
        int counter = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (Math.abs(nums[i] - nums[j]) == k) counter++;
            }
        }
        return counter;
    }

    public int finalValueAfterOperations(String[] operations) {
        int output = 0;
        for (String s : operations) {
            if (s.charAt(1) == '+') output++;
            else output--;
        }
        return output;
    }

    public int maximumPopulation(int[][] logs) {
        HashMap<Integer, Integer> hm = new HashMap<>();
        for (int[] intArray : logs) {
            int yearStart = intArray[0];
            int yearEnd = intArray[1];
            for (int i = yearStart; i < yearEnd; i++) {
                hm.merge(i, 1, Integer::sum);
            }
        }
        Integer maxValue = Collections.max(hm.values());
        Integer output = Integer.MAX_VALUE;
        for (Map.Entry<Integer, Integer> entry : hm.entrySet()) {
            if (entry.getValue() == maxValue) {
                output = Math.min(output, entry.getKey());
            }
        }
        return output;


    }

    public int maxDistance(int[] nums1, int[] nums2) {
        int indexOne = 0;
        int indexTwo = 0;
        int currentMax = 0;
        while (indexOne < nums1.length && indexTwo < nums2.length) {
            int first = nums1[indexOne];
            int second = nums2[indexTwo];
            if (first > second) {
                indexOne++;
                if (indexOne > indexTwo) indexTwo++;
            } else {
                currentMax = Math.max(currentMax, indexTwo - indexOne);
                indexTwo++;
            }
        }
        return currentMax;
    }

    public int maxSumMinProduct(int[] nums) {
        PriorityQueue<Integer> PQ = new PriorityQueue<>(Collections.reverseOrder());
        for (int i : nums) PQ.add(i);
        int max = PQ.poll();
        int sum = max;
        while (!PQ.isEmpty()) {
            int nextNumber = PQ.poll();
            int currentSum = sum + nextNumber;
            int currentMax = currentSum * nextNumber;
            if (currentMax > max) {
                max = currentMax;
                sum = currentSum;
            } else {
                return max;
            }
        }
        return max;
    }
//    Input: box = [["#","#","*",".","*","."],
//                 ["#","#","#","*",".","."],
//                 ["#","#","#",".","#","."]]

    public static char[][] rotateTheBox(char[][] box) {
        for (char[] charArray : box) {
            for (int i = charArray.length - 2; i > -1; i--) {
                if (i < charArray.length - 1) {
                    if (charArray[i] == '#' && charArray[i + 1] == '.') {
                        charArray[i] = '.';
                        charArray[i + 1] = '#';
                        i = i + 2;
                    }
                }
            }
        }
        int n = box.length;
        int m = box[0].length;

        char newbox[][] = new char[m][n];
        // here we are using clock wise rotaion
        // so oth column become n-1th row;
        int l = n - 1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                newbox[j][l] = box[i][j];
            }
            l--;
        }
        return newbox;

    }

    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) return false;
        int currentSum = targetSum - root.val;
        if (currentSum == 0 && root.left == null && root.right == null) return true;
        boolean left = hasPathSum(root.left, currentSum);
        boolean right = hasPathSum(root.right, currentSum);
        return (left || right);
    }

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> output2 = new ArrayList<>();
        if (root == null) return null;
        int currentSum = targetSum - root.val;
        if (currentSum == 0 && root.left == null && root.right == null) {
            List<Integer> output = new ArrayList<>();
            output.add(root.val);
            output2.add(output);
            return output2;
        } else {
            List<List<Integer>> left = pathSum(root.left, currentSum);
            List<List<Integer>> right = pathSum(root.right, currentSum);
            if (left != null) {
                for (List<Integer> L : left) {
                    L.add(0, root.val);
                    output2.add(L);
                }
            }
            if (right != null) {
                for (List<Integer> R : right) {
                    R.add(0, root.val);
                    output2.add(R);
                }
            }
            return output2;
        }

    }

    public void flatten(TreeNode root) {
        if (root == null) return;
        currentTN = new TreeNode();
        //Since the currentTN is always changing, for the final output, we want a copy of the whole tree;
        TreeNode localWholeTree = currentTN;
        //Flatten
        flatten2(root);
        //Reassign the original input
        root.left = null;
        root.right = localWholeTree.right;
    }

    public void flatten2(TreeNode root) {
        if (root == null) return;
        currentTN.val = root.val;
        if (root.left != null) {
            currentTN.right = new TreeNode();
            currentTN.left = null;
            currentTN = currentTN.right;
            flatten2(root.left);
        }
        if (root.right != null) {
            currentTN.right = new TreeNode();
            currentTN.left = null;
            currentTN = currentTN.right;
            flatten(root.right);
        }
    }

    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) return null;
        if (head.next == null) return new TreeNode(head.val);
        //Tortoise and Hare method
        ListNode slow = head;
        ListNode fast = head.next;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        TreeNode tn = new TreeNode(slow.next.val);
        //Copy the right side
        ListNode rightLN = slow.next.next;
        //Delete the right half from head
        slow.next = null;
        //Create the left and right child of the tree;
        tn.left = sortedListToBST(head);
        tn.right = sortedListToBST(rightLN);
        return tn;
    }

    public int listNodeLength(ListNode head) {
        int output = 0;
        while (head != null) {
            output++;
            head = head.next;
        }
        return output;
    }

    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) return head;
        if (head.next == null) return head;
        if (head.val == head.next.val) {
            return deleteDuplicates2(head.next);
        } else {
            head.next = deleteDuplicates(head.next);
            return head;
        }
    }

    public ListNode deleteDuplicates2(ListNode head) {
        if (head == null) return null;
        if (head.next == null) return null;
        if (head.val != head.next.val) {
            return deleteDuplicates(head.next);
        } else {
            return deleteDuplicates2(head.next);
        }
    }

    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if ((p == null && q != null) || (p != null && q == null)) {
            return false;
        }
        boolean one = (p.val == q.val);
        boolean two = (isSameTree(p.left, q.left));
        boolean three = (isSameTree(p.right, q.right));
        return (one && two && three);
    }

    List<Integer> listOutput = new ArrayList<>();

    public List<Integer> inorderTraversal(TreeNode root) {
        if (root == null) return listOutput;
        listOutput.add(root.val);
        inorderTraversal(root.left);
        inorderTraversal(root.right);
        return listOutput;

    }

    public int[] plusOne(int[] digits) {
        int n = digits.length;
        for (int i = n - 1; i >= 0; i++) {
            digits[i]++;
            if (digits[i] < 10) return digits;
            digits[i] = 0;
        }

        int[] output = new int[n + 1];
        output[0] = 1;
        return output;
    }

    public int maxProfit(int[] prices) {
        int profit = 0;
        for (int i = 1; i < prices.length; i++) {
            int currentProfit = prices[i] - prices[i - 1];
            profit = profit + Math.max(currentProfit, 0);
        }
        return profit;
    }

    public String reversePrefix(String word, char ch) {
        int index = word.indexOf(ch);
        if (index == -1) return word;
        String output = reverseString(word.substring(0, index + 1));
        output = output + word.substring(index + 2);
        return output;
    }

    public String reverseString(String word) {
        String output = "";
        for (char c : word.toCharArray()) {
            output = c + output;
        }
        return output;
    }

    public boolean checkAlmostEquivalent(String word1, String word2) {
        int[] frequencyOne = new int[26];
        int[] frequencyTwo = new int[26];
        for (char c : word1.toCharArray()) {
            System.out.println(c);
            int index = c - '0';
            frequencyOne[index]++;
        }
        for (char c : word2.toCharArray()) {
            int index = c - '0';
            frequencyTwo[index]++;
        }
        for (int i = 0; i < 26; i++) {
            int difference = Math.abs(frequencyOne[i] - frequencyTwo[i]);
            if (difference > 3) return false;
        }
        return true;
    }

    public int wateringPlants(int[] plants, int capacity) {
        int output = 1;
        int currentCapacity = capacity;
        for (int i = 0; i < plants.length; i++) {
            int position = i + 1;
            currentCapacity = currentCapacity - plants[i];
            plants[i] = 0;
            if (currentCapacity <= 0) {
                output = output + (2 * position) - 1;
                currentCapacity = capacity;
                i--;
            } else {
                output++;
            }

        }
        return output;
    }

//    Map<Integer, TreeMap<Integer, Integer>> map = new HashMap<>();
//    //Map< Key = Integer, Value = TreeMap>
//    //TreeMap< Key = Index, Value = No. occurances>
//    public RangeFreqQuery(int[] arr) {
//        for (int i = 0; i < arr.length; i++) {
//            Integer currentInteger = arr[i];
//            map.putIfAbsent(currentInteger, new TreeMap<>());
//            TreeMap currentIntegerTreeMap = map.get(currentInteger);
//            Integer treeMapSize = currentIntegerTreeMap.size() + 1;
//            currentIntegerTreeMap.put(i, treeMapSize);
//        }
//    }

    //    public int query(int left, int right, int value) {
//        if (!map.containsKey(value)) return 0;
//        //Find lowest and highest starting index of occurance
//        Map.Entry<Integer, Integer> largest = map.get(value).floorEntry(right);
//        Map.Entry<Integer, Integer> smallest = map.get(value).ceilingEntry(left);
//        if (largest == null || smallest == null) return 0;
//        else {
//            return (largest.getValue() - smallest.getValue());
//        }
//    }
    public boolean asteroidsDestroyed(int mass, int[] asteroids) {
        Arrays.sort(asteroids);
        int currentMass = mass;
        for (int i : asteroids) {
            if (currentMass >= i) {
                if (currentMass + i >= Integer.MAX_VALUE) return true;
                currentMass = currentMass + i;
            } else {
                return false;
            }
        }
        return true;
    }

    public int maxTwoEvents(int[][] events) {
        TreeMap<Integer, Integer> treeMap = new TreeMap<>();
        Arrays.sort(events, (a, b) ->
                a[0] != b[0] ? b[0] - a[0] : a[1] - b[1]);
        Integer max = 0;
        for (int[] intArray : events) {
            max = Math.max(intArray[2], max);
            treeMap.put(intArray[0], max);
        }
        int output = 0;
        for (int[] intArray : events) {
            int currentMax = intArray[2];
            int finishTime = intArray[1] + 1;
            if (treeMap.ceilingKey(finishTime) != null) {
                currentMax = currentMax + treeMap.get(treeMap.ceilingKey(finishTime));
            }

            output = Math.max(output, currentMax);
        }

        return output;

    }

    public int minimumMoves(String s) {
        int output = 0;
        if (s == null || s.length() == 0) {
            return 0;
        } else if (s.length() <= 3) {
            for (char c : s.toCharArray()) {
                if (c != 'O') return 1;
            }
            return 0;
        } else {
            char c = s.charAt(0);
            if (c != 'O') {
                output++;
                output = output + minimumMoves(s.substring(3));
            } else {
                output = output + minimumMoves(s.substring(1));
            }
        }
        return output;
    }

    public boolean allZeros(String s) {
        System.out.println("allZeros input: " + s);
        for (char c : s.toCharArray()) {
            if (c != '0') return false;
        }
        return true;
    }

    public int countWords(String[] words1, String[] words2) {
        int output = 0;
        HashMap<String, Integer> hm = new HashMap<>();
        HashMap<String, Integer> hm2 = new HashMap<>();
        for (String s : words1) {
            hm.merge(s, 1, Integer::sum);
        }
        for (String s : words2) {
            hm2.merge(s, 1, Integer::sum);
        }
        for (String s : hm.keySet()) {
            if (hm2.get(s) != null && hm.get(s) == 1) {
                output++;
            }
        }
        return output;
    }

    public int minimumBuckets(String hamsters) {
        char[] charArray = hamsters.toCharArray();
        for (int i = 0; i < charArray.length; i++) {
            char c = charArray[i];
            if (c == 'H') {
                if (i - 1 >= 0 && charArray[i - 1] == 'B') continue;
                if ((i + 1) < charArray.length && charArray[i + 1] == '.') {
                    charArray[i + 1] = 'B';
                } else if ((i - 1) >= 0 && charArray[i - 1] == '.') {
                    charArray[i - 1] = 'B';
                } else {
                    return -1;
                }
            }
        }
        int output = 0;
        for (char c : charArray) {
            if (c == 'B') output++;
        }
        return output;
    }

    public int triangularSum(int[] nums) {
        if (nums.length == 1) return nums[0];
        int newSize = nums.length - 1;
        int[] output = new int[newSize];
        for (int i = 0; i < newSize; i++) {
            output[i] = (nums[i] + nums[i + 1]) % 10;
        }
        return triangularSum(output);
    }

    public int minSteps(String s, String t) {
        Map<Character, Integer> map = new HashMap<>();
        for (Character c : s.toCharArray()) {
            map.merge(c, 1, Integer::sum);
        }
        for (Character c : t.toCharArray()) {
            if (map.containsKey(c)) {
                map.put(c, map.get(c) - 1);
            } else {
                map.put(c, -1);
            }
        }
        int output = 0;
        for (Integer i : map.values()) {
            output = output + Math.abs(i);
        }
        return output;
    }

    public String largestWordCount(String[] messages, String[] senders) {
        TreeMap<String, Integer> map = new TreeMap<>();
        for (int i = 0; i < messages.length; i++) {
            Integer I = wordCount(messages[i]);
            String sender = senders[i];
            if (map.containsKey(sender)) {
                I = I + map.get(sender);
                map.put(sender, I);
            } else {
                map.put(sender, I);
            }
        }
        int maxValueInMap = (Collections.max(map.values()));
        List<String> topSenders = new ArrayList<>();
        for (String key : map.keySet()) {
            if (map.get(key) == maxValueInMap) {
                topSenders.add(key);
            }
        }
        Collections.sort(topSenders, (a, b) -> {
            return b.compareTo(a);
        });
        return topSenders.get(0);
    }

    public Integer wordCount(String s) {
        Integer i = 0;
        if (s == null || s.equals("")) return i;
        for (char c : s.toCharArray()) {
            if (c == ' ') i++;
        }
        i++;
        return i;
    }

    public int longestPalindrome(String[] words) {
        int output = 0;
        for (int i = 0; i < words.length - 1; i++) {
            String s = words[i];
            if (s.equals("")) continue;
            String sBackwards = "" + s.charAt(1) + s.charAt(0);
            for (int j = i + 1; j < words.length; j++) {
                String ss = words[j];
                if (sBackwards.equals(ss)) {
                    words[i] = "";
                    ss = "";
                    output = output + 4;
                    break;
                }
            }
        }
        for (String s : words) {
            if (!s.equals("")) {
                if (s.charAt(0) == s.charAt(1)) {
                    output = output + 2;
                    break;
                }
            }
        }


        return output;
    }
//
//    public List<Integer> goodDaysToRobBank(int[] security, int time) {
//
//        List<Integer> nonIncreasing = new ArrayList<>();
//        List<Integer> output = new ArrayList<>();
//        if (time == 0) {
//            for (int i = 0; i < security.length; i++) {
//                output.add(i);
//            }
//            return output;
//        }
//        int minimumDays = (time * 2) + 1;
//        if (security.length < minimumDays) return output;
//
//
//        int counter = time - 1;
//        for (int i = 1; i < security.length; i++) {
//            if (security[i - 1] >= security[i]) {
//                if (counter <= 0) {
//                    nonIncreasing.add(i);
//                    System.out.println(i + " added to nonIncreasing");
//                }
//                counter--;
//            } else {
//                counter = time - 1;
//            }
//        }
//        counter = time - 1;
//        for (int i = 0; i < security.length - 1; i++) {
//            if (security[i] <= security[i + 1]) {
//                if (counter <= 0) {
//                    if (nonIncreasing.contains(i)) {
//                        output.add(i);
//                    }
//                }
//                counter--;
//            } else {
//                counter = time - 1;
//            }
//        }
//        return output;
//    }

    public long[] sumOfThree(long num) {
        long[] out = new long[3];
        long[] empty = new long[0];
        if ((num % 3) != 0) return empty;
        long middle = num / 3;
        out[0] = middle - 1;
        out[1] = middle;
        out[2] = middle + 1;
        return out;
    }

    public long smallestNumber(long num) {
        long output = 0;
        Map<Integer, Integer> map = new HashMap<>();
        boolean negative = (num < 0);
        if (negative) num = num * -1;
        while (num != 0) {
            Integer I = Math.toIntExact((num % 10));
            num = num / 10;
            map.merge(I, 1, Integer::sum);
        }

        if (!negative) {
            for (int i = 1; i < 10; i++) {
                if (map.containsKey(i)) {
                    map.put(i, map.get(i) - 1);
                    output = i;
                    break;
                }
            }
            for (int i = 0; i < 10; i++) {
                if (map.get(i) != null) {
                    while (map.get(i) != 0) {
                        output = output * 10;
                        output = output + i;
                        map.put(i, map.get(i) - 1);
                    }
                }
            }
            return output;
        } else {
            for (int i = 9; i > -1; i--) {
                if (map.get(i) != null) {
                    while (map.get(i) != 0) {
                        output = output * 10;
                        output = output + i;
                        map.put(i, map.get(i) - 1);
                    }
                }
            }
            return output * -1;
        }
    }

    public ListNode deleteMiddle(ListNode head) {
        ListNode output = new ListNode();
        output.next = head;
        ListNode slow = output.next;
        ListNode fast = head.next;
        if (fast == null) return output.next;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return output.next;
    }

    public int maximumDetonation(int[][] bombs) {
        List<Bomb> bombList = new ArrayList<>();
        for (int[] bomb : bombs) {
            Bomb b = new Bomb(bomb[0], bomb[1], bomb[2]);
            bombList.add(b);
        }
        //Find how many bombs a single bombs can detonate
        for (Bomb b : bombList) {
            for (Bomb bb : bombList) {
                if (canDetonate(b, bb)) {
                    b.bombsWithinRange.add(bb);
                }
            }
        }
        int output = 0;
        //Sum
        for (Bomb b : bombList) {
            HashSet<Bomb> hs = new HashSet<>();
            hs.add(b);
            for (Bomb bb : b.bombsWithinRange) {
                hs.add(bb);
                for (Bomb bbb : bb.bombsWithinRange) hs.add(bbb);
            }
            output = Math.max(output, hs.size());
        }
        return output;
    }

    public boolean canDetonate(Bomb one, Bomb two) {
        int xDistance = one.x - two.x;
        int yDistance = one.y - two.y;
        double distance = Math.sqrt(Math.pow(xDistance, 2) + Math.pow(yDistance, 2));
        return (one.power >= distance);
    }

    public int[] nodesBetweenCriticalPoints(ListNode head) {
        int[] output = {-1, -1};
        if (head.next == null || head.next.next == null) {
            return output;
        }
        ListNode first = head;
        ListNode second = head.next;
        ListNode third = head.next.next;
        while (first != null && second != null && third != null) {
            int i = first.val;
            int ii = second.val;
            int iii = third.val;
            if (i > ii && ii < iii) {
                second.val = 0;
            } else if (i < ii && ii > iii) {
                second.val = Integer.MAX_VALUE;
            }
            first = first.next;
            second = second.next;
            third = third.next;
        }
        int currentMinimum = Integer.MAX_VALUE;
        int currentMaximum = -1;
        int longestCounter = 0;
        int minimumCounter = 0;
        boolean startedCounting = false;
        while (head != null) {
            int current = head.val;
            if (startedCounting) {
                if (current == 0 || current == Integer.MAX_VALUE) {
                    currentMinimum = Math.min(currentMinimum, minimumCounter);
                    minimumCounter = 0;
                    currentMaximum = Math.max(currentMaximum, longestCounter);
                }
                longestCounter++;
                minimumCounter++;
            } else {
                if (current == 0 || current == Integer.MAX_VALUE) {
                    startedCounting = true;
                    longestCounter++;
                    minimumCounter++;
                }
            }
            head = head.next;
        }

        output[1] = currentMaximum;
        if ((currentMaximum == -1)) {
            output[0] = -1;
        } else {
            output[0] = currentMinimum;
        }
        return output;
    }

    public long subArrayRanges(int[] nums) {
        long output = 0;
        for (int i = 0; i < nums.length; i++) {
            int min = nums[i];
            int max = nums[i];
            for (int j = i + 1; j < nums.length; j++) {
                min = Math.min(min, nums[j]);
                max = Math.max(max, nums[j]);
                output = output + (max - min);
            }
        }
        return output;
    }

    public void deleteNode(ListNode node) {
        if (node.next == null) {
            node = null;
        }
        ListNode current = node;
        ListNode next = current.next;
        while (next != null) {
            current.val = next.val;
            if (next.next == null) {
                current.next = null;
                break;
            }
            current = next;
            next = next.next;
        }
    }

    public ListNode mergeInBetween(ListNode list1, int a, int b, ListNode list2) {
        int counter = 0;
        ListNode firstList = list1;
        while (counter + 1 != a) {
            firstList = firstList.next;
            counter++;
        }
        ListNode firstListRemaining = firstList;
        while (counter != (b + 1)) {
            firstListRemaining = firstListRemaining.next;
            counter++;
        }
        firstList.next = list2;
        ListNode current = firstList.next;
        while (current != null) {
            firstList = firstList.next;
            current = firstList.next;
        }
        firstList.next = firstListRemaining;
        return list1;

    }

    public int[] nextLargerNodes(ListNode head) {
        List<Integer> list = new ArrayList<>();
        list.add(head.val);
        ListNode next = head.next;
        while (next != null) {
            head = head.next;
            list.add(head.val);
            next = head.next;
        }
        int[] output = new int[list.size()];
        output[list.size() - 1] = 0;
        int currentMax = list.get(list.size() - 1);
        for (int i = list.size() - 2; i >= 0; i--) {
            int current = list.get(i);
            if (current > currentMax) {
                currentMax = current;
                output[i] = 0;
            } else if (current < currentMax && current < list.get(i + 1)) {
                currentMax = current;
                output[i] = currentMax;
            } else {
                output[i] = currentMax;
            }
        }
        return output;

    }

    public ListNode oddEvenList(ListNode head) {
        ListNode output = new ListNode();
        ListNode outputIndex = output;
        ListNode first = head;
        ListNode second = first.next;
        while (first != null) {
            if (first.next == null) {
                first.next = output.next;
                return head;
            }
            second = first.next;
            outputIndex.next = new ListNode();
            outputIndex = outputIndex.next;
            outputIndex.val = second.val;
            if (second.next == null) {
                first.next = output.next;
                return head;
            }
            first.next = second.next;
            first = first.next;
        }

        return head;
    }

    public ListNode removeNodes(ListNode head) {
        if (head == null) return null;
        ListNode current = head;
        ListNode next = current.next;
        if (next == null) return head;
        if (current.val < next.val) {
            return (removeNodes(next));
        } else {
            current.next = (removeNodes(current.next));
            if (current.next.val != next.val) {
                return removeNodes(head);
            } else {
                return head;
            }

        }
    }

    public boolean checkStraightLine(int[][] coordinates) {
        int[] first = coordinates[0];
        int[] second = coordinates[1];
        double dx = second[0] - first[0];
        double dy = second[1] - first[1];
        for (int i = 0; i < coordinates.length - 1; i++) {
            first = coordinates[i];
            second = coordinates[i + 1];
            double dxcurrent = second[0] - first[0];
            double dycurrent = second[1] - first[1];
            if ((dycurrent * dx) != (dxcurrent * dy)) {
                return false;
            }
        }
        return true;
    }

    public boolean isSubPath(ListNode head, TreeNode root) {
        if (root == null) return false;
        if (isSubPath2(head, root)) return true;
        else {
            return (isSubPath(head, root.left) || isSubPath(head, root.right));
        }
    }

    public boolean isSubPath2(ListNode head, TreeNode root) {
        if (head == null) return true;
        if (root == null) return false;
        else {
            if (head.val != root.val) return false;
            else {
                return (isSubPath2(head.next, root.left) || isSubPath2(head.next, root.right));
            }
        }
    }

    public int numComponents(ListNode head, int[] nums) {
        int output = 0;
        List<Integer> list = Arrays.stream(nums).boxed().toList();
        ListNode current = head;
        ListNode next = current.next;
        if (next == null) {
            return output;
        }
        while (next != null) {
            boolean one = list.contains(current.val);
            boolean two = list.contains(next.val);
            if (one && two) {
                output++;
            }
            next = next.next;
            current = current.next;
        }
        return output;

    }

    public ListNode[] splitListToParts(ListNode head, int k) {
        ListNode[] output = new ListNode[k];
        int length = 0;
        for (ListNode ln = head; ln != null; ln = ln.next) {
            length++;
        }
        int sectionSize = length / k;
        int remainder = length % k;
        for (int i = 0; i < k; i++) {
            ListNode current = new ListNode();
            ListNode currentIndex = current;
            int currentSize = sectionSize;
            if (remainder > 0) {
                currentSize++;
                remainder--;
            }
            while (currentSize != 0) {
                currentIndex.next = new ListNode(head.val);
                currentIndex = currentIndex.next;
                head = head.next;
                currentSize--;
            }
            output[i] = current.next;

        }
        return output;


    }

    public Node flatten(Node head) {
        Node current = head;
        while (current != null) {
            if (current.child != null && current.next == null) {
                current.child.prev = current;
                current.next = current.child;
                current.child = null;
            } else if (current.child != null && current.next != null) {
                Node next = current.next;
                current.child.prev = current;
                current.next = current.child;
                current.child = null;
                Node tmp = current.next;
                while (tmp.next != null) {
                    tmp = tmp.next;
                }
                tmp.next = next;
                next.prev = tmp;
            }
            current = current.next;
        }
        return head;


    }


    public ListNode reverseBetween(ListNode head, int left, int right) {
        if (left == right) return head;
        if (left == 1) {
            if (right == 1) return head;
            ListNode current = head;
            ListNode next = current.next;
            while (right != 1) {
                current = next;
                next = current.next;
                right -= 1;
            }
            current.next = null;
            ListNode output = reverseListNode(head);
            ListNode current2 = output;
            ListNode next2 = current2.next;
            while (next2 != null) {
                current2 = next2;
                next2 = current2.next;
            }
            current2.next = next;
            return head;
        } else {
            ListNode current = head;
            ListNode next = current.next;
            while (left != 2) {
                current = next;
                next = current.next;
                left -= 1;
                right -= 1;
            }
            current.next = null;
            //ListNode next is the head of the list to be reversed
            ListNode currentt = next;
            ListNode nextt = currentt.next;
            right -= 1;
            while (right != 1) {
                currentt = nextt;
                nextt = currentt.next;
                right -= 1;
            }
            currentt.next = null;
            ListNode output = reverseListNode(next);
            current.next = output;
            ListNode current2 = output;
            ListNode next2 = current2.next;
            while (next2 != null) {
                current2 = next2;
                next2 = current2.next;
            }
            current2.next = nextt;
            return head;
        }
    }

    public ListNode reverseListNode(ListNode head) {
        if (head == null) return head;
        ListNode current = head;
        ListNode next = head.next;
        ListNode prev = null;
        while (current != null) {
            next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        return prev;

    }


    public int countPrimeSetBits(int left, int right) {
        int counter = 0;
        for (int i = left; i < right + 1; i++) {
            String s = Integer.toBinaryString(i);
            if (isPrime(countSetBits(s))) {
                counter++;
                System.out.println(i);
            }
        }
        return counter;
    }

    public int countSetBits(String s) {
        if (s == null || s.equals("")) return 0;
        char c = s.charAt(0);
        String next = s.substring(1);
        if (c == '1') {
            return (1 + countSetBits(next));
        } else {
            return (countSetBits(next));
        }
    }

    public boolean isPrime(int in) {
        if (in == 1) return false;
        for (int i = 2; i < in; i++) {
            if (in % i == 0) return false;
        }
        return true;
    }


    public boolean isValidBST(TreeNode root) {
        if (root == null) return true;
        boolean left = validateBST(root.left, Long.MIN_VALUE, root.val);
        boolean right = validateBST(root.right, root.val, Long.MAX_VALUE);
        return (left && right);
    }

    public boolean validateBST(TreeNode root, long min, long max) {
        if (root == null) return true;
        if (root.val <= min || root.val >= max) return false;
        boolean left = validateBST(root.left, min, root.val);
        boolean right = validateBST(root.right, root.val, max);
        return (left && right);
    }

//    Queue<Integer> q;
//    public BSTIterator(TreeNode root) {
//        q = new LinkedList<>();
//        if (root != null) {
//            addToList(root.left);
//            q.add(root.val);
//            addToList(root.right);
//        }
//    }
//    public void addToList(TreeNode root) {
//        if (root != null) {
//            addToList(root.left);
//            q.add(root.val);
//            addToList(root.right);
//        }
//    }
//
//    public int next() {
//        return (q.poll());
//    }
//
//    public boolean hasNext() {
//        return (!q.isEmpty());
//    }


    public List<List<Integer>> levelOrder(TreeNode root) {
        List<TreeNode> tnList = new ArrayList<>();
        tnList.add(root);
        return levelOrder2(tnList);
    }

    public List<List<Integer>> levelOrder2(List<TreeNode> tnList) {
        List<List<Integer>> output = new ArrayList<>();
        if (tnList.size() == 0) return output;
        List<Integer> currentLevel = new ArrayList<>();
        List<TreeNode> nextLevel = new ArrayList<>();
        for (TreeNode tn : tnList) {
            if (tn != null) {
                currentLevel.add(tn.val);
                if (tn.left != null) nextLevel.add(tn.left);
                if (tn.right != null) nextLevel.add(tn.right);
            }

        }
        output = levelOrder2(nextLevel);
        output.add(0, currentLevel);
        return output;
    }


    public TreeNode insertIntoBST(TreeNode root, int val) {
        if (root == null) {
            return (new TreeNode(val));
        } else {
            int difference = (val - root.val);
            if (difference == -1) {
                TreeNode left = root.left;
                root.left = new TreeNode(val);
                root.left.left = left;
                return root;
            } else if (difference == 1) {
                TreeNode right = root.right;
                root.right = new TreeNode(val);
                root.right.right = right;
                return root;
            } else {
                if (root.val < val) {
                    root.right = insertIntoBST(root.right, val);
                } else {
                    root.left = insertIntoBST(root.left, val);
                }
                return root;
            }
        }
    }


    public boolean isCompleteTree(TreeNode root) {
        List<TreeNode> tnList = new ArrayList<>();
        tnList.add(root);
        return isCompleteTree2(tnList);
    }

    public boolean isCompleteTree2(List<TreeNode> tnList) {
        List<TreeNode> tnNext = new ArrayList<>();
        if (listAllNullElements(tnList)) return true;
        int max = tnList.size() - 1;
        for (int i = 0; i < tnList.size(); i++) {
            TreeNode current = tnList.get(i);
            if (current == null) return (i == max);
            tnNext.add(current.left);
            tnNext.add(current.right);
        }
        return isCompleteTree2(tnNext);
    }

    public boolean listAllNullElements(List<TreeNode> in) {
        if (in == null) return true;
        for (TreeNode tn : in) {
            if (tn != null) return false;
        }
        return true;
    }


    Map<Integer, Integer> map;

    public TreeNode bstToGst(TreeNode root) {
        if (root == null) return root;
        map = new HashMap<>();
        bstToMap(root);
        List<Integer> keySet = new ArrayList<>(map.keySet());
        Collections.sort(keySet, new Comparator<Integer>() {
            public int compare(Integer a, Integer b) {
                return (b - a);
            }
        });
        Integer currentMax = 0;
        for (Integer I : keySet) {
            currentMax = currentMax + I;
            map.put(I, currentMax);
        }
        return (reassignBST(root));
    }

    public TreeNode reassignBST(TreeNode root) {
        if (root == null) return root;
        root.val = map.get(root.val);
        root.left = reassignBST(root.left);
        root.right = reassignBST(root.right);
        return root;
    }

    public void bstToMap(TreeNode root) {
        if (root != null) {
            map.put(root.val, root.val);
            bstToMap(root.left);
            bstToMap(root.right);
        }
    }

    List<List<Integer>> treeRepresentationList;
    int currentNode = 1;
    int currentLayer = 0;

    public List<Integer> pathInZigZagTree(int label) {
        List<Integer> output = new ArrayList<>();
        treeRepresentationList = new ArrayList<>();
        if (label == 1) {
            output.add(label);
            return output;
        }
        createTreeList(label);
        int index = treeRepresentationList.get(0).indexOf(label);

        for (List<Integer> list : treeRepresentationList) {
            output.add(0, list.get(index));
            index = index / 2;
        }
        return output;
    }

    public void createTreeList(int in) {
        while (currentNode <= in) {
            List<Integer> output = new ArrayList<>();
            boolean reverse = (currentLayer % 2 != 0);
            int limit = (int) Math.pow(2, currentLayer);
            for (int i = 0; i < limit; i++) {
                output.add(currentNode);
                currentNode++;
            }
            if (reverse) {
                Collections.sort(output, Collections.reverseOrder());
            }
            currentLayer++;
            treeRepresentationList.add(0, output);
        }
    }

    public int goodNodes(TreeNode root) {
        if (root == null) return 0;
        return goodNodes2(root, Integer.MIN_VALUE);
    }

    public int goodNodes2(TreeNode root, int max) {
        int output = 0;
        if (root == null) return output;
        if (max <= root.val) {
            output++;
            max = root.val;
        }
        return (output + goodNodes2(root.left, max) + goodNodes2(root.right, max));
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return root;
        if (root.val == p.val || root.val == q.val) return root;
        int pp = (p.val < q.val ? p.val : q.val);
        int qq = (p.val < q.val ? q.val : p.val);
        boolean leftContains = treeContains(root.left, pp);
        boolean rightContains = treeContains(root.right, qq);
        if (leftContains && rightContains) return root;
        if (leftContains && !rightContains) return lowestCommonAncestor(root.left, p, q);
        if (!leftContains && rightContains) return lowestCommonAncestor(root.right, p, q);
        return root;

    }

    public boolean treeContains(TreeNode root, int i) {
        if (root == null) return false;
        if (root.val == i) return true;
        return (treeContains(root.left, i) || treeContains(root.right, i));
    }

    public int minTime(int n, int[][] edges, List<Boolean> hasApple) {
        boolean[] boolArray = new boolean[hasApple.size()];
        for (int i = 0; i < hasApple.size(); i++) {
            boolArray[i] = hasApple.get(i);
        }
        List<Integer> appleList = new ArrayList<>();
        for (int i = hasApple.size() - 1; i >= 0; i--) {
            int[] edge = edges[i];
            int first = edge[0];
            int second = edge[1];
            if (appleList.contains(second)) {
                boolArray[i] = true;
                appleList.add(first);
            }
            boolean apple = boolArray[i];
            if (apple) {
                appleList.add(first);
            }
        }
        int output = 0;
        for (boolean b : boolArray) {
            if (b) output = output + 2;
        }
        return output;
    }

    public TreeNode insertIntoMaxTree(TreeNode root, int val) {
        if (root == null) {
            TreeNode output = new TreeNode(val);
            return output;
        }
        double difference;
        System.out.println(root.val);
        System.out.println(val);
        if (root.val < val) {
            TreeNode output = new TreeNode(val);
//            difference = val / 2;
//            if (root.val <= difference) {
//                output.left = root;
//            } else {
//                output.right = root;
//            }
            output.left = root;
            return output;
        }
        difference = (root.val / 2);
        if (root.val <= difference) {
            root.left = insertIntoMaxTree(root.left, val);
        } else {
            root.right = insertIntoMaxTree(root.right, val);
        }
        return root;
    }


    public int minIncrements(int n, int[] cost) {
        int output = 0;
        if (cost.length == 0) return output;
        for (int i = 1; i < cost.length; i = i + 2) {
            int first = cost[i];
            int second = cost[i + 1];
            int difference = Math.abs(second - first);
            output += difference;
        }
        return output;
    }

    public Map<Integer, List<Integer>> edgesToMap(int[][] edges) {
        Map<Integer, List<Integer>> nodeChildMap = new HashMap<>();
        Queue<Integer> nodeQueue = new PriorityQueue<>();
        nodeQueue.add(0);
        while (!nodeQueue.isEmpty()) {
            List<Integer> currentChild = new ArrayList<>();
            Integer I = nodeQueue.poll();
            for (int[] edge : edges) {
                int a = edge[0];
                int b = edge[1];
                if (a == I || b == I) {
                    int child = (a == I) ? b : a;
                    if (!nodeChildMap.keySet().contains(child)) {
                        nodeQueue.add(child);
                        currentChild.add(child);
                    }
                }
            }
            nodeChildMap.put(I, currentChild);
        }
        return nodeChildMap;
    }

    class LockingTree {

        int[] tree;
        boolean[] lockStatus;
        int[] lockUser;

        public LockingTree(int[] parent) {
            int n = parent.length;
            tree = parent;
            lockStatus = new boolean[n];
            lockUser = new int[n];
        }

        public boolean lock(int num, int user) {
            if (lockStatus[num]) return false;
            lockStatus[num] = true;
            lockUser[num] = user;
            return true;
        }

        public boolean unlock(int num, int user) {
            if (!lockStatus[num]) return false;
            if (lockUser[num] != user) return false;
            lockStatus[num] = false;
            lockUser[num] = 0;
            return true;
        }

        public boolean upgrade(int num, int user) {
//            The node is unlocked,
//            It has at least one locked descendant (by any user), and
//            It does not have any locked ancestors.
            if (lockStatus[num]) return false;
            List<Integer> allNodes = new ArrayList<>();
            allNodes.add(num);
            for (int i = 0; i < tree.length; i++) {
                if (allNodes.contains(tree[i])) {
                    allNodes.add(i);
                }
            }
            boolean lockedDescendant = false;
            for (Integer I : allNodes) {
                if (lockStatus[I]) {
                    lockedDescendant = true;
                    break;
                }
            }
            if (!lockedDescendant) return false;
            int parent = tree[num];
            while (parent != -1) {
                if (lockStatus[parent]) return false;
                parent = tree[parent];
            }
            lockStatus[num] = true;
            lockUser[num] = user;
            allNodes.remove(num);
            for (Integer I : allNodes) {
                lockStatus[I] = false;
                lockUser[I] = 0;
            }
            return true;
        }
    }

    public int removeDuplicates(int[] nums) {
        if (nums.length <= 2) {
            return nums.length;
        }
        int i = 0;
        int j = 1;
        for (int k = 2; k < nums.length; k++) {
            if (nums[i] == nums[j] && nums[j] == nums[k]) continue;
            else {
                i++;
                j++;
                nums[j] = nums[k];
            }
        }
        return j + 1;
    }

    public void rotate(int[] nums, int k) {
        int n = nums.length;
        if (k == n) return;
        if (k > n) k = k % n;
        int[] newArray = new int[n];
        int index = n - k;
        for (int i = 0; i < n; i++, index++) {
            if (index == n) index = 0;
            newArray[i] = nums[index];
        }
        for (int i = 0; i < n; i++) {
            nums[i] = newArray[i];
        }
    }

    public boolean canJump(int[] nums) {
        if (nums.length == 1) {
            return true;
        }
        //Must equal or larger than limit
        int limit = nums.length - 1;
        boolean[] jump = new boolean[nums.length];
        //Find positions which can reach within one
        for (int i = 0; i < nums.length; i++) {
            if (i + nums[i] >= limit) jump[i] = true;
        }
        for (int i = nums.length - 1; i >= 0; i--) {
            if (jump[i] || nums[i] == 0) continue;
            for (int j = 1; j < nums[i] + 1; j++) {
                if (jump[i + j]) {
                    jump[i] = true;
                    break;
                }
            }
        }
        return jump[0];
    }

    public int hIndex(int[] citations) {
        Arrays.sort(citations);
        int output = 0;
        for (int i = citations.length - 1; i >= 0; i--) {
            System.out.println(citations[i]);
            if (citations[i] >= output) {
                output++;
            } else break;

        }
        return output;
    }

    public int canCompleteCircuit(int[] gas, int[] cost) {
        for (int i = 0; i < gas.length; i++) {
            gas[i] = gas[i] - cost[i];
        }
        if (arraySum(gas) < 0) return -1;
        for (int i = 0; i < gas.length; i++) {
            int currentGas = gas[i];
            System.out.println("i is " + i + "currentgas is " + currentGas);
            if (currentGas < 0) continue;
            int index = i;
            for (int j = 0; j < gas.length; j++) {
                index++;
                if (index == gas.length) index = 0;
                if (index == i) return i;
                currentGas += gas[index];
                System.out.println("currentGas is " + currentGas);
                if (currentGas < 0) {
                    i += j;
                }
            }
        }

        return -1;
    }

    public int arraySum(int[] in) {
        int sum = 0;
        for (int i : in) {
            sum += i;
        }
        return sum;
    }

    public String reverseWords(String s) {
        String output = "";
        String[] sArray = s.split("[\s]+");
        for (String ss : sArray) {
            if (ss != "") {
                ss = ss.replaceAll("[\s]+", "");
                output = " " + ss + output;
            }
        }
        return output.substring(1);
    }

    int[] sortedArray;

    public int[] twoSum(int[] numbers, int target) {
        sortedArray = numbers;
        for (int i = numbers.length - 1; i > 0; i++) {
            int left = target - sortedArray[i];
            left = sortedArrayContains(left);
            if (left != -1) {
                int[] output = {left + 1, i + 1};
                return output;
            }

        }
        return null;
    }

    public int sortedArrayContains(int target) {
        for (int i = 0; i < sortedArray.length; i++) {
            if (sortedArray[i] == target) return i;
            if (sortedArray[i] > target) return -1;
        }
        return -1;
    }

    public int minimumTotal(List<List<Integer>> triangle) {
        int n = triangle.size();
        if (n == 1) {
            return (triangle.get(0).get(0));
        }
        List<Integer> firstList = triangle.get(triangle.size() - 2);
        List<Integer> secondList = triangle.get(triangle.size() - 1);
        List<Integer> newList = new ArrayList<>();
        triangle.remove(triangle.size() - 1);
        triangle.remove(triangle.size() - 1);
        for (int i = 0; i < firstList.size(); i++) {
            Integer I = firstList.get(i) + Math.max(secondList.get(i), secondList.get(i + 1));
            newList.add(i, I);
        }

        triangle.add(triangle.size(), newList);
        return minimumTotal(triangle);
    }

    int[][] gameGrid;
    int xBound;
    int yBound;

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        //int[x][y]
        gameGrid = obstacleGrid;
        xBound = gameGrid[0].length;
        yBound = gameGrid.length;
        return uniquePath(0, 0);
    }

    public int uniquePath(int x, int y) {
        if (x >= xBound || y >= yBound) return 0;
        if (gameGrid[y][x] == 1) return 0;
        if (x == xBound - 1 && y == yBound - 1) {
            System.out.println(x);
            System.out.println(y);
            return 1;
        }

        int down = uniquePath(y + 1, x);
        int right = uniquePath(y, x + 1);
        return (down + right);
    }

    String currentPalindrome = "";
    String pString;

    public String longestPalindrome(String s) {
        int n = s.length();
        pString = s;
        if (n == 1) return s;
        if (n == 2) {
            if (s.charAt(0) == s.charAt(1)) return s;
            else return s.substring(1);
        }
        for (int i = 0; i < s.length() - 2; i++) {
            stringSweep(i, i + 1);
            stringSweep(i, i + 2);
            stringSweep(i + 1, i + 2);
        }
        if (currentPalindrome.equals("")) return s.substring(s.length() - 1);
        return currentPalindrome;
    }

    public void stringSweep(int start, int end) {
        while (start >= 0 && end <= pString.length() - 1) {
            if (pString.charAt(start) == pString.charAt(end)) {
                String s = pString.substring(start, end + 1);
                if (s.length() >= currentPalindrome.length()) {
                    currentPalindrome = s;
                }
            } else {
                break;
            }
            start--;
            end++;
        }
    }

    public boolean isInterleave(String s1, String s2, String s3) {
        if (s1.length() == 0 && s2.length() == 0 && s3.length() == 0) return true;
        if (s3.length() == 0) return false;
        if (s1.length() == 0) return (s2.equals(s3));
        if (s2.length() == 0) return (s1.equals(s3));
        char c = s1.charAt(0);
        char cc = s2.charAt(0);
        char ccc = s3.charAt(0);
        if (c == c && cc == ccc) {
            boolean one = isInterleave(s1.substring(1), s2, s3.substring(1));
            boolean two = isInterleave(s1, s2.substring(1), s3.substring(1));
            return (one || two);
        } else if (c == ccc) {
            return isInterleave(s1.substring(1), s2, s3.substring(1));
        } else if (cc == ccc) {
            return isInterleave(s1, s2.substring(1), s3.substring(1));
        } else {
            return false;
        }
    }

    public int findKthLargest(int[] nums, int k) {
        int currentKth = 1;
        int currentMax = Integer.MIN_VALUE;
        for (int i = 0; i < k; i++) {
            currentMax = Integer.MIN_VALUE;
            for (int j : nums) currentMax = Math.max(j, currentMax);
            if (currentKth == k) return currentMax;
            for (int j = 0; j < nums.length; j++) {
                if (nums[j] == currentMax) {
                    nums[j] = Integer.MIN_VALUE;
                    break;
                }
            }
            currentKth++;
        }
        return currentMax;
    }

    Map<Integer, Integer> nextMap;
    Map<Integer, Integer> randomMap;
    Map<Integer, Integer> valueMap;
    Integer headValue;

    public Node copyRandomList(Node head) {
        headValue = head.val;
        nextMap = new HashMap<>();
        randomMap = new HashMap<>();
        valueMap = new HashMap<>();
        int counter = 1;
        while (head != null) {
            Integer value = head.val;
            if (head.next == null) {
                nextMap.put(value, null);
            } else {
                nextMap.put(value, head.next.val);
            }
            if (head.random == null) {
                randomMap.put(value, null);
            } else {
                randomMap.put(value, head.random.val);
            }
            head = head.next;
            counter++;
        }
        return constructNode();
    }

    public Node constructNode() {
        Integer currentHeadValue = headValue;
        Map<Integer, Node> nodeMap = new HashMap<>();
        while (currentHeadValue != null) {
            nodeMap.put(currentHeadValue, new Node(currentHeadValue));
            currentHeadValue = nextMap.get(currentHeadValue);
        }
        Integer currentNextNode = headValue;
        while (currentNextNode != null) {
            Integer nextNode = nextMap.get(currentNextNode);
            nodeMap.get(currentNextNode).next = nodeMap.get(nextNode);
            currentNextNode = nextNode;
        }
        for (Integer I : randomMap.keySet()) {
            Integer randomTo = randomMap.get(I);
            nodeMap.get(I).random = nodeMap.get(randomTo);
        }
        return nodeMap.get(headValue);
    }

    public int rob(int[] nums) {
        int rob = 0;
        int skip = 0;
        for (int i = 0; i < nums.length; i++) {
            //If rob the current house, for sure I cannot rob the previous house
            //The max value thus far by choosing to rob is the current house + max of not robbing the prior house
            int currentRob = nums[i] + skip;
            //If I skip this house, have the option for the previous house. Therefore the max
            //benifit will be the max of rob and skip of the prior house;
            int currentSkip = Math.max(rob, skip);
            //Update rob and skip
            rob = currentRob;
            skip = currentSkip;
        }
        return Math.max(rob, skip);
    }

    public int minSubArrayLen(int target, int[] nums) {
        if (nums.length == 1) {
            if (nums[0] >= target) return 1;
            return 0;
        }
        int output = Integer.MAX_VALUE;
        boolean flag = false;
        int left = 0;
        int right = 0;
        int currentSum = nums[right];
        while (right < nums.length) {
            System.out.println("left is " + left);
            System.out.println("right is " + right);
            System.out.println(currentSum);
            if (currentSum < target) {
                right++;
                if (right > nums.length) break;
                currentSum += nums[right];
            } else if (currentSum >= target) {
                flag = true;
                output = Math.min(output, (right - left + 1));
                currentSum -= nums[left];
                left++;
            }
        }
        if (flag) return output;
        else return 0;
    }

//    List<Double> output;

//    public List<Double> averageOfLevels(TreeNode root) {
//        output = new ArrayList<>();
//        List<TreeNode> tn = new ArrayList<>();
//        tn.add(root);
//        List<Double> output = averageOfLevels2(tn);
//        return output;
//    }

//    public List<Double> averageOfLevels2(List<TreeNode> tn) {
//        if (tn.size() == 0) return output;
//        List<TreeNode> tnNext = new ArrayList<>();
//        int sum = 0;
//        int divisor = 0;
//        for (TreeNode tnn : tn) {
//            if (tnn != null) {
//                sum += tnn.val;
//                divisor++;
//                if (tnn.left != null) tnNext.add(tnn.left);
//                if (tnn.right != null) tnNext.add(tnn.right);
//            }
//        }
//
//        double currentAverage = (double) sum / (double) divisor;
//        output.add(currentAverage);
//        return (averageOfLevels2(tnNext));
//
//    }

    public ListNode rotateRight(ListNode head, int k) {
        if (k == 0) return head;
        if (head.next == null) return head;
        ListNode output = head;
        ListNode next = head.next;
        while (next != null) {
            System.out.println(output.val);
            System.out.println(next.val);
            output = head.next;
            next = output.next;
            if (next.next == null) break;
        }
        output.next = null;
        next.next = head;
        return (rotateRight(next, k--));
    }

    public boolean isSymmetric(TreeNode root) {
        List<TreeNode> tnList = new ArrayList<>();
        if (root == null) return true;
        tnList.add(root.left);
        tnList.add(root.right);
        return isSymmetric2(tnList);
    }

    public boolean isSymmetric2(List<TreeNode> tnList) {
        if (tnList == null) return true;
        if (tnList.size() == 0) return true;
        if (tnList.size() % 2 != 0) return false;
        List<TreeNode> tnListNext = new ArrayList<>();
        List<Integer> currentLayerValues = new ArrayList<>();
        for (TreeNode tn : tnList) {
            if (tn != null) {
                System.out.println(tn);
                currentLayerValues.add(tn.val);
                tnListNext.add(tn.left);
                tnListNext.add(tn.right);
            }
        }
        if (!isSymmetricList(currentLayerValues)) return false;
        System.out.println("next");
        return isSymmetric2(tnListNext);
    }

    public boolean isSymmetricList(List<Integer> in) {
        if (in == null) return true;
        if (in.size() == 1) return false;
        int n = in.size();
        int endIndex = n - 1;
        for (int i = 0; i < n / 2; i++, endIndex--) {
            if (!in.get(i).equals(in.get(endIndex))) return false;
        }
        return true;
    }


    public int sumNumbers(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null && root.right == null) {
            return root.val;
        }
        return sumNumbers2(root.left, root.val) + sumNumbers2(root.right, root.val);
    }

    public int sumNumbers2(TreeNode root, int i) {
        if (root == null) return 0;
        i = i * 10;
        i = i + root.val;
        if (root.left == null && root.right == null) {
            return (i);
        }
        System.out.println(i);
        return sumNumbers2(root.left, root.val) + sumNumbers2(root.right, root.val);
    }

    public int findPeakElement(int[] nums) {
        if (nums.length == 1) return 0;
        int currentPeak = Integer.MIN_VALUE;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] >= currentPeak) {
                currentPeak = nums[i];
            } else if (nums[i] < currentPeak) {
                return (i - 1);
            }
        }
        return 0;
    }

    public int[] searchRange(int[] nums, int target) {
        int[] output = {-1, -1};
        int[] empty = {-1, -1};
        if (nums == null || nums.length == 0) return output;
        boolean found = false;
        for (int i = 0; i < nums.length; i++) {
            if (found) {
                if (nums[i] == target) {
                    output[1] = i;
                }
            } else {
                if (nums[i] == target) {
                    found = true;
                    output[0] = i;
                }
            }
        }
        return output;
    }

    List<Integer> out;

    public List<Integer> rightSideView(TreeNode root) {
        out = new ArrayList<>();
        if (root == null) return out;
        List<TreeNode> tnList = new ArrayList<>();
        tnList.add(root);
        return rightSideView2(tnList);
    }

    public List<Integer> rightSideView2(List<TreeNode> tnList) {
        if (tnList == null || tnList.size() == 0) {
            return out;
        } else {
            out.add(tnList.get(tnList.size() - 1).val);
            List<TreeNode> tnListNext = new ArrayList<>();
            for (TreeNode tn : tnList) {
                if (tn.left != null) tnListNext.add(tn.left);
                if (tn.right != null) tnListNext.add(tn.right);
            }
            return rightSideView2(tnListNext);
        }
    }

    public int coinChange(int[] coins, int amount) {
        Arrays.sort(coins);
        int output = 0;
        int n = coins.length - 1;
        while (n > -1) {
            if (amount == 0) return output;
            int currentCoin = coins[n];
            if (amount >= currentCoin) {
                amount -= currentCoin;
                output++;
            } else {
                n--;
            }
        }
        if (amount != 0) return -1;
        else return output;
    }

    public void setZeroes(int[][] matrix) {
        int xLength = matrix[0].length;
        int yLength = matrix.length;
        List<Integer> yList = new ArrayList<>();
        List<Integer> xList = new ArrayList<>();
        for (int x = 0; x < xLength; x++) {
            for (int y = 0; y < matrix.length; y++) {
                if (matrix[y][x] == 0) {
                    yList.add(y);
                    xList.add(x);
                }
            }
        }
        for (int i : yList) {
            int[] yArray = new int[xLength];
            matrix[i] = yArray;
        }
        for (int i : xList) {
            for (int[] arr : matrix) {
                arr[i] = 0;
            }
        }
    }

    public Node connect(Node root) {
        if (root == null) return root;
        List<Node> tnList = new ArrayList<>();
        List<Node> tnListNext = new ArrayList<>();
        tnList.add(root);
        while (tnList.size() != 0) {
            Node current = tnList.get(0);
            if (current.left != null) tnListNext.add(current.left);
            if (current.right != null) tnListNext.add(current.right);
            if (tnList.size() == 1) {
                current.next = null;
                tnList = tnListNext;
                tnListNext = new ArrayList<>();
            } else {
                current.next = tnList.get(1);
                tnList.remove(0);
            }
        }
        return root;
    }

    List<String> wordDictonary;

    public boolean wordBreak(String s, List<String> wordDict) {
        wordDictonary = wordDict;
        return wordBreak2(s);
    }

    public boolean wordBreak2(String s) {
        int n = s.length();
        for (int i = 1; i < s.length() + 1; i++) {
            String subString = s.substring(0, i);
            if (wordDictonary.contains(subString)) {
                if (i == n) return true;
                else {
                    String secondHalf = s.substring(i);
                    System.out.println(subString);
                    System.out.println(secondHalf);
                    return wordBreak2(secondHalf);
                }
            }
        }
        return false;
    }

    public TreeNode pruneTree(TreeNode root) {
        if (root == null) return null;
        if (root.val == 0) {
            TreeNode newLeft = pruneTree(root.left);
            TreeNode newRight = pruneTree(root.right);
            if (newLeft == null && newRight == null) return null;
            else {
                root.left = newLeft;
                root.right = newRight;
                return root;
            }
        } else if (root.val == 1) {
            TreeNode newLeft = pruneTree(root.left);
            TreeNode newRight = pruneTree(root.right);
            root.left = newLeft;
            root.right = newRight;
            return root;
        }
        return root;
    }

    TreeMap<Integer, Integer> treeMapKey;
    TreeNode tnRoot;

    public TreeNode convertBST(TreeNode root) {
        if (root == null) return root;
        List<TreeNode> tnList = new ArrayList<>();
        tnList.add(root);
        treeMapKey = createTM(tnList);
        Set<Integer> keySet = treeMapKey.keySet();
        List<Integer> listSet = new ArrayList<>(keySet);
        Collections.sort(listSet, Collections.reverseOrder());
        Integer currentMax = 0;
        for (Integer I : listSet) {
            currentMax += I;
            treeMapKey.put(I, currentMax);
        }
        updateTreeByMap(root);
        return root;
    }

    public void updateTreeByMap(TreeNode tn) {
        if (tn == null) return;
        int key = tn.val;
        tn.val = treeMapKey.get(key);
        updateTreeByMap(tn.left);
        updateTreeByMap(tn.right);
    }

    public TreeMap<Integer, Integer> createTM(List<TreeNode> tnList) {
        TreeMap<Integer, Integer> out = new TreeMap<>();
        while (tnList.size() != 0) {
            TreeNode current = tnList.get(0);
            out.put(current.val, 0);
            if (current.left != null) tnList.add(current.left);
            if (current.right != null) tnList.add(current.right);
            tnList.remove(0);
        }
        return out;
    }


    public int removeCoveredIntervals(int[][] intervals) {
        int output = intervals.length;
        for (int i = 0; i < intervals.length; i++) {
            int[] interval = intervals[i];
            int start = interval[0];
            int end = interval[1];
            for (int j = 0; j < intervals.length; j++) {
                if (j == i) continue;
                else {
                    int[] currentInterval = intervals[j];
                    int currentStart = currentInterval[0];
                    int currentEnd = currentInterval[1];
                    if (currentStart <= start && end <= currentEnd) {
                        output--;
                        break;
                    }
                }

            }
        }
        return output;
    }

    public TreeNode constructMaximumBinaryTree(int[] nums) {
        if (nums.length == 0) return null;
        int max = Integer.MIN_VALUE;
        int maxIndex = 0;
        for (int i = 0; i < nums.length; i++) {
            int current = nums[i];
            max = Math.max(current, max);
            if (max == current) maxIndex = i;
        }
        TreeNode out = new TreeNode(max);
        int[] left = Arrays.copyOfRange(nums, 0, maxIndex);
        int[] right = Arrays.copyOfRange(nums, maxIndex + 1, nums.length);
        out.left = constructMaximumBinaryTree(left);
        out.right = constructMaximumBinaryTree(right);
        return out;
    }

    public TreeNode buildTree(int[] inorder, int[] postorder) {
        int n = inorder.length;
        if (n == 0) return null;
        int currentNode = postorder[n - 1];
        int ioIndex = 0;
        //Find the index of the current node, in "inorder"
        for (int i = 0; i < n; i++) {
            if (inorder[i] == currentNode) {
                ioIndex = i;
                break;
            }
        }
        TreeNode out = new TreeNode(currentNode);
        int[] IOLeft = Arrays.copyOfRange(inorder, 0, ioIndex);
        int[] IORight = Arrays.copyOfRange(inorder, ioIndex + 1, n);
        int m = IORight.length;
        int poIndex = n - 1 - m;
        int[] PORight = Arrays.copyOfRange(postorder, poIndex, n - 1);
        int[] POLeft = Arrays.copyOfRange(postorder, 0, poIndex);
        out.left = buildTree(IOLeft, POLeft);
        out.right = buildTree(IORight, PORight);
        return out;
    }

    public int[] countSubTrees(int n, int[][] edges, String labels) {
        int[] output = new int[n];
        for (int i = 0; i < n; i++) {
            List<Integer> nodes = new ArrayList<>();
            nodes.add(i);
            for (int[] currentNode : edges) {
                int first = currentNode[0];
                int second = currentNode[1];
                if (nodes.contains(first)) {
                    nodes.add(second);
                }
            }
            char currentLabel = labels.charAt(i);
            int currentLabelCounter = 0;
            for (Integer I : nodes) {
                char currentChar = labels.charAt(I);
                if (currentChar == currentLabel) currentLabelCounter++;
            }
            output[i] = currentLabelCounter;
        }
        return output;
    }

    class FindElements {
        List<Integer> nodeList;

        public FindElements(TreeNode root) {
            nodeList = new ArrayList<>();
            recoverTree(0, root);
        }

        public void recoverTree(int currentVal, TreeNode root) {
            if (root == null) return;
            else {
                nodeList.add(currentVal);
                recoverTree((2 * currentVal) + 1, root.left);
                recoverTree((2 * currentVal) + 2, root.right);
            }
        }

        public boolean find(int target) {
            return nodeList.contains(target);
        }
    }

    public TreeNode trimBST(TreeNode root, int low, int high) {
        if (root == null) return null;
        int val = root.val;
        if (val < low) return trimBST(root.right, low, high);
        if (high < val) return trimBST(root.left, low, high);
        else {
            root.left = trimBST(root.left, low, high);
            root.right = trimBST(root.right, low, high);
            return root;
        }
    }

    public String tree2str(TreeNode root) {
        if (root == null) return null;
        String left = tree2str(root.left);
        String right = tree2str(root.right);
        String output = "" + root.val;
        if (left != null) output += "(" + left + ")";
        if (right != null) {
            if (left == null) output += "()(" + right + ")";
            else output += "(" + right + ")";
        }
        return output;
    }

    List<Integer> allElements;

    public List<Integer> getAllElements(TreeNode root1, TreeNode root2) {
        allElements = new ArrayList<>();
        getAllElements(root1);
        getAllElements(root2);
        Collections.sort(allElements);
        return allElements;
    }

    public void getAllElements(TreeNode root) {
        if (root == null) {
            return;
        }
        allElements.add(root.val);
        getAllElements(root.left);
        getAllElements(root.right);
    }

    public int maxProduct(TreeNode root) {
        if (root == null) {
            return 0;
        } else {
            int output = Integer.MIN_VALUE;
            int val = root.val;
            int leftSumVal = val + nodeSum(root.left);
            int leftSum = nodeSum(root.left);
            int rightSumVal = val + nodeSum(root.right);
            int rightSum = nodeSum(root.right);
            output = Math.max(output, leftSum * rightSumVal);
            output = Math.max(output, rightSum * leftSumVal);
            if (root.left != null) {
                root.left.val += rightSumVal;
                output = Math.max(output, maxProduct(root.left));
            }
            if (root.right != null) {
                root.right.val += leftSumVal;
                output = Math.max(output, maxProduct(root.right));
            }
            return (int) (output % (Math.pow(10, 9) + 7));
        }
    }

    public int nodeSum(TreeNode root) {
        if (root == null) {
            return 0;
        } else {
            return (root.val + nodeSum(root.left) + nodeSum(root.right));
        }
    }

    public String getDirections(TreeNode root, int startValue, int destValue) {
        if (root.val == startValue) return pathTo(destValue, root);
        String toStart = pathTo(startValue, root);
        String toEnd = pathTo(destValue, root);
        if (root.val == destValue) {
            return toStart.replaceAll("[LR]", "U");
        }
        char c = toStart.charAt(0);
        char cc = toEnd.charAt(0);
        System.out.println(toStart);
        System.out.println(toEnd);
        while (toStart.length() > 0 && toEnd.length() > 0) {
            c = toStart.charAt(0);
            cc = toEnd.charAt(0);
            if (c == cc) {
                toStart = toStart.substring(1);
                toEnd = toEnd.substring(1);
            }
        }
        toStart = toStart.replaceAll("[LR]", "U");
        return (toStart + toEnd);
    }

    public String pathTo(int target, TreeNode node) {
        if (node == null) return null;
        if (node.val == target) return "";
        else {
            if (node.left != null) {
                String left = pathTo(target, node.left);
                if (left != null) return ("L" + left);
            } else if (node.right != null) {
                String right = pathTo(target, node.right);
                if (right != null) return ("R" + right);
            }
            return null;
        }
    }

    public QuadNode intersect(QuadNode quadTree1, QuadNode quadTree2) {
        if (quadTree1.isLeaf && quadTree2.isLeaf) {
            quadTree1.val = quadTree1.val || quadTree2.val;
            return quadTree1;
        }
        if (quadTree1.isLeaf) {
            if (quadTree1.val) return quadTree1;
            else return quadTree2;
        }
        if (quadTree2.isLeaf) {
            if (quadTree2.val) return quadTree2;
            else return quadTree1;
        } else {
            QuadNode out = new QuadNode();
            out.val = false;
            out.isLeaf = false;
            out.topLeft = intersect(quadTree1.topLeft, quadTree2.topLeft);
            out.topRight = intersect(quadTree1.topRight, quadTree2.topRight);
            out.bottomLeft = intersect(quadTree1.bottomLeft, quadTree2.bottomLeft);
            out.bottomRight = intersect(quadTree1.bottomRight, quadTree2.bottomRight);
            return out;
        }
    }

    public int findBottomLeftValue(TreeNode root) {
        Deque<TreeNode> tnQueue = new ArrayDeque<>();
        tnQueue.add(root);
        while (!tnQueue.isEmpty()) {
            TreeNode currentNode = tnQueue.poll();
            if (tnQueue.isEmpty() && currentNode.left == null && currentNode.right == null) return currentNode.val;
            if (currentNode.right != null) tnQueue.add(currentNode.right);
            if (currentNode.left != null) tnQueue.add(currentNode.left);
        }
        return 0;
    }

    public int longestZigZag(TreeNode root) {
        if (root == null) return 0;
        int a = zagLeft(root);
        int b = zagRight(root);
        int c = longestZigZag(root.left);
        int d = longestZigZag(root.right);
        return Math.max(a, Math.max(b, Math.max(c, d)));
    }

    public int zagLeft(TreeNode root) {
        if (root == null) {
            return 0;
        } else {
            return 1 + zagRight(root.right);
        }
    }

    public int zagRight(TreeNode root) {
        if (root == null) {
            return 0;
        } else {
            return 1 + zagLeft(root.left);
        }
    }

    public int numTrees(int n) {
        if (n == 0) return 1;
        if (n == 1) return 1;
        if (n == 2) return 2;
        int output = 0;
        for (int i = 0; i < n; i++) {
            int j = n - 1 - i;
            output += (numTrees(i) * numTrees(j));
        }
        return output;
    }

    public boolean flipEquiv(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null) return true;
        if (root1 == null && root2 != null) return false;
        if (root1 != null && root2 == null) return false;
        if (root1.val != root2.val) return false;
        boolean one = flipEquiv(root1.left, root2.left) && flipEquiv(root1.right, root2.right);
        boolean two = flipEquiv(root1.left, root2.right) && flipEquiv(root1.right, root2.left);
        return (one || two);
    }

    Map<Integer, TreeNode> tnMap;

    public TreeNode createBinaryTree(int[][] descriptions) {
        tnMap = new HashMap<>();
        TreeNode newest = new TreeNode();
        TreeNode current;
        for (int[] node : descriptions) {
            int from = node[0];
            int to = node[1];
            int left = node[2];
            if (!tnMap.containsKey(from)) {
                newest = new TreeNode(from);
                tnMap.put(from, newest);
            }
            if (!tnMap.containsKey(to)) {
                current = new TreeNode(to);
                tnMap.put(to, current);
            }
            if (left == 1) {
                tnMap.get(from).left = tnMap.get(to);
            } else {
                tnMap.get(from).right = tnMap.get(to);
            }
        }
        return newest;
    }

    class StockPrice {
        HashMap<Integer, Integer> hm;
        int latestTime = Integer.MIN_VALUE;
        int max = Integer.MIN_VALUE;
        int min = Integer.MAX_VALUE;

        public StockPrice() {
            hm = new HashMap<>();
        }

        public void update(int timestamp, int price) {
            hm.put(timestamp, price);
            latestTime = Math.max(latestTime, timestamp);
            max = Math.max(max, price);
            min = Math.min(min, price);
        }

        public int current() {
            return hm.get(latestTime);
        }

        public int maximum() {
            List<Integer> l = new ArrayList<>(map.values());
            return Collections.max(l);
        }

        public int minimum() {
            List<Integer> l = new ArrayList<>(map.values());
            return Collections.min(l);
        }
    }

    public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        if (n == 1) return 1;
        int prior = nums[0];
        int max = Integer.MIN_VALUE;
        int currentMax = 1;
        for (int i = 1; i < n; i++) {
            int current = nums[i];
            if (prior < current) {
                currentMax++;
                max = Math.max(max, currentMax);

            } else {
                currentMax = 1;
            }
            prior = current;
        }
        return max;
    }

    public boolean increasingTriplet(int[] nums) {
        int n = nums.length;
        if (n < 3) return false;
        for (int i = 0; i < n - 2; i++) {
            int first = nums[i];
            for (int j = i + 1; j < n - 1; j++) {
                int second = nums[j];
                if (first >= second) continue;
                for (int k = j + 1; k < n; k++) {
                    int third = nums[k];
                    if (second < third) return true;
                }
            }
        }
        return false;
    }

    public int missingNumber(int[] nums) {
        int currentSum = 0;
        for (int i = 0; i < nums.length; i++) {
            currentSum += i + 1;
            currentSum -= nums[i];
        }
        return currentSum;
    }

    public int bulbSwitch(int n) {
        if (n == 1) return 1;
        if (n % 2 == 0) return (n / 2);
        else {
            return (bulbSwitch(n - 1));
        }
    }

    public void moveZeroes(int[] nums) {
        int indexTwo = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0) continue;
            else {
                nums[indexTwo] = nums[i];
                indexTwo++;
            }
        }
        for (int i = indexTwo; i < nums.length; i++) {
            nums[i] = 0;
        }
    }

    boolean isBadVersion(int version) {
        return true;
    }

    public int firstBadVersion(int n) {
        if (n == 1) return n;
        if (n == 2) {
            if (isBadVersion(1)) return 1;
            return 2;
        }
        int mid = (n / 2) + (n % 2);
        if (isBadVersion(mid)) return firstBadVersion(mid);
        else {
            boolean isCurrentVersion = isBadVersion(n);
            while (isCurrentVersion) {
                n = n - 1;
                isCurrentVersion = isBadVersion(n);
            }
            return n + 1;
        }
    }

    public ListNode sortList(ListNode head) {
        ListNode prev = new ListNode();
        ListNode current = head;
        ListNode next = current.next;
        if (next == null) return head;
        while (next != null) {
            int first = current.val;
            int second = next.val;
            if (first > second) {
                prev.next = next;
                current.next = next.next;
                next.next = current;
                return sortList(head);
            } else {
                prev = current;
                current = current.next;
                next = current.next;
            }
        }
        return head;
    }

    public void reorderList(ListNode head) {
        if (head == null) return;
        ListNode current = head;
        ListNode next = current.next;
        if (next == null || next.next == null) return;
        else {
            ListNode recurse = current.next;
            ListNode nextt = next.next;
            while (nextt.next != null) {
                next = next.next;
                nextt = next.next;
            }
            current.next = nextt;
            next.next = null;
            reorderList(recurse);
            nextt.next = recurse;
        }
    }

    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<TreeNode> tnList = new ArrayList<>();
        tnList.add(root);
        return levelOrderBottom2(tnList);
    }

    public List<List<Integer>> levelOrderBottom2(List<TreeNode> tnList) {
        if (tnList.size() == 0) {
            List<List<Integer>> output = new ArrayList<>();
            return output;
        }
        List<Integer> currentLevel = new ArrayList<>();
        List<TreeNode> nextTNLevel = new ArrayList<>();
        for (TreeNode tn : tnList) {
            currentLevel.add(tn.val);
            if (tn.left != null) nextTNLevel.add(tn.left);
            if (tn.right != null) nextTNLevel.add(tn.right);
        }
        var output = levelOrderBottom2(nextTNLevel);
        output.add(currentLevel);
        return output;
    }

    HashMap<Integer, Node> hmNode;

    public Node cloneGraph(Node node) {
        if (node == null) {
            return node;
        }
        hmNode = new HashMap<>();
        int val = node.val;
        cloneGraph2(node);
        return hmNode.get(val);
    }

    public void cloneGraph2(Node node) {
        int key = node.val;
        if (!hmNode.containsKey(key)) {
            Node currentNode = new Node(key);
            hmNode.put(key, currentNode);
            for (Node n : node.neighbors) {
                int currentNeighbourKey = n.val;
                cloneGraph2(n);
                hmNode.get(key).neighbors.add(hmNode.get(currentNeighbourKey));
            }
        }
    }

    int[][] heightMap;
    int xMax;
    int yMax;

    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        heightMap = heights;
        xMax = heightMap[0].length;
        yMax = heightMap.length;
        List<List<Integer>> out = new ArrayList<>();
        for (int y = 0; y < yMax; y++) {
            for (int x = 0; x < xMax; x++) {
                int[] currentCoordinates = {y, x};
                if (canReachAlantic(currentCoordinates) && canReachPacific(currentCoordinates)) {
                    List<Integer> currentList = new ArrayList<>();
                    currentList.add(y);
                    currentList.add(x);
                    out.add(currentList);
                }
            }
        }
        return out;
    }

    public boolean canReachAlantic(int[] in) {
        int y = in[0];
        int x = in[1];
        if (xMax - 1 == x || yMax - 1 == y) return true;
        else {
            int[] down = {y + 1, x};
            if (canReachAlantic(down) && heightMap[y][x] >= heightMap[y + 1][x]) return true;
            int[] right = {y, x + 1};
            if (canReachAlantic(right) && heightMap[y][x] >= heightMap[y][x + 1]) return true;
        }
        return false;
    }

    public boolean canReachPacific(int[] in) {
        int y = in[0];
        int x = in[1];
        if (x <= 0 || y <= 0) return true;
        else {
            int[] up = {y - 1, x};
            if (canReachPacific(up) && heightMap[y][x] >= heightMap[y - 1][x]) return true;
            int[] left = {y, x - 1};
            if (canReachPacific(left) && heightMap[y][x] >= heightMap[y][x - 1]) return true;
        }
        return false;
    }

    public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
        if (desiredTotal == maxChoosableInteger + 1) return false;
        if (maxChoosableInteger >= desiredTotal) return true;
        return (!canIWin(maxChoosableInteger - 1, desiredTotal - maxChoosableInteger));
    }

    //    public List<List<Integer>> levelOrder(Node root) {
//        List<Node> nodeList = new ArrayList<>();
//        nodeList.add(root);
//        return levelOrder2(nodeList);
//    }
    public int minCameraCover(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null && root.right == null) return 1;
        int first = 1;
        first += minCameraCover2(root.left);
        first += minCameraCover2(root.right);
        int second = 0;
        if (root.left != null) {
            second++;
            second += minCameraCover2(root.left.left);
            second += minCameraCover2(root.left.right);
        }
        if (root.right != null) {
            second++;
            second += minCameraCover2(root.right.left);
            second += minCameraCover2(root.right.right);
        }
        return (Math.min(first, second));
    }

    //Assume the given root is covered
    public int minCameraCover2(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null && root.right == null) return 0;
        int first = 1;
        first += minCameraCover2(root.left);
        first += minCameraCover2(root.right);
        int second = 0;
        if (root.left != null) {
            second++;
            second += minCameraCover2(root.left.left);
            second += minCameraCover2(root.left.right);
        }
        if (root.right != null) {
            second++;
            second += minCameraCover2(root.right.left);
            second += minCameraCover2(root.right.right);
        }
        return (Math.min(first, second));
    }


    public boolean allNegativeBST(TreeNode root) {
        if (root == null) return true;
        if (root.val >= 0) return false;
        return (allNegativeBST(root.left) && allNegativeBST(root.right));
    }


    public int[] sumOfDistancesInTree(int n, int[][] edges) {
        int[] output = new int[n];
        if (n == 1) return output;
        for (int i = 0; i < n; i++) {
            System.out.println(i);
            TreeMap<Integer, Integer> tm = new TreeMap<>();
            tm.put(i, 0);
            boolean flag = true;
            while (flag) {
                System.out.println(tm.keySet().toString());
                flag = false;
                for (int[] edge : edges) {
                    int first = edge[0];
                    int second = edge[1];
                    if (tm.keySet().contains(second) && !tm.keySet().contains(first)) {
                        first = edge[1];
                        second = edge[0];
                    }
                    System.out.println(first);
                    System.out.println(second);
                    if (tm.keySet().contains(first) && !tm.keySet().contains(second)) {
                        flag = true;
                        int priorDistance = tm.get(first);
                        tm.put(second, priorDistance + 1);
                    }
                }

            }
            int sum = 0;
            for (Integer I : tm.values()) sum += I;
            output[i] = sum;
        }
        return output;
    }

    public class Edge {
        List<Integer> edges = new ArrayList<>();
        boolean isCoin;
        boolean isLeaf;

        public Edge(boolean isCoin) {
            this.isCoin = isCoin;
        }

    }

    HashMap<Integer, Edge> edgeHM;

    public int collectTheCoins(int[] coins, int[][] edges) {
        if (coins.length <= 5) return 0;
        HashMap<Integer, Edge> edgeHM = new HashMap<>();
        for (int i = 0; i < coins.length; i++) {
            Edge currentEdge = new Edge(coins[i] == 1);
            for (int[] edge : edges) {
                if (edge[0] == i) currentEdge.edges.add(edge[1]);
                if (edge[1] == i) currentEdge.edges.add(edge[0]);
            }
            edgeHM.put(i, currentEdge);
        }

        //Remove all non-coin leafs;
        boolean flag = true;
        while (flag) {
            flag = false;
            Set<Integer> keys = edgeHM.keySet();
            List<Integer> toRemove = new ArrayList<>();
            for (Integer key : keys) {
                Edge e = edgeHM.get(key);
                if (!e.isCoin && e.edges.size() == 1) {
                    //Remove references to this leaf
                    edgeHM.get(e.edges.get(0)).edges.remove(key);
                    flag = true;
                    //Remove the lead
                    toRemove.add(key);
                }
            }
            for (Integer key : toRemove) {
                System.out.println(key);
                edgeHM.remove(key);
            }
        }
        //Reduce leaf coins by two
        for (int i = 0; i < 2; i++) {
            if (edgeHM.size() == 2) return 0;
            List<Integer> currentLeafCoins = new ArrayList<>();
            for (Integer key : edgeHM.keySet()) {
                Edge e = edgeHM.get(key);
                if (e.edges.size() == 1 && e.isCoin) {
                    System.out.println("removing " + key);
                    currentLeafCoins.add(key);

                }
            }
            for (Integer key : currentLeafCoins) {
                Edge e = edgeHM.get(key);
                edgeHM.get(e.edges.get(0)).edges.remove(key);
                edgeHM.get(e.edges.get(0)).isCoin = true;
            }
            for (Integer key : currentLeafCoins) {
                edgeHM.remove(key);
            }
        }

        Set<Integer> output = edgeHM.keySet();
        return ((output.size() - 1) * 2);
    }

    public List<Integer> findDuplicates(int[] nums) {
        Arrays.sort(nums);
        List<Integer> out = new ArrayList<>();
        if (nums.length == 1) return out;
        for (int i = 1; i < nums.length; i++) {
            int current = nums[i - 1];
            int next = nums[i];
            if (current == next) out.add(next);
        }
        return out;
    }

    public int threeSumMulti(int[] arr, int target) {
        int output = 0;
        int n = arr.length;
        Arrays.sort(arr);
        for (int i = 0; i < n - 2; i++) {
            for (int j = i + 1; j < n - 1; j++) {
                for (int k = j + 1; k < n; k++) {
                    int currentSum = (arr[i] + arr[j] + arr[k]);
                    if (currentSum == target) output++;
                    if (currentSum > target) break;
                }
            }
        }
        return (int) (output % (Math.pow(10, 9) + 7));
    }

    public List<List<String>> displayTable(List<List<String>> orders) {
        List<List<String>> output = new ArrayList<>();
        HashMap<String, List<String>> hmDisplay = new HashMap<>();
        Set<String> foodSet = new HashSet<>();
        for (List<String> ls : orders) {
            String currentTable = ls.get(1);
            String currentFood = ls.get(2);
            //Add to foodSet
            foodSet.add(currentFood);
            if (!hmDisplay.containsKey(currentTable)) {
                hmDisplay.put(currentTable, new ArrayList<>());
            }
            hmDisplay.get(currentTable).add(currentFood);
        }
        List<String> foodList = new ArrayList<>(foodSet);
        Collections.sort(foodList);
        for (String table : hmDisplay.keySet()) {
            List<String> currentTableDisplay = new ArrayList<>();
            var currentOrders = hmDisplay.get(table);
            currentTableDisplay.add(table);
            for (String food : foodList) {
                int foodFrequency = Collections.frequency(currentOrders, food);
                currentTableDisplay.add(String.valueOf(foodFrequency));
            }
            output.add(currentTableDisplay);
        }
        Collections.sort(output, new Comparator<List<String>>() {
            public int compare(List<String> a, List<String> b) {
                return (Integer.valueOf(a.get(0)) - Integer.valueOf(b.get(0)));
            }
        });
        foodList.add(0, "Table");
        output.add(0, foodList);
        return output;
    }


    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                return (a[0] - b[0]);
            }
        });
        int n = intervals.length;
        for (int i = 1; i < intervals.length; i++) {
            int[] firstInterval = intervals[i - 1];
            int firstStart = firstInterval[0];
            int firstEnd = firstInterval[1];
            int[] secondInterval = intervals[i];
            int secondStart = secondInterval[0];
            int secondEnd = secondInterval[1];
            if (firstStart <= secondStart && secondStart <= firstEnd) {
                secondInterval[0] = Math.min(firstStart, secondStart);
                secondInterval[1] = Math.max(firstEnd, secondEnd);
                intervals[i - 1] = null;
                n--;
            }
        }
        int i = 0;
        int[][] output = new int[n][];
        for (int[] intArray : intervals) {
            if (intArray != null) {
                output[i] = intArray;
                i++;
            }
        }
        return output;
    }

    public int kthSmallest(TreeNode root, int k) {
        if (root == null) return 0;
        List<TreeNode> rootList = new ArrayList<>();
        rootList.add(root);
        List<Integer> valueList = rootToList(rootList);
        Collections.sort(valueList);
        if (k >= valueList.size()) return 0;
        else return valueList.get(k - 1);
    }

    public List<Integer> rootToList(List<TreeNode> root) {
        List<Integer> out = new ArrayList<>();
        List<TreeNode> nextRootList = new ArrayList<>();
        if (root == null || root.size() == 0) return out;
        for (TreeNode tn : root) {
            out.add(tn.val);
            if (tn.left != null) nextRootList.add(tn.left);
            if (tn.right != null) nextRootList.add(tn.right);
        }
        out.addAll(rootToList(nextRootList));
        return out;
    }


    public void sortColors(int[] nums) {
        for (int i = 1; i < nums.length; i++) {
            int first = nums[i - 1];
            int second = nums[i];
            if (first > second) {
                nums[i - 1] = second;
                nums[i] = first;
                sortColors(nums);
                break;
            }
        }
    }


    public int[] findOriginalArray(int[] changed) {
        int halfLength = changed.length / 2;
        int[] output = new int[0];
        List<Integer> outputList = new ArrayList<>();
        Arrays.sort(changed);
        // Create a mutable list instead of an immutable one.
        List<Integer> list = new ArrayList<>(Arrays.stream(changed).boxed().toList());
        int index = 0;
        while (!list.isEmpty()) {
            Integer first = list.get(0);
            list.remove(first);
            Integer second = first * 2;
            if (list.contains(second)) {
                list.remove(second);
                outputList.add(first);
                index++;
            } else {
                return output;
            }
        }
        return outputList.stream().mapToInt(i -> i).toArray();
    }


    public int minSwaps(int[] nums) {
        int n = nums.length;
        int zero = 0, one = 0;
        for (int i : nums) {
            if (i == 0) zero++;
            else one++;
        }
        if (zero == 0 || one == 0) return 0;
        //Counting with ones
        int index = 0;
        int minimumZeroCount = Integer.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            index = i;
            if (nums[i] != 1) continue;
            int countZeros = 0;
            for (int j = 0; j < one; j++) {
                index = i + j;
                if (index >= n) index -= n;
                if (nums[index] != 1) countZeros++;
            }
            if (countZeros > minimumZeroCount) break;
            minimumZeroCount = Math.min(minimumZeroCount, countZeros);
        }
        //Counting with Zeros
        //Counting with ones
        index = 0;
        int minimumOneCount = Integer.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            index = i;
            if (nums[i] != 0) continue;
            int countOnes = 0;
            for (int j = 0; j < zero; j++) {
                index = i + j;
                if (index >= n) index -= n;
                if (nums[index] != 0) countOnes++;
            }
            if (countOnes > minimumOneCount) break;
            minimumOneCount = Math.min(minimumOneCount, countOnes);
        }
        System.out.println(minimumOneCount);
        System.out.println(minimumZeroCount);
        return Math.min(minimumOneCount, minimumZeroCount);
    }


    public List<Integer> goodDaysToRobBank(int[] security, int time) {
        List<Integer> output = new ArrayList<>();
        int n = security.length;
        if (time == 0) {
            for (int i = 0; i < security.length; i++) {
                output.add(i);
            }
            return output;
        }
        int minimumDays = (time * 2) + 1;
        if (security.length < minimumDays) return output;
        int[] nonIncreasingConsec = new int[n];
        int[] nonDecreasingConsec = new int[n];
        int currentConsec = 0;
        for (int i = 1; i < (n - time); i++) {
            int firstDay = security[i - 1];
            int secondDay = security[i];
            if (firstDay >= secondDay) {
                currentConsec++;
            } else {
                currentConsec = 0;
            }
            nonIncreasingConsec[i] = currentConsec;
        }
        currentConsec = 0;
        for (int i = n - 2; i >= time; i--) {
            int firstDay = security[i];
            int secondDay = security[i + 1];
            if (firstDay <= secondDay) {
                currentConsec++;
            } else {
                currentConsec = 0;
            }
            nonDecreasingConsec[i] = currentConsec;
        }
        for (int i = time; i < n - time; i++) {
            if (nonDecreasingConsec[i] >= time && nonIncreasingConsec[i] >= time) {
                output.add(i);
            }
        }
        return output;
    }

    class Player {
        int win;
        int loss;

        public Player() {
            this.win = 0;
            this.loss = 0;
        }

        public void addWin() {
            this.win++;
        }

        public void addLoss() {
            this.loss++;
        }
    }

    public List<List<Integer>> findWinners(int[][] matches) {
        HashMap<Integer, Player> playerHM = new HashMap<>();
        for (int[] arr : matches) {
            int winner = arr[0];
            int loser = arr[1];
            if (!playerHM.containsKey(winner)) playerHM.put(winner, new Player());
            if (!playerHM.containsKey(loser)) playerHM.put(loser, new Player());
            playerHM.get(winner).addWin();
            playerHM.get(loser).addLoss();
        }
        List<List<Integer>> output = new ArrayList<>();
        List<Integer> zeroLosses = new ArrayList<>();
        List<Integer> oneLoss = new ArrayList<>();
        for (var key : playerHM.keySet()) {
            Player current = playerHM.get(key);
            if (current.loss == 0) {
                zeroLosses.add(key);
            } else if (current.loss == 1) {
                oneLoss.add(key);
            }
        }
        Collections.sort(oneLoss);
        Collections.sort(zeroLosses);
        output.add(0, zeroLosses);
        output.add(1, oneLoss);
        return output;

    }


    public class Seed {
        int plantTime;
        int growTime;

        public Seed(int plantTime, int growTime) {
            this.plantTime = plantTime;
            this.growTime = growTime;
        }
    }

    public int earliestFullBloom(int[] plantTime, int[] growTime) {
        List<Seed> seedList = new ArrayList<>();
        for (int i = 0; i < plantTime.length; i++) {
            Seed current = new Seed(plantTime[i], growTime[i]);
            seedList.add(current);
        }
        Collections.sort(seedList, new Comparator<Seed>() {
            public int compare(Seed a, Seed b) {
                if (a.growTime != b.growTime) {
                    return (b.growTime - a.growTime);
                } else {
                    return (b.plantTime - a.plantTime);
                }
            }
        });
        int output = 0;
        int timeSkip = 0;
        for (Seed s : seedList) {
            int pTime = s.plantTime;
            int gTime = s.growTime;
            output += pTime;
            timeSkip = Math.max(timeSkip, (output + gTime));
        }
        return timeSkip;
    }

    public class Grid {
        int[][] matrix;
        //Non inclusive
        int xLimit;
        int yLimit;

        public Grid(int yLimit, int xLimit) {
            this.matrix = new int[yLimit][xLimit];
            this.xLimit = xLimit;
            this.yLimit = yLimit;
        }

        public void flood(int[] cell) {
            int y = cell[0];
            int x = cell[1];
            matrix[y - 1][x - 1] = 1;
        }

        public int get(int y, int x) {
            if (x < 0 || x >= xLimit) return 1;
            else return matrix[y][x];
        }

        public boolean canCross() {
            List<Integer> starting = new ArrayList<>();
            for (int x = 0; x < xLimit; x++) {
                if (matrix[0][x] == 0) starting.add(x);
            }
            if (starting.size() == 0) return false;
            for (Integer i : starting) {
                if (canCross2(0, i)) {
                    reset();
                    return true;
                }
            }
            return false;
        }

        public boolean canCross2(int y, int x) {
            if (y == yLimit) return true;
            if (x < 0 || x >= xLimit || y < 0) return false;
            if (matrix[y][x] == 1 || matrix[y][x] == 2) return false;
            matrix[y][x] = 2;
            return (canCross2(y + 1, x) || canCross2(y, x + 1) || canCross2(y, x - 1) || canCross2(y - 1, x));
        }

        public void printString() {
            for (int[] intArr : matrix) {
                System.out.println(Arrays.toString(intArr));
            }
        }

        public void reset() {
            for (int x = 0; x < xLimit; x++) {
                for (int y = 0; y < yLimit; y++) {
                    if (matrix[y][x] == 2) {
                        matrix[y][x] = 0;
                    }
                }
            }
        }
    }

    public int latestDayToCross(int row, int col, int[][] cells) {
        Grid g = new Grid(row, col);
        int output = 0;
        for (int[] cell : cells) {
            g.flood(cell);


            if (!g.canCross()) return output;
            else output++;

        }
        return output;
    }

    HashMap<Integer, List<Integer>> tnHM;
    HashMap<Integer, HashMap<Integer, List<Integer>>> layerHM;
    HashMap<Integer, List<Integer>> tnHM2;

    public List<List<Integer>> verticalTraversal(TreeNode root) {
        List<List<Integer>> out = new ArrayList<>();
        if (root == null) return out;
        tnHM = new HashMap<>();
        verticalTraversal2(root, 0, 0);
        //Refactor the layerHM
        List<Integer> layerKeyList = new ArrayList<>(layerHM.keySet());
        Collections.sort(layerKeyList);
        for (Integer layerKey : layerKeyList) {
            HashMap<Integer, List<Integer>> currentLayer = layerHM.get(layerKey);
            List<Integer> currentLayerXKey = new ArrayList<>(currentLayer.keySet());
            for (Integer currentLayerKey : currentLayerXKey) {
                List<Integer> currentLayerNodes = currentLayer.get(currentLayerKey);
                Collections.sort(currentLayerNodes);
                if (!tnHM.containsKey(currentLayerKey)) {
                    tnHM.put(currentLayerKey, new ArrayList<>());
                }
                var tnHMList = tnHM.get(currentLayerKey);
                tnHMList.addAll(currentLayerNodes);
            }
        }
        List<Integer> keyList = new ArrayList<>(tnHM.keySet());
        Collections.sort(keyList);
        for (Integer key : keyList) {
            out.add(tnHM.get(key));
        }
        return out;
    }

    public void verticalTraversal2(TreeNode root, int xIndex, int layer) {
        if (root == null) return;
        if (!layerHM.containsKey(layer)) {
            layerHM.put(layer, new HashMap<>());
        }
        tnHM2 = layerHM.get(layer);
        if (!tnHM2.containsKey(xIndex)) {
            tnHM2.put(xIndex, new ArrayList<>());
        }
        tnHM2.get(xIndex).add(root.val);
        verticalTraversal2(root.left, xIndex - 1, layer + 1);
        verticalTraversal2(root.right, xIndex + 1, layer + 1);
    }

    Deque<TreeNode> tnQueue;
    Integer subTreeMax;

    public int maxSumBST(TreeNode root) {
        subTreeMax = 0;
        tnQueue = new ArrayDeque<>();
        isValidBST2(root);
        for (TreeNode tn : tnQueue) {
            Integer I = sumBST(tn);
            subTreeMax = Math.max(subTreeMax, I);
        }
        return subTreeMax;
    }

    public void isValidBST2(TreeNode root) {
        if (root == null) return;
        boolean left = validateBST2(root.left, Long.MIN_VALUE, root.val);
        boolean right = validateBST2(root.right, root.val, Long.MAX_VALUE);
        if (left && right) tnQueue.add(root);
        else {
            isValidBST2(root.left);
            isValidBST(root.right);
        }
    }

    public boolean validateBST2(TreeNode root, long min, long max) {
        if (root == null) return true;
        if (root.val <= min || root.val >= max) return false;
        boolean left = validateBST2(root.left, min, root.val);
        boolean right = validateBST2(root.right, root.val, max);
        return (left && right);
    }

    public Integer sumBST(TreeNode root) {
        if (root == null) return 0;
        Integer value = root.val;
        Integer left = sumBST(root.left);
        Integer right = sumBST(root.right);
        value = value + left + right;
        subTreeMax = Math.max(subTreeMax, value);
        return value;
    }

    public int minimumDeletions(String s) {
        int n = s.length();
        boolean switched = false;
        //Counting errors from first switch to b
        int count1 = 0;
        for (int i = 0; i < n; i++) {
            char c = s.charAt(i);
            if (!switched) {
                if (c == 'b') switched = true;
            } else {
                if (c == 'a') count1++;
            }
        }
        //First switch to a from the back
        boolean switched2 = false;
        int count2 = 0;
        for (int i = n - 1; i >= 0; i--) {
            char c = s.charAt(i);
            if (!switched2) {
                if (c == 'a') switched2 = true;
            } else {
                if (c == 'b') count2++;
            }
        }
        //Count as if first half should be a, second is b
        int swapIndex = n / 2;
        int count3 = 0;
        int topIndex = n - 1;
        for (int i = 0; i < swapIndex; i++, topIndex--) {
            char c = s.charAt(i);
            char d = s.charAt(topIndex);
            if (c != 'a') count3++;
            if (d != 'b') count3++;
        }
        count3 += (n % 2);
        return Math.min(count1, Math.min(count2, count3));
    }


    public int[] canSeePersonsCount(int[] heights) {
        int n = heights.length;
        int[] output = new int[heights.length];
        Deque<Integer> queueStack = new ArrayDeque<>();
        for (int i = n - 1; i >= 0; i--) {
            int currentHeight = heights[i];
            if (queueStack.isEmpty()) {
                output[i] = 0;
                queueStack.addFirst(currentHeight);
            } else {
                int nextHeight = queueStack.peek();
                if (currentHeight < nextHeight) {
                    output[i] = 1;
                    queueStack.addFirst(currentHeight);
                } else if (currentHeight == nextHeight) {
                    output[i] = 1;
                } else {
                    while (!queueStack.isEmpty()) {
                        nextHeight = queueStack.peek();
                        if (currentHeight > nextHeight) {
                            queueStack.pop();
                            output[i]++;
                        } else if (currentHeight == nextHeight) {
                            queueStack.pop();
                            output[i]++;
                            break;
                        } else {
                            output[i]++;
                            break;
                        }
                    }
                    queueStack.addFirst(currentHeight);
                }
            }
        }
        return output;
    }


    public static void main(String[] args) {
        int[][] edges = {{0, 1}, {0, 2}, {1, 3}, {1, 4}, {2, 5}, {5, 6}, {5, 7}};
        int[] coins = {0, 0, 0, 1, 1, 0, 0, 1};
    }
}

import com.sun.security.jgss.GSSUtil;
import com.sun.source.tree.Tree;

import java.util.*;
import java.util.stream.Collectors;

public class Leetcode3 extends Leetcode2 {

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
        dpArray3D[index][xZeros][xOnes] = maxForm(sArray, xZeros, xOnes, index + 1);
        if ((xZeros - zeroCount) >= 0 && (xOnes - oneCount) >= 0) {
            dpArray3D[index][xZeros][xOnes]
                    = Math.max(dpArray3D[index][xZeros][xOnes], maxForm(sArray, xZeros - zeroCount, xOnes - oneCount, index + 1));
        }
        return dpArray3D[index][xZeros][xOnes];
    }


    public int findRadius(int[] houses, int[] heaters) {
        Arrays.sort(houses);
        Arrays.sort(heaters);
        int heaterIndex = 0;
        //House to heater distance
        int minDistance = -1;
        for (int house : houses) {
            int distance = Integer.MAX_VALUE;
            for (int i = heaterIndex; i < heaters.length; i++) {
                int heaterPos = heaters[i];
                int currentDistance = Math.abs(heaterPos - house);
                if (currentDistance <= minDistance) {
                    distance = currentDistance;
                    break;
                }
                if (currentDistance <= distance) {
                    distance = currentDistance;
                    heaterIndex = i;
                } else {
                    break;
                }
            }
            minDistance = Math.max(minDistance, distance);
        }
        return minDistance;
    }


    public int subarraySum(int[] nums, int k) {
        int n = nums.length;
        if (n == 1) return (nums[0] == k) ? 1 : 0;
        int output = 0;
        for (int i = 1; i < nums.length; i++) {
            nums[i] += nums[i - 1];
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == k) output++;
            for (int j = i - 1; j >= 0; j--) {
                if (nums[i] - nums[j] == k) output++;
            }
        }
        return output;
    }


    public boolean validSquare(int[] p1, int[] p2, int[] p3, int[] p4) {
        double distanceA = Double.MAX_VALUE;
        double distanceB = Double.MIN_VALUE;
        double[][] square = new double[4][2];
        square[0][0] = p1[0];
        square[0][1] = p1[1];
        square[1][0] = p2[0];
        square[1][1] = p2[1];
        square[2][0] = p3[0];
        square[2][1] = p3[1];
        square[3][0] = p4[0];
        square[3][1] = p4[1];
        double x = square[0][0];
        double y = square[0][1];
        for (int i = 1; i < 4; i++) {
            double[] currentSquare = square[i];
            double xx = currentSquare[0];
            double yy = currentSquare[1];
            double currentDistance = Math.sqrt(Math.pow(Math.abs(xx - x), 2) + Math.pow(Math.abs(yy - y), 2));
            distanceA = Math.min(distanceA, currentDistance);
            distanceB = Math.max(distanceB, currentDistance);
        }
        for (int i = 1; i < 4; i++) {
            int checkSum = 0;
            x = square[i][0];
            y = square[i][1];
            for (int j = 0; j < 4; j++) {
                if (j == i) continue;
                double xx = square[j][0];
                double yy = square[j][1];
                double currentDistance = Math.sqrt(Math.pow(Math.abs(xx - x), 2) + Math.pow(Math.abs(yy - y), 2));
                if (currentDistance == distanceA) checkSum++;
                else if (currentDistance == distanceB) checkSum = checkSum + 2;
                else return false;
            }
            if (checkSum != 4) return false;
        }
        return true;
    }


    public int findLongestChain(int[][] pairs) {
        int n = pairs.length;
        Arrays.sort(pairs, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                int a1 = a[0];
                int a2 = a[1];
                int b1 = b[0];
                int b2 = b[1];
                if (a2 != b2) return (b2 - a2);
                return (b1 - a1);
            }
        });
        int[] dp = new int[2002];
        int index = 2001;
        int start = 0;
        int end = 0;
        for (int[] pair : pairs) {
            start = pair[0];
            end = pair[1];
            while (index != end + 1) {
                dp[index] = Math.max(dp[index], dp[index + 1]);
                index--;
            }
            dp[start] = Math.max(dp[start], dp[index] + 1);
        }
        return dp[start];
    }


    public boolean canPartitionKSubsets(int[] nums, int k) {
        if (k == 1) return true;
        long sum = 0;
        for (int i : nums) sum += i;
        if (sum % k != 0) return false;
        int[] buckets = new int[k];
        int target = (int) (sum / k);
        return canPartitionKSubsets2(nums, 0, target, buckets);
    }

    public boolean canPartitionKSubsets2(int[] nums, int index, int targetSum, int[] buckets) {
        if (index == nums.length) return true;
        int currentNumber = nums[index];
        for (int i = 0; i < buckets.length; i++) {
            int currentBucket = buckets[i];
            if (currentBucket + currentNumber <= targetSum) {
                buckets[i] += currentBucket;
                if (canPartitionKSubsets2(nums, index + 1, targetSum, buckets)) return true;
                buckets[i] -= currentBucket;
            }
        }
        return false;
    }


    public class Job {
        int start;
        int end;
        int profit;

        public Job(int start, int end, int profit) {
            this.start = start;
            this.end = end;
            this.profit = profit;
        }

        public static final Comparator<Job> endComparator = new Comparator<Job>() {
            @Override
            public int compare(Job j1, Job j2) {
                return Integer.compare(j2.end, j1.end);
            }
        };


    }

    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
        int n = startTime.length;
        if (n == 1) return profit[0];
        Job[] jobs = new Job[n];
        for (int i = 0; i < n; i++) {
            jobs[i] = new Job(startTime[i], endTime[i], profit[i]);
        }
        Arrays.sort(jobs, Job.endComparator);
        int[] dpArray = new int[jobs[0].end + 1];
        int dpPointer = jobs[0].end;
        int start = 0;
        int output = 0;
        for (Job j : jobs) {
            start = j.start;
            int end = j.end;
            int p = j.profit;
            System.out.println("start, end, profit, dpPointer in the loop");
            System.out.println(start);
            System.out.println(end);
            System.out.println(profit);
            while (dpPointer > end) {
                System.out.println(dpPointer);
                dpPointer--;
                dpArray[dpPointer] = Math.max(dpArray[dpPointer], dpArray[dpPointer + 1]);
            }
            System.out.println("after while loop, dpArray[start], dpArray[dpPointer] + p, dpstart after ");
            System.out.println(dpArray[start]);
            System.out.println(dpArray[dpPointer] + p);
            dpArray[start] = Math.max(dpArray[start], dpArray[dpPointer] + p);
            System.out.println(dpArray[start]);
            output = Math.max(output, dpArray[start]);
        }
        System.out.println(Arrays.toString(dpArray));
        return output;
    }


    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> a = combSum2(candidates, 0, target);
        Set<List<Integer>> s = new HashSet<>(a);
        return new ArrayList<>(s);
    }

    public List<List<Integer>> combSum2(int[] marks, int startIndex, int target) {
        List<List<Integer>> out = new ArrayList<>();
        if (target < 0 || startIndex == marks.length) return out;
        if (target == 0) {
            out.add(new ArrayList<>());
            return out;
        } else {
            out = combSum2(marks, startIndex + 1, target);
            int currentMark = marks[startIndex];
            var nextComb = combSum2(marks, startIndex + 1, target - currentMark);
            for (var v : nextComb) {
                v.add(currentMark);
                Collections.sort(v);
                if (!out.contains(v)) out.add(v);
            }
            return out;
        }
    }


    static HashMap<Integer, List<List<Integer>>> pathsFromKeyToEnd;

    public static List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        int n = graph.length;
        pathsFromKeyToEnd = new HashMap<>();
        List<Integer> lastNode = new ArrayList<>();
        lastNode.add(n - 1);
        List<List<Integer>> dList = new ArrayList<>();
        dList.add(lastNode);
        pathsFromKeyToEnd.put(n - 1, dList);
        pathDFS(0, graph);
        return pathsFromKeyToEnd.get(0);
    }

    public static void pathDFS(int currentNode, int[][] graph) {
        List<List<Integer>> list = new ArrayList<>();
        for (int path : graph[currentNode]) {
            if (pathsFromKeyToEnd.get(path) == null) {
                pathDFS(path, graph);
            }
            List<List<Integer>> dList = pathsFromKeyToEnd.get(path);
            for (List<Integer> currentList : dList) {
                List<Integer> currentListCopy = new ArrayList<>(currentList);
                currentListCopy.add(0, currentNode);
                list.add(currentListCopy);
            }
        }
        pathsFromKeyToEnd.put(currentNode, list);
    }


    public List<Integer> largestValues(TreeNode root) {
        List<Integer> output = new ArrayList<>();
        if (root == null) return output;
        Deque<TreeNode> dq = new ArrayDeque<>();
        dq.add(root);
        while (!dq.isEmpty()) {
            int currentMax = Integer.MAX_VALUE;
            Deque<TreeNode> nextDQ = new ArrayDeque<>();
            while (!dq.isEmpty()) {
                TreeNode tn = dq.pollFirst();
                currentMax = Math.max(currentMax, tn.val);
                if (tn.left != null) nextDQ.add(tn.left);
                if (tn.right != null) nextDQ.add(tn.right);
            }
            output.add(output.size(), currentMax);
            dq = nextDQ;
        }
        return output;
    }


    public int longestSubarray(int[] nums) {
        int output = 0;
        int priorLength = 0;
        int currentLength = 0;
        for (int i : nums) {
            switch (i) {
                case 1 -> currentLength++;
                case 0 -> {
                    output = Math.max(output, priorLength + currentLength);
                    priorLength = currentLength;
                    currentLength = 0;
                }

            }
        }
        output = Math.max(output, priorLength + currentLength);
        if (output == nums.length) return output - 1;
        else return output;
    }


    public List<String> printVertically(String s) {
        HashMap<Integer, String> hm = new HashMap<>();
        String[] sArray = s.split(" ");
        for (String ss : sArray) {
            int n = ss.length();
            for (int i = 0; i < n; i++) {
                char c = ss.charAt(i);
                if (!hm.containsKey(i)) hm.put(i, "");
                String current = hm.get(i) + c;
                hm.put(i, current);
            }
        }
        List<String> output = new ArrayList<>(hm.values());
        return output;
    }


    public boolean validPath(int n, int[][] edges, int source, int destination) {
        if (source == destination) return true;
        Deque<Integer> queue = new ArrayDeque<>();
        Set<Integer> visited = new HashSet<>();
        queue.add(source);
        visited.add(source);
        while (!queue.isEmpty()) {
            int currentSource = queue.pollFirst();
            for (int[] edge : edges) {
                int a = edge[0];
                int b = edge[1];
                if (a == currentSource || b == currentSource) {
                    int currentDest = (a == currentSource) ? b : a;
                    if (currentDest == destination) return true;
                    if (!visited.contains(currentDest)) {
                        visited.add(currentDest);
                        queue.addLast(currentDest);
                    }
                } else {
                    continue;
                }
            }
        }
        return false;
    }


    public int findCenter(int[][] edges) {
        HashMap<Integer, Integer> hm = new HashMap<>();
        for (int[] edge : edges) {
            for (int i : edge) {
                hm.merge(i, 1, Integer::sum);
            }
        }
        int key = Collections.max(hm.entrySet(), Map.Entry.comparingByValue()).getKey();
        return key;
    }


    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int K) {
        int[][] graph = new int[n][n];
        for (int[] row : graph) {
            Arrays.fill(row, Integer.MAX_VALUE);
        }
        for (int[] flight : flights) {
            graph[flight[0]][flight[1]] = flight[2];
        }

        int[] costs = new int[n];
        Arrays.fill(costs, Integer.MAX_VALUE);
        costs[src] = 0;

        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{src, 0});

        int steps = 0;
        while (!queue.isEmpty() && steps <= K) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int[] current = queue.poll();
                int currentCity = current[0];
                int currentCost = current[1];
                for (int nextCity = 0; nextCity < n; nextCity++) {
                    if (graph[currentCity][nextCity] != Integer.MAX_VALUE) {
                        int nextCost = currentCost + graph[currentCity][nextCity];
                        if (nextCost < costs[nextCity]) {
                            costs[nextCity] = nextCost;
                            queue.offer(new int[]{nextCity, nextCost});
                        }
                    }
                }
            }
            steps++;
        }

        return costs[dst] == Integer.MAX_VALUE ? -1 : costs[dst];
    }


    public int networkDelayTime(int[][] times, int n, int k) {
        int[] dist = new int[n];
        Arrays.fill(dist, Integer.MAX_VALUE);
        Map<Integer, List<int[]>> graph = new HashMap<>();
        for (int[] time : times) {
            graph.computeIfAbsent(time[0] - 1, x -> new ArrayList<>()).add(new int[]{time[1] - 1, time[2]});
        }
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
        pq.add(new int[]{k - 1, 0});
        while (!pq.isEmpty()) {
            int[] curr = pq.poll();
            int node = curr[0];
            int time = curr[1];
            if (time > dist[node]) continue;
            if (graph.containsKey(node)) {
                for (int[] edge : graph.get(node)) {
                    int next = edge[0];
                    int nextTime = time + edge[1];
                    if (nextTime < dist[next]) {
                        dist[next] = nextTime;
                        pq.offer(new int[]{next, nextTime});
                    }
                }
            }
        }
        int maxWait = Arrays.stream(dist).max().getAsInt();
        return maxWait == Integer.MAX_VALUE ? -1 : maxWait;
    }


    public int countStudents(int[] students, int[] sandwiches) {
        int n = students.length;
        if (n == 1) return (students[0] == sandwiches[0]) ? 0 : 1;
        Deque<Integer> queue = new ArrayDeque<>();
        Deque<Integer> food = new ArrayDeque<>();
        for (int i : students) queue.addLast(i);
        for (int i : sandwiches) queue.addLast(i);
        while (n > 0) {
            n--;
            int currentStudent = queue.pollFirst();
            int currentFood = food.pollFirst();
            if (currentStudent != currentFood) {
                queue.addLast(currentStudent);
                food.addLast(currentFood);
            } else {
                n = queue.size();
            }
        }
        return queue.size();
    }


    public int[] asteroidCollision(int[] asteroids) {
        Deque<Integer> dq = new ArrayDeque<>();
        for (int asteroid : asteroids) {
            if (dq.isEmpty()) dq.addLast(asteroid);
            else {
                int priorAsteroid = dq.peekLast();
                if (priorAsteroid > 0 && asteroid > 0
                        || priorAsteroid < 0 && asteroid < 0
                        || priorAsteroid < 0 && asteroid > 0) dq.addLast(asteroid);
                else {
                    while (!dq.isEmpty() && dq.peekLast() > 0 && dq.peekLast() < Math.abs(asteroid)) {

                        dq.pollLast();
                    }
                    if (dq.isEmpty() || dq.peekLast() < 0) {

                        dq.addLast(asteroid);
                    } else if (dq.peekLast() == Math.abs(asteroid)) {
                        dq.pollLast();
                    }
                }
            }
        }
        int[] output = new int[dq.size()];
        int i = 0;
        while (!dq.isEmpty()) {
            output[i] = dq.pollFirst();
            i++;
        }
        return output;
    }


    public int[] sumEvenAfterQueries(int[] nums, int[][] queries) {
        long evenSum = 0;
        for (int i : nums) if (i % 2 == 0) evenSum += i;
        int[] output = new int[queries.length];
        for (int i = 0; i < queries.length; i++) {
            int[] query = queries[i];
            int addValue = query[0];
            int index = query[1];
            if (nums[index] % 2 == 0) evenSum -= nums[index];
            nums[index] += addValue;
            if (nums[index] % 2 == 0) evenSum += nums[index];
            output[i] = (int) evenSum;
        }
        return output;
    }


    public boolean isRobotBounded(String instructions) {
        int x = 0;
        int y = 0;
        List<int[]> pathLogic = Arrays.asList(
                new int[]{0, 1},  // Right
                new int[]{1, 0},  // Down
                new int[]{0, -1}, // Left
                new int[]{-1, 0}  // Up
        );
        for (int i = 0; i < 4; i++) {
            for (char c : instructions.toCharArray()) {
                switch (c) {
                    case 'G' -> {
                        int[] path = pathLogic.get(0);
                        x += path[0];
                        y += path[1];
                    }
                    case 'R' -> Collections.rotate(pathLogic, -1);
                    case 'L' -> Collections.rotate(pathLogic, 1);
                }
            }
            if (x == 0 && y == 0) return true;
        }
        return (x == 0 && y == 0);
    }


    public int[] missingRolls(int[] rolls, int mean, int n) {
        int totalRolls = rolls.length + n;
        int upperBound = n * 6;
        int lowerBound = n * 1;
        int knownSum = 0;
        for (int i : rolls) knownSum += i;
        int missingSum = (mean * totalRolls) - knownSum;
        if (missingSum < lowerBound || upperBound < missingSum) return new int[0];
        int[] output = new int[n];
        for (int i = 0; i < n; i++) {
            if (i == n - 1) {
                output[i] = missingSum;
            } else {
                output[i] = missingSum / (n - i);
                missingSum = missingSum - output[i];
            }
        }
        return output;
    }


    public int findJudge(int n, int[][] trust) {
        if (n == 1) return (trust.length == 0) ? 1 : -1;
        if (trust.length < n - 1) return -1;
        Set<Integer> visited = new HashSet<>();
        int[] trustArray = new int[n + 1];
        for (int[] t : trust) {
            visited.add(t[0]);
            trustArray[t[1]]++;
        }
        System.out.println(Arrays.toString(trustArray));
        for (int i = 1; i < n + 1; i++) {
            if (trustArray[i] == n - 1 && !visited.contains(i)) return i;
        }
        return -1;
    }


    public List<Integer> eventualSafeNodes(int[][] graph) {
        Set<Integer> recurrentStates = new HashSet<>();
        List<Integer> output = new ArrayList<>();
        boolean activeSearch = true;
        while (activeSearch) {
            int priorSetSize = recurrentStates.size();
            for (int i = 0; i < graph.length; i++) {
                if (recurrentStates.contains(i)) continue;
                int[] pathTo = graph[i];
                if (pathTo.length == 0) recurrentStates.add(i);
                else {
                    boolean allPathToRecurrent = true;
                    for (int path : pathTo) {
                        if (!recurrentStates.contains(path)) {
                            allPathToRecurrent = false;
                            break;
                        }
                    }
                    if (allPathToRecurrent) recurrentStates.add(i);
                }
            }
            activeSearch = (priorSetSize != recurrentStates.size());
        }
        System.out.println(recurrentStates);
        for (int i = 0; i < graph.length; i++) {
            if (recurrentStates.contains(i)) continue;
            int[] pathTo = graph[i];
            for (int path : pathTo) {
                if (recurrentStates.contains(path)) {
                    output.add(i);
                    break;
                }
            }
        }
        for (int i : recurrentStates) output.add(i);
        Collections.sort(output);
        return output;
    }


//    public static List<List<Integer>> permute(int[] nums) {
//        List<Integer> list = new ArrayList<>(Arrays.stream(nums).boxed().toList());
//        if (nums.length == 1) {
//            List<List<Integer>> output = new ArrayList<>();
//            output.add(list);
//            return output;
//        } else {
//            return permute2(list);
//        }
//
//    }
//    public static List<List<Integer>> permute2(List<Integer> input) {
//        List<Integer> currentList = new ArrayList<>(input);
//        List<List<Integer>> output = new ArrayList<>();
//        if (input.size() == 1) {
//            output.add(currentList);
//            return output;
//        } else if (input.size() == 2) {
//            output.add(currentList);
//            currentList = new ArrayList<>(currentList);
//            Collections.rotate(currentList,-1);
//            output.add(currentList);
//            return output;
//        } else {
//            for (int i = 0; i < input.size(); i++) {
//                Integer first = currentList.get(0);
//                List<List<Integer>> nextList = permute2(currentList.subList(1, currentList.size()));
//                for (List<Integer> l : nextList) {
//                    List<Integer> ll = new ArrayList<>(l);
//                    ll.add(0, first);
//                    output.add(ll);
//                }
//                Collections.rotate(currentList, -1);
//            }
//        }
//        return output;
//    }


//    public List<List<Integer>> permute(int[] nums) {
//        List<List<Integer>> output = new ArrayList<>();
//        List<Integer> numsList = new ArrayList<>(Arrays.stream(nums).boxed().toList());
//        if (nums.length == 1) {
//            output.add(numsList);
//            return output;
//        } else {
//            permute2(output, new ArrayList<>(), numsList, nums.length);
//            return output;
//        }
//    }
//    public void permute2(List<List<Integer>> output, List<Integer> subOutput, List<Integer> numsList, int limit) {
//        if (subOutput.size() == limit) output.add(new ArrayList<>(subOutput));
//        else {
//            for (Integer I : numsList) {
//                if (!subOutput.contains(I)) {
//                    subOutput.add(I);
//                    permute2(output,subOutput, numsList, limit);
//                    subOutput.remove(I);
//                }
//            }
//        }
//    }


    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> output = new ArrayList<>();
        Deque<Integer> dqNums = new ArrayDeque<>(Arrays.stream(nums).boxed().toList());
        if (nums.length == 1) {
            output.add(new ArrayList<>(dqNums));
            return output;
        } else {
            permute2(output, new ArrayDeque<>(), dqNums);
            return output;
        }
    }

    public void permute2(List<List<Integer>> output, Deque<Integer> dq, Deque<Integer> nums) {
        int n = nums.size();
        if (n == 0) {
            output.add(new ArrayList<>(dq));
        } else {
            for (int i = 0; i < n; i++) {
                dq.addLast(nums.pollFirst());
                permute2(output, dq, nums);
                nums.addLast(dq.pollLast());
            }
        }
    }


    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> output = new ArrayList<>();
        Deque<Integer> dqNums = new ArrayDeque<>(Arrays.stream(nums).boxed().toList());
        if (nums.length == 1) {
            output.add(new ArrayList<>(dqNums));
            return output;
        } else {
            permuteUnique2(output, new ArrayDeque<>(), dqNums);
            return output;
        }
    }

    public void permuteUnique2(List<List<Integer>> output, Deque<Integer> dq, Deque<Integer> nums) {
        int n = nums.size();
        if (n == 0) {
            output.add(new ArrayList<>(dq));
        } else {
            Set<Integer> s = new HashSet<>();
            for (int i = 0; i < n; i++) {
                Integer I = nums.pollFirst();
                if (!s.contains(I)) {
                    s.add(I);
                    dq.addLast(I);
                    permuteUnique2(output, dq, nums);
                    nums.addLast(dq.pollLast());
                } else {
                    nums.addLast(I);
                }
            }
        }
    }

    public int edgeScore(int[] edges) {
        int n = edges.length;
        long[] scoreArray = new long[n];
        for (int i = 0; i < n; i++) {
            int edgeTo = edges[i];
            scoreArray[edgeTo] += i;
        }
        System.out.println(Arrays.toString(scoreArray));
        long currentMax = scoreArray[n - 1];
        int maxNode = n - 1;
        for (int i = n - 1; i >= 0; i--) {
            long score = scoreArray[i];
            if (score >= currentMax) {
                currentMax = score;
                maxNode = i;
            }
        }
        return maxNode;
    }


    public List<Integer> findSmallestSetOfVertices(int n, List<List<Integer>> edges) {
        int[] graphDP = new int[n];
        for (List<Integer> edge : edges) {
            int edgeTo = edge.get(1);
            graphDP[edgeTo] = 1;
        }
        List<Integer> output = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (graphDP[i] == 0) output.add(i);
        }
        return output;
    }


    public String lastNonEmptyString(String s) {
        int n = s.length();
        if (n == 1) return s;
        HashMap<Character, Integer> frequencyMap = new HashMap<>();
        char[] cArray = s.toCharArray();
        for (Character c : cArray) frequencyMap.merge(c, 1, Integer::sum);
        int maxValueInMap = (Collections.max(frequencyMap.values()));
        List<Character> charList = new ArrayList<>();
        for (Map.Entry<Character, Integer> entry : frequencyMap.entrySet()) {
            if (entry.getValue() == maxValueInMap) charList.add(entry.getKey());
        }
        Collections.sort(charList, new Comparator<Character>() {
            public int compare(Character a, Character b) {
                return (s.lastIndexOf(a) - s.lastIndexOf(b));
            }
        });
        String output = "";
        for (Character c : charList) output += c;
        return output;
    }


    public int[] pivotArray(int[] nums, int pivot) {
        int n = nums.length;
        int[] output = new int[n];
        Deque<Integer> dq = new ArrayDeque<>();
        int index = 0;
        for (int i : nums) {
            if (i < pivot) {
                output[index] = i;
                index++;
            } else if (i == pivot) {
                dq.addFirst(i);
            } else {
                dq.addLast(i);
            }
        }
        while (!dq.isEmpty()) {
            output[index] = dq.pollFirst();
            index++;
        }
        return output;
    }


    public int[] rearrangeArray(int[] nums) {
        int n = nums.length;
        Deque<Integer> positive = new ArrayDeque<>();
        Deque<Integer> negative = new ArrayDeque<>();
        for (int i : nums) {
            if (i < 0) negative.addLast(i);
            else positive.addLast(i);
        }
        int index = 0;
        while (!positive.isEmpty() && !negative.isEmpty()) {
            nums[index] = positive.pollFirst();
            index++;
            nums[index] = negative.pollFirst();
            index++;
        }
        positive = (positive.isEmpty()) ? negative : positive;
        while (!positive.isEmpty()) {
            nums[index] = positive.pollFirst();
            index++;
        }
        return nums;
    }


    public String addSpaces(String s, int[] spaces) {
        int sLength = s.length();
        char[] cArray = new char[sLength + spaces.length];
        int outputIndex = 0;
        int spaceIndex = 0;
        int sIndex = 0;
        for (char c : s.toCharArray()) {
            int currentSpace = spaces[spaceIndex];
            if (sIndex == currentSpace) {
                cArray[outputIndex] = ' ';
                outputIndex++;
                spaceIndex = Math.min(spaceIndex + 1, spaces.length - 1);
            }
            cArray[outputIndex] = c;
            outputIndex++;
            sIndex++;
        }
        return String.valueOf(cArray);
    }


    public int[][] onesMinusZeros(int[][] grid) {
        int yMax = grid.length;
        int xMax = grid[0].length;
        int[][] output = new int[yMax][xMax];
        int[] rowSum = new int[yMax];
        int[] colSum = new int[xMax];


        for (int y = 0; y < yMax; y++) {
            for (int x = 0; x < xMax; x++) {
                int currentSquare = grid[y][x];
                currentSquare = (currentSquare == 1) ? 1 : -1;
                rowSum[y] += currentSquare;
                colSum[x] += currentSquare;
            }
        }
        for (int y = 0; y < yMax; y++) {
            for (int x = 0; x < xMax; x++) {
                output[y][x] = rowSum[y] + colSum[x];
            }
        }

        return output;
    }


    public long taskSchedulerII(int[] tasks, int space) {
        int n = tasks.length;
        int spaceInclusive = space + 1;
        HashMap<Integer, Long> hmIndex = new HashMap<>();
        long dayCounter = 0;
        for (int task : tasks) {
            dayCounter++;
            if (hmIndex.containsKey(task)) {
                Long previousIndex = hmIndex.get(task);
                dayCounter = Math.max(dayCounter, previousIndex + spaceInclusive);
            }
            hmIndex.put(task, dayCounter);
        }
        return dayCounter;
    }


    public int[] arrayChange(int[] nums, int[][] operations) {
        int n = nums.length;
        HashMap<Integer, Integer> hm = new HashMap<>();
        //HashMap
        //Key = An element of nums. Note: Every element in nums is unique
        //Value = The index in nums corresponding to the Key
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            hm.put(num, i);
        }
        for (int[] operation : operations) {
            int a = operation[0];
            int b = operation[1];
            hm.put(b, hm.remove(a));
        }
        for (var entry : hm.entrySet()) {
            nums[entry.getValue()] = entry.getKey();
        }
        return nums;
    }


    public List<Integer> relocateMarbles(int[] nums, int[] moveFrom, int[] moveTo) {
        int n = nums.length;
        HashMap<Integer, Integer> hm = new HashMap<>();
        for (int i : nums) hm.put(i, i);
        int m = moveFrom.length;
        for (int i = 0; i < m; i++) {
            int from = moveFrom[i];
            int to = moveTo[i];
            hm.remove(from);
            hm.put(to, to);
        }
        List<Integer> output = new ArrayList<>(hm.keySet());
        Collections.sort(output);
        return output;
    }


    public int minOperationsMaxProfit(int[] customers, int boardingCost, int runningCost) {
        int maxProfit = 0;
        int profitRotations = -1;
        int countRotations = 0;
        int profit = 0;
        Deque<Integer> dq = new ArrayDeque<>(Arrays.stream(customers).boxed().toList());
        while (!dq.isEmpty()) {
            int customer = dq.pollFirst();
            if (customer > 4) {
                dq.addFirst(customer - 4);
                customer = 4;

            }
            countRotations++;
            profit += (customer * boardingCost * 2) - runningCost;
            if (profit > maxProfit) {
                maxProfit = profit;
                profitRotations = countRotations;
            }
        }
        return profitRotations;
    }


//    public boolean isMatch(String s, String p) {
//        int sLength = s.length();
//        while (p.length() >= 2 && p.charAt(0) == '*' && p.charAt(1) == '*') p = p.substring(1);
//        if (p.equals("*")) return true;
//        int pLength = p.length();
//        Boolean[][] memory = new Boolean[sLength][pLength];
//        return isMatch(s.toCharArray(), p.toCharArray(), 0, 0, memory);
//    }
//    public boolean isMatch(char[] s, char[] p, int sIndex, int pIndex, Boolean[][] memory) {
//        if (pIndex == p.length) return sIndex == s.length;
//        if (sIndex == s.length) {
//            for (int i = pIndex; i < p.length; i++) {
//                if (p[i] != '*') return false;
//            }
//            return true;
//        }
//        if (memory[sIndex][pIndex] != null) return memory[sIndex][pIndex];
//        char sChar = s[sIndex];
//        char pChar = p[pIndex];
//        if (sChar == pChar || pChar == '?') {
//            memory[sIndex][pIndex] = isMatch(s, p, sIndex+1, pIndex+1, memory);
//        }
//        else if (pChar == '*') {
//            memory[sIndex][pIndex] = isMatch(s, p, sIndex+1, pIndex, memory)
//                    || isMatch(s, p, sIndex, pIndex+1, memory);
//        } else {
//            memory[sIndex][pIndex] = false;
//        }
//        return memory[sIndex][pIndex];
//    }


    public boolean isMatch(String s, String p) {
        return isMatch(s.toCharArray(), p.toCharArray(), 0, 0);
    }

    public boolean isMatch(char[] s, char[] p, int sIndex, int pIndex) {
        if (pIndex >= p.length) return sIndex == s.length;
        if (sIndex >= s.length) return false;
        char pChar = p[pIndex];
        if (pChar == '*') return isMatch(s, p, sIndex, pIndex + 1);
        if (sIndex == s.length) return p[p.length - 1] == '*';
        boolean nextCharWC = (pIndex + 1 < p.length && p[pIndex + 1] == '*');
        char sChar = s[sIndex];
        if (nextCharWC) {
            if (pChar == sChar || pChar == '?') {
                return isMatch(s, p, sIndex + 1, pIndex) || isMatch(s, p, sIndex, pIndex + 2);
            } else {
                return isMatch(s, p, sIndex, pIndex + 2);
            }
        } else {
            if (pChar == sChar || pChar == '?') return isMatch(s, p, sIndex + 1, pIndex + 1);
            else return false;
        }
    }


    public int removeElement(int[] nums, int val) {
        int index = 0;
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            if (num == val) {
                continue;
            } else {
                nums[index] = num;
                index++;
            }
        }
        return index;
    }


    public List<List<Integer>> subsets(int[] nums) {
        int n = nums.length;
        List<List<Integer>> output = new ArrayList<>();
        List<Integer> base = new ArrayList<>();
        output.add(base);
        subsets(output, new ArrayList<>(), nums, 0);
        return output;
    }

    public void subsets(List<List<Integer>> output, List<Integer> subOutput, int[] nums, int index) {
        if (index == nums.length) return;
        for (int i = index; i < nums.length; i++) {
            subOutput.add(nums[i]);
            output.add(new ArrayList<>(subOutput));
            subsets(output, subOutput, nums, i + 1);
            subOutput.remove(subOutput.size() - 1);
        }
    }

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        Set<List<Integer>> output = new HashSet<>();
        List<Integer> base = new ArrayList<>();
        output.add(base);
        subsetsWithDup(output, new ArrayList<>(), nums, 0);
        return new ArrayList<>(output);
    }

    public void subsetsWithDup(Set<List<Integer>> output, List<Integer> subOutput, int[] nums, int index) {
        if (index == nums.length) return;
        for (int i = index; i < nums.length; i++) {
            subOutput.add(nums[i]);
            output.add(new ArrayList<>(subOutput));
            subsetsWithDup(output, subOutput, nums, i + 1);
            subOutput.remove(subOutput.size() - 1);
        }
    }


    public Node connect(Node root) {
        if (root == null) return null;
        Deque<Node> dq = new ArrayDeque<>();
        dq.add(root);
        connect(dq);
        return root;
    }

    public void connect(Deque<Node> dq) {
        if (dq.isEmpty()) return;
        Deque<Node> nextDQ = new ArrayDeque<>();
        while (!dq.isEmpty()) {
            Node current = dq.pollFirst();
            if (current == null) continue;
            current.next = dq.peekFirst();
            if (current.left != null) nextDQ.addLast(current.left);
            if (current.right != null) nextDQ.addLast(current.right);
        }
        connect(nextDQ);
    }


    public int numDistinct(String s, String t) {
        int sLength = s.length();
        int tLength = t.length();
        Integer[][] dp = new Integer[sLength + 1][tLength + 1];
        return numDistinct(s, t, 0, 0, dp);
    }

    public int numDistinct(String s, String t, int sIndex, int tIndex, Integer[][] dp) {
        if (tIndex == t.length()) return dp[sIndex][tIndex] = 1;
        if (sIndex == s.length()) return dp[sIndex][tIndex] = 0;
        if (dp[sIndex][tIndex] != null) return dp[sIndex][tIndex];
        char sChar = s.charAt(sIndex);
        char tChar = t.charAt(tIndex);
        if (sChar == tChar) {
            return dp[sIndex][tIndex] = numDistinct(s, t, sIndex + 1, tIndex + 1, dp) + numDistinct(s, t, sIndex + 1, tIndex, dp);
        } else {
            return dp[sIndex][tIndex] = numDistinct(s, t, sIndex + 1, tIndex, dp);
        }
    }


    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[] dpArray = new int[n];
        int maxProfit = 0;
        if (n < 3) {
            maxProfit = Math.max(maxProfit, prices[n - 1] - prices[0]);
            return maxProfit;
        }
        int maxSell = prices[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            int price = prices[i];
            dpArray[i] = Math.max(maxSell - price, dpArray[i + 1]);
            maxSell = Math.max(maxSell, price);
        }
        int minBuy = prices[0];
        for (int i = 1; i <= n - 1; i++) {
            int price = prices[i];
            maxProfit = Math.max(maxProfit, price - minBuy + dpArray[i]);
            minBuy = Math.min(minBuy, price);
        }
        return maxProfit;
    }


    public int longestConsecutive(int[] nums) {
        int n = nums.length;
        if (n <= 1) return n;
        Arrays.sort(nums);
        int output = 0;
        int counter = 1;
        int expected = nums[0] + 1;
        for (int i = 1; i < n; i++) {
            int num = nums[i];
            if (num < expected) {
                continue;
            } else if (num == expected) {
                counter++;
            } else {
                output = Math.max(output, counter);
                counter = 1;
            }
            expected = num + 1;
        }
        output = Math.max(output, counter);
        return output;
    }


    public int candy(int[] ratings) {
        int n = ratings.length;
        if (n == 1) return 1;
        int output = 0;
        int[] candyArray = new int[n];
        for (int i = 1; i < n; i++) {
            if (ratings[i] > ratings[i - 1]) {
                candyArray[i] = candyArray[i - 1] + 1;
            } else {
                candyArray[i] = 1;
            }
        }
        for (int i = n - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1]) {
                candyArray[i] = Math.max(candyArray[i], candyArray[i + 1] + 1);
            }
        }
        for (int candy : candyArray) output += candy;
        return output;
    }


    public int singleNumber(int[] nums) {
        int n = nums.length;
        if (n < 4) return nums[0];
        Arrays.sort(nums);
        for (int i = 2; i < n; i = i + 3) {
            int a = nums[i - 2];
            int b = nums[i - 1];
            int c = nums[i];
            if (a != b || b != c) {
                int output = (a == b) ? c : (b == c) ? a : b;
                return output;
            }
        }
        return nums[n - 1];
    }


    public int maximumGap(int[] nums) {
        int n = nums.length;
        if (n <= 1) return 0;
        Arrays.sort(nums);
        int prior = nums[0];
        int diff = 0;
        for (int i = 1; i < n; i++) {
            int num = nums[i];
            diff = Math.max(diff, num - prior);
            prior = num;
        }
        return diff;
    }


    public int findDuplicate(int[] nums) {
        boolean[] b = new boolean[100000];
        for (int i : nums) {
            if (b[i] == true) return i;
            else b[i] = true;
        }
        return 2;
    }


    public String removeDuplicateLetters(String s) {
        Set<Character> set = new HashSet<>();
        for (char c : s.toCharArray()) set.add(c);
        char[] cArray = new char[set.size()];
        int index = 0;
        for (Character c : set) {
            cArray[index] = c;
            index++;
        }
        return String.valueOf(cArray);
    }

    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());
        for (int i : nums) {
            if (!pq.contains(i)) pq.add(i);
        }
        while (k > 1) {
            pq.poll();
            k--;
        }
        return pq.poll();
    }


    public boolean canMeasureWater(int x, int y, int target) {
        Boolean[][] memory = new Boolean[x + 1][y + 1];
        return canMeasureWater(0, 0, x, y, target, memory);
    }

    public boolean canMeasureWater(int x, int y, int xMax, int yMax, int target, Boolean[][] memory) {
        if (x == target || y == target || x + y == target) return true;
        if (memory[x][y] != null) return memory[x][y];
        memory[x][y] = false;
        boolean a = canMeasureWater(Math.min(xMax, x + y), 0, xMax, yMax, target, memory);
        boolean b = canMeasureWater(0, Math.min(yMax, x + y), xMax, yMax, target, memory);
        boolean c = canMeasureWater(0, y, xMax, yMax, target, memory);
        boolean d = canMeasureWater(x, 0, xMax, yMax, target, memory);
        boolean e = canMeasureWater(xMax, y, xMax, yMax, target, memory);
        boolean f = canMeasureWater(x, yMax, xMax, yMax, target, memory);
        memory[x][y] = a || b || c || d || e || f;
        return memory[x][y];
    }

    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        List<List<Integer>> output = new ArrayList<>();
        return kSmallestPairs(nums1, nums2, 0, 0, k, output);
    }

    public List<List<Integer>> kSmallestPairs(int[] a, int[] b, int aIndex, int bIndex, int k, List<List<Integer>> output) {
        if (output.size() == k) return output;
        List<Integer> list = new ArrayList<>();
        list.add(a[aIndex]);
        list.add(b[bIndex]);
        output.add(list);
        if (aIndex == a.length - 1 && bIndex == b.length - 1) return output;
        if (aIndex == a.length - 1) {
            aIndex = 0;
            bIndex++;
        } else if (bIndex == b.length - 1) {
            bIndex = 0;
            aIndex++;
        } else {
            int nextA = a[aIndex + 1];
            int nextB = b[bIndex + 1];
            if (nextA + b[bIndex] > a[aIndex] + nextB) {
                bIndex++;
            } else {
                aIndex++;
            }
        }
        return kSmallestPairs(a, b, aIndex, bIndex, k, output);
    }


    public int minMoves(int[] nums) {
        long max = 0;
        long sum = 0;
        for (int i : nums) {
            max = Math.max(max, i);
            sum += i;
        }
        return (int) (sum - max * nums.length);
    }


    public boolean predictTheWinner(int[] nums) {
        int n = nums.length;
        if (n <= 2) return true;
        Integer[][] dp = new Integer[n][n];
        boolean flag = false;
        predictTheWinner(nums, dp, 0, n - 1, 0, flag);
        return flag;
    }

    public void predictTheWinner(int[] nums, Integer[][] dp, int leftPointer, int rightPointer, int score, boolean flag) {
        if (leftPointer >= rightPointer) {
            if (score >= 0) flag = true;
            return;
        } else {
            if (dp[leftPointer][rightPointer] != null) {
                if (dp[leftPointer][rightPointer] >= 0) flag = true;
                return;
            }
            int left = nums[leftPointer];
            int right = nums[rightPointer];
            int leftScore = score + left;
            int rightScore = score + right;
            predictTheWinner(nums, dp, leftPointer + 1, rightPointer, leftScore, flag);
            predictTheWinner(nums, dp, leftPointer, rightPointer - 1, rightScore, flag);
            dp[leftPointer][rightPointer] = Math.max(leftScore, rightScore);
        }
    }


    public int arrangeCoins(int n) {
        int i = 1;
        int counter = 0;
        while (i <= n) {
            counter++;
            n = n - i;
            i++;
        }
        return counter;
    }

    HashMap<Integer, Integer> subtreeHM;

    public int[] findFrequentTreeSum(TreeNode root) {
        subtreeHM = new HashMap<>();
        if (root.left == null && root.right == null) {
            int[] output = {root.val};
            return output;
        } else {
            findFrequentTreeSum2(root);
            List<Integer> keyList = new ArrayList<>(subtreeHM.keySet());
            System.out.println(keyList);
            //Sort the list based on hashmap value
            Collections.sort(keyList, new Comparator<Integer>() {
                public int compare(Integer a, Integer b) {
                    return (subtreeHM.get(b) - subtreeHM.get(a));
                }
            });
            int maxFrequency = subtreeHM.get(keyList.get(0));
            List<Integer> output = new ArrayList<>();
            for (Integer key : keyList) {
                int currentFrequency = subtreeHM.get(key);
                if (currentFrequency == maxFrequency) output.add(key);
                else break;
            }
            return output.stream().mapToInt(i -> i).toArray();
        }
    }

    public Integer findFrequentTreeSum2(TreeNode root) {
        if (root.left == null && root.right == null) {
            subtreeHM.merge(root.val, 1, Integer::sum);
            return root.val;
        } else {
            Integer left = (root.left == null) ? 0 : findFrequentTreeSum2(root.left);
            Integer right = (root.right == null) ? 0 : findFrequentTreeSum2(root.right);
            Integer sum = left + right + root.val;
            subtreeHM.merge(sum, 1, Integer::sum);
            return sum;
        }
    }


    public boolean[] canEat(int[] candiesCount, int[][] queries) {
        int n = queries.length;
        int candies = candiesCount.length;
        boolean[] output = new boolean[n];
        long[] mustEat = new long[candies];
        mustEat[0] = candiesCount[0];
        for (int i = 1; i < candies; i++) {
            mustEat[i] += mustEat[i - 1] + candiesCount[i];
        }
        for (int i = 0; i < n; i++) {
            int favType = queries[i][0];
            int favDay = queries[i][1];
            long dailyCap = queries[i][2];
            long maxCandy = (favDay + 1) * dailyCap;
            long minCandy = (favType == 0) ? 0 : mustEat[favType - 1];
            output[i] = (maxCandy > minCandy) && ((favDay + 1) <= mustEat[favType]);
        }
        return output;
    }

    public int sumOfUnique(int[] nums) {
        Boolean[] flag = new Boolean[101];
        long sum = 0;
        for (int i : nums) {
            if (flag[i] == null) {
                flag[i] = true;
                sum += i;
            } else if (flag[i] == true) {
                flag[i] = false;
                sum -= i;
            } else {
                continue;
            }
        }
        return (int) sum;
    }


    public int maxOperations(int[] nums, int k) {
        int n = nums.length;
        if (k == 1 || n == 1) return 0;
        Arrays.sort(nums);
        int leftIndex = 0;
        int rightIndex = n - 1;
        int output = 0;
        while (leftIndex < rightIndex) {
            int left = nums[leftIndex];
            int right = nums[rightIndex];
            int currentSum = left + right;
            if (currentSum == k) {
                leftIndex++;
                rightIndex--;
                output++;
            } else if (currentSum < k) {
                leftIndex++;
            } else {
                rightIndex--;
            }
        }
        return output;
    }


    public boolean isEvenOddTree(TreeNode root) {
        if (root == null) return false;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int level = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            int prevValue = (level % 2 == 0) ? Integer.MIN_VALUE : Integer.MAX_VALUE;
            for (int i = 0; i < size; i++) {
                TreeNode current = queue.poll();
                int currentValue = current.val;
                if (level % 2 == 0) {
                    if (currentValue % 2 == 0 || currentValue <= prevValue) return false;
                } else {
                    if (currentValue % 2 == 1 || currentValue >= prevValue) return false;
                }
                prevValue = currentValue;
                if (current.left != null) queue.offer(current.left);
                if (current.right != null) queue.offer(current.right);
            }
            level++;
        }
        return true;
    }


    public int maximumUniqueSubarray(int[] nums) {
        int n = nums.length;
        int output = 0;
        HashMap<Integer, Integer> indexHM = new HashMap<>();
        int currentStartingIndex = 0;
        int currentSum = 0;
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            if (indexHM.containsKey(num) && indexHM.get(num) >= currentStartingIndex) {
                currentStartingIndex = indexHM.get(num) + 1;
                currentSum = 0;
                for (int j = currentStartingIndex; j <= i; j++) {
                    currentSum += nums[j];
                }
            } else {
                currentSum += num;
            }
            indexHM.put(num, i);
            output = Math.max(output, currentSum);
        }
        return output;
    }


    public int bestTeamScore(int[] scores, int[] ages) {
        int n = scores.length;
        int[][] players = new int[n][2];
        for (int i = 0; i < n; i++) {
            int score = scores[i];
            int age = ages[i];
            int[] tmp = {score, age};
            players[i] = tmp;
        }
        Arrays.sort(players, (a, b) -> {
            if (a[1] == b[1]) return a[0] - b[0];
            return a[1] - b[1];
        });
        int output = 0;
        for (int i = 0; i < n; i++) {
            int[] player = players[i];
            int currentScore = player[0];
            int scoreSum = currentScore;
            for (int j = 1; j < n; j++) {
                int[] nextPlayer = players[j];
                if (nextPlayer[0] <= currentScore) scoreSum += nextPlayer[0];
            }
            output = Math.max(output, scoreSum);
        }
        return output;
    }


    public int maxCoins(int[] piles) {
        Arrays.sort(piles);
        int n = piles.length;
        int steps = n / 3;
        int index = n - 2;
        int output = 0;
        while (steps != 0) {
            output += piles[index];
            index -= 2;
            steps--;
        }
        return output;
    }


    public int addedInteger(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        return nums2[0] - nums1[0];
    }


    public List<List<Integer>> threeSum(int[] nums) {
        int n = nums.length;
        List<List<Integer>> output = new ArrayList<>();
        if (n < 3) return output;
        Arrays.sort(nums);
        HashMap<Integer, Integer> hm = new HashMap<>();
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            hm.put(num, i);
        }
        for (int i = 0; i < n - 2; i++) {
            int first = nums[i];
            if (first > 0) break;
            if (i > 0 && first == nums[i - 1]) continue;
            for (int j = i + 1; j < n - 1; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;
                int second = nums[j];
                int third = (first + second) * -1;
                if (hm.containsKey(third) && hm.get(third) > j) {
                    output.add(Arrays.asList(first, second, third));
                }
            }
        }
        return output;
    }


    public static List<List<Integer>> fourSum(int[] nums, int target) {
        int n = nums.length;
        List<List<Integer>> output = new ArrayList<>();
        if (n < 4) return output;
        Arrays.sort(nums);

        for (int i = 0; i < n - 3; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;  // Skip duplicate numbers
            for (int j = i + 1; j < n - 2; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;  // Skip duplicate numbers
                int k = j + 1, l = n - 1;
                while (k < l) {
                    long sum = (long) nums[i] + nums[j] + nums[k] + nums[l];  // Use long to prevent overflow
                    if (sum == target) {
                        output.add(Arrays.asList(nums[i], nums[j], nums[k], nums[l]));
                        while (k < l && nums[k] == nums[k + 1]) k++;  // Skip duplicate numbers
                        while (k < l && nums[l] == nums[l - 1]) l--;  // Skip duplicate numbers
                        k++;
                        l--;
                    } else if (sum < target) {
                        k++;
                    } else {
                        l--;
                    }
                }
            }
        }
        return output;
    }


    public int threeSumClosest(int[] nums, int target) {
        int n = nums.length;
        Arrays.sort(nums);
        int difference = Integer.MAX_VALUE;
        int output = 0;
        for (int i = 0; i < n - 2; i++) {
            int first = nums[i];
            if (i > 0 && first == nums[i - 1]) continue;
            for (int j = i + 1; j < n - 1; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;
                int second = nums[j];
                for (int k = j + 1; k < n; k++) {
                    if (k > j + 1 && nums[k] == nums[k - 1]) continue;
                    int third = nums[k];
                    int currentSum = first + second + third;
                    int currentDifference = Math.abs(target - currentSum);
                    if (currentDifference < difference) {
                        output = currentSum;
                        difference = currentDifference;
                    }
                }
            }
        }
        return output;
    }


    public List<String> findRepeatedDnaSequences(String s) {
        int n = s.length();
        List<String> output = new ArrayList<>();
        if (n < 10) return output;
        HashMap<String, Integer> hm = new HashMap<>();
        for (int i = 0; i < n - 9; i++) {
            String sub = s.substring(i, i + 10);
            hm.merge(sub, 1, Integer::sum);
        }
        for (var entry : hm.entrySet()) {
            if (entry.getValue() > 1) output.add(entry.getKey());
        }
        return output;
    }


    public List<Integer> majorityElement(int[] nums) {
        List<Integer> output = new ArrayList<>();
        int n = nums.length;
        int threshold = n / 3;
        if (n < 3) {
            output.add(nums[0]);
            if (nums[0] != nums[1]) output.add(nums[1]);
            return output;
        }
        Arrays.sort(nums);
        for (int i = 0; i < n; i++) {
            int first = nums[i];
            if (i > 0 && first == nums[i - 1]) continue;
            if (i + threshold >= n) break;
            int second = nums[i + threshold];
            if (first == second) {
                output.add(first);
                i += threshold;
            }
        }
        return output;
    }


    int globalPathSum;

    public int pathSum(TreeNode root, int targetSum) {
        if (root == null) return 0;
        pathSumFromNode(root, targetSum, 0);
        pathSum(root.left, targetSum);
        pathSum(root.right, targetSum);
        return globalPathSum;
    }

    private void pathSumFromNode(TreeNode node, long targetSum, long currentSum) {
        if (node == null) return;
        currentSum += node.val;
        if (currentSum == targetSum) {
            globalPathSum++;
        }
        pathSumFromNode(node.left, targetSum, currentSum);
        pathSumFromNode(node.right, targetSum, currentSum);
    }


    public ListNode swapNodes(ListNode head, int k) {
        ListNode ln = head;
        List<ListNode> list = new ArrayList<>();
        while (ln != null) {
            list.add(ln);
            ln = ln.next;
        }
        int tmp = list.get(k - 1).val;
        list.get(k - 1).val = list.get(list.size() - k).val;
        list.get(list.size() - k).val = tmp;
        return head;
    }


    public int firstMissingPositive(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        int prev = 1;
        for (int i = 0; i < n; i++) {
            if (nums[i] <= 0) continue;
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            if (nums[i] == prev) {
                prev++;
            }
            return prev;
        }
        return prev;
    }


    public List<List<Integer>> getSkyline(int[][] buildings) {
        List<int[]> events = new ArrayList<>();
        for (int[] b : buildings) {
            events.add(new int[]{b[0], b[2], 1}); // start of building
            events.add(new int[]{b[1], b[2], -1}); // end of building
        }
        Collections.sort(events, (a, b) -> {
            if (a[0] != b[0]) return a[0] - b[0];
            return (a[2] * b[1]) - (b[2] * a[1]);
        });
        TreeMap<Integer, Integer> heightMap = new TreeMap<>(Collections.reverseOrder());
        heightMap.put(0, 1);
        List<List<Integer>> result = new ArrayList<>();
        int prevHeight = 0;
        for (int[] event : events) {
            int x = event[0], height = event[1], type = event[2];
            if (type == 1) {
                heightMap.put(height, heightMap.getOrDefault(height, 0) + 1);
            } else {
                heightMap.put(height, heightMap.get(height) - 1);
                if (heightMap.get(height) == 0) heightMap.remove(height);
            }
            int currentHeight = heightMap.firstKey();
            if (currentHeight != prevHeight) {
                result.add(Arrays.asList(x, currentHeight));
                prevHeight = currentHeight;
            }
        }
        List<List<Integer>> output = new ArrayList<>();
        for (List<Integer> skyline : result) if (skyline.get(1) != 0) output.add(skyline);
        return output;
    }




    public static String minWindow(String s, String t) {
        if (t.length() == 1) {
            if (s.contains(t)) return t;
            else return "";
        }
        char[] sChar = s.toCharArray();
        char[] tChar = t.toCharArray();
        HashMap<Character, Integer> frequencyMap = new HashMap<>();
        HashMap<Character, List<Integer>> indexMap = new HashMap<>();
        for (Character C : tChar) frequencyMap.merge(C, 1, Integer::sum);
        for (int i = 0; i < sChar.length; i++) {
            Character S = sChar[i];
            if (!indexMap.containsKey(S)) indexMap.put(S, new ArrayList<>());
            indexMap.get(S).add(i);
        }
        String output = "";
        try {
            while (true) {
                String current = "";
                int startIndex = Integer.MAX_VALUE;
                int endIndex = Integer.MIN_VALUE;
                Character toRemove = null;
                for (Character C : frequencyMap.keySet()) {
                    Integer frequencyRequired = frequencyMap.get(C) - 1;
                    List<Integer> list = indexMap.get(C);
                    Integer firstIndex = list.get(0);
                    Integer lastIndex = list.get(frequencyRequired);
                    if (firstIndex < startIndex) {
                        startIndex = firstIndex;
                        toRemove = C;
                    }
                    endIndex = Math.max(endIndex, lastIndex);
                }
                current = s.substring(startIndex, endIndex + 1);
                if (output.equals("") || current.length() < output.length()) output = current;
                indexMap.get(toRemove).remove(0);
            }
        } catch (Exception e) {
            return output;
        }
    }



    public int eraseOverlapIntervals(int[][] intervals) {
        //Sort by end time, first to last
        Arrays.sort(intervals, (a, b) -> a[1] - b[1]);
        int n = intervals.length;
        int overlapCounter = 0;
        for (int i = 1; i < n; i++) {
            int[] previousInterval = intervals[i-1];
            int[] currentInterval = intervals[i];
            if (currentInterval[0] < previousInterval[1]) {
                overlapCounter++;
                intervals[i] = previousInterval;
            }
        }
        return overlapCounter;
    }

    public class Point {
        int index;
        int sum;
        public Point(int index, int sum) {
            this.index = index;
            this.sum = sum;
        }
    }



    public boolean makesquare(int[] matchsticks) {
        Arrays.sort(matchsticks);
        int sum = 0;
        for (int i : matchsticks) sum += i;
        if (sum % 4 != 0) return false;
        int maxLength = sum / 4;
        if (matchsticks[matchsticks.length - 1] > maxLength) return false;
        int[] lengths = new int[4];
        Arrays.fill(lengths, maxLength);
        return makesquare(matchsticks, 0, lengths);
    }
    public boolean makesquare(int[] matchsticks, int matchIndex, int[] lengths) {
        int n = matchsticks.length;
        if (matchIndex == n) {
            for (int l : lengths) if (l != 0) return false;
            return true;
        }
        int currentMatch = matchsticks[matchIndex];
        for (int i = 0; i < 4; i++) {
            if (lengths[i] >= currentMatch) {
                lengths[i] -= currentMatch;
                if (makesquare(matchsticks, matchIndex + 1, lengths)) return true;
                lengths[i] += currentMatch;
            }
        }
        return false;
    }




    public static int[] findRedundantConnection(int[][] edges) {
        int n = edges.length;
        boolean[][] graph = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            boolean[] visited = new boolean[n];
            int[] edge = edges[i];
            int from = edge[0] - 1;
            int to = edge[1] - 1;
            if (findRedundantConnectionSearch(graph, from, to, visited)) return edge;
            graph[from][to] = true;
        }
        return edges[n-1];
    }

    public static boolean findRedundantConnectionSearch(boolean[][] graph, int from, int to, boolean[] visited) {
        if (graph[from][to]) return true;
        if (visited[from]) return false;
        boolean[] fromGraph = graph[from];
        visited[from] = true;
        for (int i = 0; i < graph.length; i++) {
            if (fromGraph[i] && !visited[i]) {
                boolean nextSearch = findRedundantConnectionSearch(graph, i, to, visited);
                if (nextSearch) return true;
            }
        }
        return false;
    }



    public int change(int amount, int[] coins) {
        Arrays.sort(coins);
        int n = coins.length;
        if (n == 0) return 0;
        if (coins.length == 1) {
            if (amount % coins[0] == 0) return 1;
            else return 0;
        }
        int[] dp = new int[amount+1];
        dp[0] = 1;
        for (int i = 0; i < n; i++) {
            int currentCoin = coins[i];
            for (int j = 0; j < amount+1; j++) {
                if (j + currentCoin > amount) break;
                if (dp[j] > 0) dp[j+currentCoin] += dp[j];
            }
        }
        return dp[amount];
    }



    public static void main(String[] args) {
        int[] a = {1, 1, 1, 1};
        int[][] g = {{1,2},{2,3},{3,4},{1,4},{1,5}};
        findRedundantConnection(g);
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


    static class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }

    class Trie {
        Node root;

        public Trie() {
            root = new Node();
        }

        public void insert(String word) {
            root.insert(word, 0);
        }

        public boolean search(String word) {
            return root.search(word, 0);
        }

        class Node {
            Node[] children;
            boolean eow;

            public Node() {
                children = new Node[26];
            }

            public void insert(String s, int index) {
                if (index == s.length()) {
                    return;
                } else {
                    char c = s.charAt(index);
                    int cIndex = c - 'a';
                    if (children[cIndex] == null) {
                        children[cIndex] = new Node();
                    }
                    children[cIndex].insert(s, index + 1);
                    if (index == s.length() - 1) children[cIndex].eow = true;
                }
            }

            public boolean search(String s, int index) {
                if (index == s.length()) return false;
                char c = s.charAt(index);
                int cIndex = c - 'a';
                Node n = children[cIndex];
                if (n == null) return false;
                if (index == s.length() - 1) return n.eow;
                return n.search(s, index + 1);
            }
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





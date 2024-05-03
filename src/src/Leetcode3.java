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
        for (int i : nums) if (i%2==0) evenSum += i;
        int[] output = new int[queries.length];
        for (int i = 0; i < queries.length; i++) {
            int[] query = queries[i];
            int addValue = query[0];
            int index = query[1];
            if (nums[index]%2==0) evenSum -= nums[index];
            nums[index] += addValue;
            if (nums[index]%2==0) evenSum += nums[index];
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
                    case 'R' -> Collections.rotate(pathLogic,-1);
                    case 'L' -> Collections.rotate(pathLogic, 1);
                }
            }
            if (x == 0 && y == 0) return true;
        }
        return (x == 0 && y == 0);
    }



    public int[] missingRolls(int[] rolls, int mean, int n) {
        int totalRolls = rolls.length + n;
        int upperBound = n*6;
        int lowerBound = n*1;
        int knownSum = 0;
        for (int i : rolls) knownSum += i;
        int missingSum = (mean * totalRolls) - knownSum;
        if (missingSum < lowerBound || upperBound < missingSum) return new int[0];
        int[] output = new int[n];
        for (int i = 0; i < n; i++) {
            if (i == n-1) {
                output[i] = missingSum;
            } else {
                output[i] = missingSum / (n-i);
                missingSum = missingSum - output[i];
            }
        }
        return output;
    }





    public int findJudge(int n, int[][] trust) {
        if (n == 1) return (trust.length==0) ? 1 : -1;
        if (trust.length < n-1) return -1;
        Set<Integer> visited = new HashSet<>();
        int[] trustArray = new int[n+1];
        for (int[] t : trust) {
            visited.add(t[0]);
            trustArray[t[1]]++;
        }
        System.out.println(Arrays.toString(trustArray));
        for (int i = 1; i < n+1; i++) {
            if (trustArray[i] == n-1 && !visited.contains(i)) return i;
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
        long currentMax = scoreArray[n-1];
        int maxNode = n-1;
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
















    public static void main(String[] args) {
        int[] a = {1,2,3};

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





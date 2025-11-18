import java.util.*;

public class Practice3 {

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


//    public int maxProfit(int[] prices) {
//        int n = prices.length;
//        int[] dpArray = new int[n];
//        int maxProfit = 0;
//        if (n < 3) {
//            maxProfit = Math.max(maxProfit, prices[n - 1] - prices[0]);
//            return maxProfit;
//        }
//        int maxSell = prices[n - 1];
//        for (int i = n - 2; i >= 0; i--) {
//            int price = prices[i];
//            dpArray[i] = Math.max(maxSell - price, dpArray[i + 1]);
//            maxSell = Math.max(maxSell, price);
//        }
//        int minBuy = prices[0];
//        for (int i = 1; i <= n - 1; i++) {
//            int price = prices[i];
//            maxProfit = Math.max(maxProfit, price - minBuy + dpArray[i]);
//            minBuy = Math.min(minBuy, price);
//        }
//        return maxProfit;
//    }


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
            int[] previousInterval = intervals[i - 1];
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
        return edges[n - 1];
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
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int i = 0; i < n; i++) {
            int currentCoin = coins[i];
            for (int j = 0; j < amount + 1; j++) {
                if (j + currentCoin > amount) break;
                if (dp[j] > 0) dp[j + currentCoin] += dp[j];
            }
        }
        return dp[amount];
    }


    public int lengthOfLongestSubsequence(List<Integer> nums, int target) {
        int n = nums.size();
        if (n == 1) {
            if (nums.get(0) == target) return 1;
            return -1;
        }
        Collections.sort(nums);
        int[] dp = new int[target + 1];
        int first = nums.get(0);
        if (first > target) return -1;
        dp[first] = 1;
        for (int i = 1; i < n; i++) {
            int num = nums.get(i);
            for (int j = target - num; j >= 0; j--) {
                if (j == 0) {
                    dp[j + num] = Math.max(dp[j + num], 1);
                } else {
                    if (dp[j] > 0) dp[j + num] = Math.max(dp[j + num], dp[j] + 1);
                }
            }
        }
        return (dp[target] == 0) ? -1 : dp[target];
    }


    public int maxValueOfCoins(List<List<Integer>> piles, int k) {
        int xMax = piles.size();
        int[][] dp = new int[xMax + 1][k + 1];
        for (int x = 0; x < xMax; x++) {
            List<Integer> pile = piles.get(x);
            for (int y = 0; y <= k; y++) {
                dp[x + 1][y] = dp[x][y];
            }
            int currentSum = 0;
            for (int c = 0; c < pile.size() && c < k; c++) {
                currentSum += pile.get(c);
                for (int y = k; y >= c + 1; y--) {
                    dp[x + 1][y] = Math.max(dp[x + 1][y], dp[x][y - (c + 1)] + currentSum);
                }
            }
        }
        return dp[xMax][k];
    }


    public int findLength(int[] nums1, int[] nums2) {
        int yMax = nums1.length;
        int xMax = nums2.length;
        int[][] dpMatrix = new int[yMax + 1][xMax + 1];
        int output = 0;
        for (int y = yMax - 1; y >= 0; y--) {
            for (int x = xMax - 1; x >= 0; x--) {
                if (nums1[y] == nums2[x]) {
                    dpMatrix[y][x] = dpMatrix[y + 1][x + 1] + 1;
                    output = Math.max(output, dpMatrix[x][y]);
                }
            }
        }
        for (int[] a : dpMatrix) System.out.println(Arrays.toString(a));
        return output;
    }


    public static int longestCommonSubsequence(String text1, String text2) {
        char[] a = text1.toCharArray();
        char[] b = text2.toCharArray();
        int yMax = text1.length();
        int xMax = text2.length();
        int[][] dpMatrix = new int[yMax + 1][xMax + 1];
        for (int y = yMax - 1; y >= 0; y--) {
            for (int x = xMax - 1; x >= 0; x--) {
                if (a[y] == b[x]) {
                    dpMatrix[y][x] = Math.max(dpMatrix[y + 1][x + 1] + 1, dpMatrix[y][x + 1]);
                } else {
                    dpMatrix[y][x] = Math.max(dpMatrix[y][x + 1], dpMatrix[y + 1][x]);
                }
            }
        }
        return dpMatrix[0][0];
    }


    public int findPairs(int[] nums, int k) {
        HashMap<Integer, Integer> hm = new HashMap<>();
        for (int i : nums) hm.merge(i, 1, Integer::sum);
        int output = 0;
        if (k == 0) {
            for (Integer key : hm.keySet()) if (hm.get(key) >= 2) output++;
            return output;
        }
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            if (set.contains(num)) continue;
            if (hm.containsKey(num + k)) output++;
            set.add(num);

        }
        return output;
    }


    public int lengthOfLIS(int[] nums) {
        TreeMap<Integer, Integer> tm = new TreeMap<>();
        for (int i : nums) {
            int currentLIS = 0;
            Integer lKey = tm.lowerKey(i);
            while (lKey != null) {
                currentLIS = Math.max(currentLIS, tm.get(lKey));
                lKey = tm.lowerKey(lKey);
            }
            tm.put(i, currentLIS + 1);
        }
        int output = 0;
        for (Integer I : tm.values()) output = Math.max(output, I);
        return output;
    }

    int maxDiff = 0;

    public int maxAncestorDiff(TreeNode root) {
        if (root == null) return maxDiff;
        if (root.left == null && root.right == null) return maxDiff;
        maxAncestorDiff(root.left, root.val, root.val);
        maxAncestorDiff(root.right, root.val, root.val);
        return maxDiff;
    }

    public void maxAncestorDiff(TreeNode root, int min, int max) {
        if (root == null) return;
        int value = root.val;
        maxDiff = Math.max(maxDiff, Math.max(Math.abs(value - min), Math.abs(value - max)));
        min = Math.min(min, value);
        max = Math.max(max, value);
        maxAncestorDiff(root.left, min, max);
        maxAncestorDiff(root.right, min, max);
    }


    public int minCostConnectPoints(int[][] points) {
        int n = points.length;
        if (n == 1) return 0;
        boolean[] searchedPoints = new boolean[n];
        //int[]: {Distance, Index of the point}
        //Priority based on smallest distance
        PriorityQueue<int[]> distanceIndexPQ = new PriorityQueue<>((a, b) -> a[0] - b[0]);
        //First point has no cost
        int[] first = {0, 0};
        distanceIndexPQ.add(first);
        //Minimum cost per point, array
        int[] minimumCost = new int[n];
        minimumCost[0] = 0;
        Arrays.fill(minimumCost, Integer.MAX_VALUE);
        int output = 0;
        while (!distanceIndexPQ.isEmpty()) {
            int[] currentQuery = distanceIndexPQ.poll();
            int index = currentQuery[1];
            int[] currentPoint = points[index];
            //If already in the MST
            if (searchedPoints[index]) continue;
            //This is already the minimum cost to connect
            int cost = currentQuery[0];
            output += cost;
            searchedPoints[index] = true;
            //Find the distance from this point to all other points, and take the minimum.
            for (int i = 0; i < n; i++) {
                if (searchedPoints[i]) continue;
                int[] queryPoint = points[i];
                int queryDistance = Math.abs(currentPoint[0] - queryPoint[0]) + Math.abs(currentPoint[1] - queryPoint[1]);
                if (queryDistance < minimumCost[i]) {
                    minimumCost[i] = queryDistance;
                    distanceIndexPQ.add(new int[]{queryDistance, i});
                }
            }
        }
        return output;
    }

    public boolean uniqueOccurrences(int[] arr) {
        HashMap<Integer, Long> frequencyMap = new HashMap<>();
        for (int i : arr) frequencyMap.merge(i, 1L, Long::sum);
        List<Long> list = new ArrayList<>(frequencyMap.values());
        Collections.sort(list);
        System.out.println(list);
        if (list.size() == 1) return true;
        for (int i = 1; i < list.size(); i++) {
            if (list.get(i) == list.get(i - 1)) return false;
        }
        return true;
    }


    public static List<List<Integer>> combinationSum22(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> output = new ArrayList<>();
        List<Integer> l = new ArrayList<>();
        int n = candidates.length;
        if (n == 1) {
            if (candidates[0] == target) {
                l.add(target);
                output.add(l);
            }
            return output;
        }
        combinationSum22(candidates, target, output, l, 0);
        return output;
    }

    public static void combinationSum22(int[] candidates, int target, List<List<Integer>> output, List<Integer> innerList, int index) {
        int n = candidates.length;
        Set<Integer> set = new HashSet<>();
        if (index == n) return;
        for (int i = index; i < n; i++) {
            int currentNumber = candidates[i];
            if (set.contains(currentNumber)) continue;
            set.add(currentNumber);
            if (target - currentNumber < 0) break;
            innerList.add(currentNumber);
            int newTarget = target - currentNumber;
            if (newTarget == 0) {
                List<Integer> l = new ArrayList<>();
                for (Integer I : innerList) l.add(I);
                if (!output.contains(l)) output.add(l);
            } else {
                combinationSum22(candidates, newTarget, output, innerList, i + 1);
            }
            innerList.remove(innerList.size() - 1);
        }
    }


    public int maximalNetworkRank(int n, int[][] roads) {
        if (n == 2) return roads.length;
        int maxRank = 0;
        boolean[][] isConnected = new boolean[n][n];
        int[] individualRank = new int[n];
        for (var road : roads) {
            int a = road[0];
            int b = road[1];
            isConnected[a][b] = true;
            isConnected[b][a] = true;
            individualRank[a]++;
            individualRank[b]++;
        }
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                int currentRank = individualRank[i] + individualRank[j];
                if (isConnected[i][j]) currentRank--;
                maxRank = Math.max(maxRank, currentRank);
            }
        }
        return maxRank;
    }


    public static List<Integer> findMinHeightTrees(int n, int[][] edges) {
        if (n == 1) return new ArrayList<>(Arrays.asList(0));
        int[] edgeCount = new int[n];
        HashMap<Integer, List<Integer>> hm = new HashMap<>();
        for (int[] edge : edges) {
            int a = edge[0];
            int b = edge[1];
            if (!hm.containsKey(a)) hm.put(a, new ArrayList<>());
            if (!hm.containsKey(b)) hm.put(b, new ArrayList<>());
            hm.get(a).add(b);
            hm.get(b).add(a);
        }
        boolean bfs = true;
        while (bfs) {
            Set<Map.Entry<Integer, List<Integer>>> entrySet = new HashSet<>(hm.entrySet());
            for (var entry : entrySet) {
                Integer key = entry.getKey();
                List<Integer> value = entry.getValue();
                if (value.size() == 1) {
                    hm.get(value.get(0)).remove(key);
                    hm.remove(key);
                }
            }
            bfs = hm.size() > 2;
        }
        return new ArrayList<>(hm.keySet());
    }


    public static int maxScoreWords(String[] words, char[] letters, int[] score) {
        int[] charFrequency = new int[26];
        for (char c : letters) charFrequency[c - 'a']++;
        int n = words.length;
        int[][] charWords = new int[n][26];
        for (int i = 0; i < n; i++) {
            char[] word = words[i].toCharArray();
            for (char c : word) charWords[i][c - 'a']++;
        }
        return maxScoreWords(charWords, charFrequency, score, 0);
    }

    private static int maxScoreWords(int[][] charWords, int[] charFrequency, int[] score, int index) {
        if (index == charWords.length) return 0;

        // Option 1: Skip the current word
        int skipWord = maxScoreWords(charWords, charFrequency, score, index + 1);

        // Option 2: Include the current word (if possible)
        int[] currentWord = charWords[index];
        boolean canFormWord = true;
        int wordScore = 0;
        for (int i = 0; i < 26; i++) {
            if (currentWord[i] > charFrequency[i]) {
                canFormWord = false;
                break;
            }
            wordScore += currentWord[i] * score[i];
        }
        int useWord = 0;
        if (canFormWord) {
            // Create a copy of charFrequency array to pass to the recursive call
            int[] newCharFrequency = charFrequency.clone();
            for (int i = 0; i < 26; i++) {
                newCharFrequency[i] -= currentWord[i];
            }
            useWord = wordScore + maxScoreWords(charWords, newCharFrequency, score, index + 1);
        }

        return Math.max(skipWord, useWord);
    }


    Integer[] dpPower;

    public int getKth(int lo, int hi, int k) {
        dpPower = new Integer[3000];
        dpPower[1] = 0;
        if (lo == hi && lo == 1) return 1;
        if (lo == hi) return getPowerNumber(lo);
        List<Integer> range = new ArrayList<>();
        for (int i = lo; i <= hi; i++) {
            range.add(i);
        }
        Collections.sort(range, (a, b) -> {
            if (getPowerNumber(a) == getPowerNumber(b)) return (a - b);
            else return (getPowerNumber(a) - getPowerNumber(b));
        });
        return range.get(k - 1);
    }

    public int getPowerNumber(int i) {
        if (dpPower[i] != null) return dpPower[i];
        boolean even = (i % 2 == 0);
        if (even) {
            dpPower[i] = getPowerNumber(i / 2) + 1;
        } else {
            dpPower[i] = getPowerNumber(i * 3 + 1) + 1;
        }
        return dpPower[i];
    }


    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[] dpArray = new int[n + 10];
        for (int i = n - 2; i >= 0; i--) {
            int price = prices[i];
            int profit = 0;
            for (int j = i + 1; j < n; j++) {
                int futurePrice = dpArray[j];
                System.out.println(price);
                System.out.println(futurePrice);
                int currentProfit = price - futurePrice + dpArray[j + 2];
                profit = Math.max(profit, currentProfit);
            }
            dpArray[i] = Math.max(dpArray[i + 1], profit);
        }
        return dpArray[0];
    }

    int[] childCount;
    int[] distanceSum;
    List<List<Integer>> graph;

    public int[] sumOfDistancesInTree(int n, int[][] edges) {
        //Build Graph
        graph = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList<>());
        }
        for (int[] edge : edges) {
            int a = edge[0];
            int b = edge[1];
            graph.get(a).add(b);
            graph.get(b).add(a);
        }
        //Child array, representing the number of child nodes, including itself
        childCount = new int[n];
        //Including self
        Arrays.fill(childCount, 1);
        //Result/Output array, representing the sum of all distances, at this node
        distanceSum = new int[n];
        //DFS from leaf to root
        //Fill in the childCount array
        //Fill in the sumDistance array, as the number of child
        sumDistanceDFS(-1, 0);
        //Reverse DFS from root to leaf
        reverseSumDistanceDFS(-1, 0);
        return distanceSum;
    }

    public void sumDistanceDFS(Integer parentNode, Integer currentNode) {
        System.out.println("ParentNode " + parentNode + " currentNode " + currentNode);
        for (Integer childNode : graph.get(currentNode)) {
            if (!Objects.equals(childNode, parentNode)) {
                sumDistanceDFS(currentNode, childNode);
                childCount[currentNode] += childCount[childNode];
                distanceSum[currentNode] += distanceSum[childNode] + childCount[childNode];
            }
        }
    }

    public void reverseSumDistanceDFS(Integer parentNode, Integer currentNode) {
        int n = childCount.length;
        for (Integer childNode : graph.get(currentNode)) {
            if (!Objects.equals(childNode, parentNode)) {
                int adjustedDistance = distanceSum[currentNode] - childCount[childNode] + n - childCount[childNode];
                distanceSum[childNode] = adjustedDistance;
                reverseSumDistanceDFS(currentNode, childNode);
            }
        }
    }


    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        if (n == 2) {
            int tmp = nums[0];
            nums[0] = nums[1];
            nums[1] = tmp;
            return nums;
        }
        int[] leftDP = new int[n];
        int[] rightDP = new int[n];
        leftDP[0] = nums[0];
        rightDP[n - 1] = nums[n - 1];
        for (int i = 1; i < n; i++) {
            leftDP[i] = leftDP[i - 1] * nums[i];
        }
        for (int i = n - 2; i >= 0; i--) {
            rightDP[i] = rightDP[i + 1] * nums[i];
        }
        nums[0] = rightDP[1];
        nums[n - 1] = leftDP[n - 2];
        for (int i = 1; i < n - 1; i++) {
            nums[i] = leftDP[i - 1] * rightDP[i + 1];
        }
        return nums;
    }


    public List<TreeNode> allPossibleFBT(int n) {
        Map<Integer, List<TreeNode>> memoryMap = new HashMap<>();
        return allPossibleFBT(n, memoryMap);
    }

    private List<TreeNode> allPossibleFBT(int n, Map<Integer, List<TreeNode>> memo) {
        if (memo.containsKey(n)) {
            return memo.get(n);
        }
        List<TreeNode> result = new ArrayList<>();
        if (n == 1) {
            result.add(new TreeNode(0));
            memo.put(n, result);
            return result;
        }
        if (n % 2 == 0) {
            return result;
        }
        for (int leftSize = 1; leftSize < n; leftSize += 2) {
            int rightSize = n - 1 - leftSize;
            List<TreeNode> leftTrees = allPossibleFBT(leftSize, memo);
            List<TreeNode> rightTrees = allPossibleFBT(rightSize, memo);
            for (TreeNode left : leftTrees) {
                for (TreeNode right : rightTrees) {
                    TreeNode root = new TreeNode(0);
                    root.left = left;
                    root.right = right;
                    result.add(root);
                }
            }
        }
        memo.put(n, result);
        return result;
    }


    public static int countRoutes(int[] locations, int start, int finish, int fuel) {
        int n = locations.length;
        int oldStart = locations[start];
        int oldFinish = locations[finish];
        Arrays.sort(locations);
        int newFinish = 0;
        int newStart = 0;
        for (int i : locations) {
            if (i == oldFinish) break;
            newFinish++;
        }
        for (int i : locations) {
            if (i == oldStart) break;
            newStart++;
        }
        Integer[][] dpMemory = new Integer[n][fuel + 1];
        Integer output = countRoutes(locations, newStart, newFinish, fuel, dpMemory);
        return output;
    }

    public static Integer countRoutes(int[] locations, int currentLocation, int target, int currentFuel, Integer[][] dpMemory) {
        int n = locations.length;
        if (dpMemory[currentLocation][currentFuel] != null) return dpMemory[currentLocation][currentFuel];
        Integer output = (currentLocation == target) ? 1 : 0;
        for (int i = 0; i < n; i++) {
            if (i == currentLocation) continue;
            int nextFuelCost = Math.abs(locations[currentLocation] - locations[i]);
            int nextFuel = currentFuel - nextFuelCost;
            if (nextFuel < 0) break;
            output += countRoutes(locations, i, target, nextFuel, dpMemory);
        }
        long mod = 1000000007;
        return dpMemory[currentLocation][currentFuel] = Math.toIntExact(output % mod);
    }


    public int[] rowAndMaximumOnes(int[][] mat) {
        int n = mat.length;
        int maxFrequency = Integer.MIN_VALUE;
        int maxIndex = -1;
        for (int i = n - 1; i >= 0; i--) {
            int[] ints = mat[i];
            int currentFrequency = 0;
            for (int currentInt : ints) if (currentInt == 1) currentFrequency++;
            if (currentFrequency >= maxFrequency) {
                maxFrequency = currentFrequency;
                maxIndex = i;
            }
        }
        return new int[]{maxIndex, maxFrequency};
    }

    public class DoubleKey {
        Integer index;
        Integer sum;
        public DoubleKey(Integer index, Integer sum) {
            this.index = index;
            this.sum = sum;
        }
    }









    public int distributeCookies(int[] cookies, int k) {
        int[] buckets = new int[k];
        return distributeCookies(cookies, buckets, 0);
    }
    public int distributeCookies(int[] cookies, int[] buckets, int index) {
        int n = cookies.length;
        if (index == n) {
            int max = -1;
            for (int i : buckets) max = Math.max(max, i);
            return max;
        }
        int currentCookie = cookies[index];
        int min = Integer.MAX_VALUE;
        int m = buckets.length;
        for (int i = 0; i < m; i++) {
            buckets[i] = buckets[i] + currentCookie;
            int next = distributeCookies(cookies, buckets, index+1);
            min = Math.min(min, next);
            buckets[i] = buckets[i] - currentCookie;
            if (buckets[i] == 0) break;
        }
        return min;
    }



    public int findTheCity(int n, int[][] edges, int distanceThreshold) {
        int[][] dpArray = new int[n][n];
        for (int[] row : dpArray) Arrays.fill(row, 100001);
        for (int i = 0; i < n; i++) dpArray[i][i] = 0;
        HashMap<Integer, List<int[]>> adjList = new HashMap<>();
        for (int i = 0; i < n; i++) {
            adjList.put(i, new ArrayList<>());
        }
        for (int[] edge : edges) {
            int a = edge[0];
            int b = edge[1];
            int c = edge[2];
            adjList.get(a).add(new int[]{b, c});
            adjList.get(b).add(new int[]{a, c});
        }
        for (int i = 0; i < n; i++) {
            PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
            pq.offer(new int[]{i, 0});
            boolean[] visited = new boolean[n];
            while (!pq.isEmpty()) {
                int[] current = pq.poll();
                int city = current[0];
                int distance = current[1];
                if (visited[city]) continue;
                visited[city] = true;
                for (int[] neighbor : adjList.get(city)) {
                    int nextCity = neighbor[0];
                    int nextDistance = neighbor[1] + distance;
                    if (nextDistance < dpArray[i][nextCity]) {
                        dpArray[i][nextCity] = nextDistance;
                        pq.offer(new int[]{nextCity, nextDistance});
                    }
                }
            }
        }
        int output = -1;
        int minReachable = Integer.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            int reachableCount = 0;
            for (int j = 0; j < n; j++) {
                if (i != j && dpArray[i][j] <= distanceThreshold) {
                    reachableCount++;
                }
            }
            if (reachableCount <= minReachable) {
                minReachable = reachableCount;
                output = i;
            }
        }
        return output;
    }






    public static int minHeightShelves(int[][] books, int shelfWidth) {
        int[] dp = new int[1000];
        Arrays.sort(books, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                int w = a[0];
                int x = a[1];
                int y = b[0];
                int z = b[1];
                if (x != z) return z - x;
                else return y - w;
            }
        });
        return minHeightShelves(books, 0, shelfWidth, dp);
    }
    public static int minHeightShelves(int[][] books, int index, int shelfWidth, int[] dp) {
        int n = books.length;
        if (index == n) return 0;
        int output = Integer.MAX_VALUE;
        int currentBookHeight = books[index][1];
        int currentBookWidth = books[index][0];
        System.out.println(currentBookHeight);
        System.out.println(currentBookWidth);
        for (int i = 0; i < dp.length; i++) {
            if (dp[i] + currentBookWidth > shelfWidth) continue;
            int current = (dp[i] == 0) ? currentBookHeight : 0;
            dp[i] += currentBookWidth;
            current += minHeightShelves(books, index+1, shelfWidth, dp);
            output = Math.min(output, current);
            dp[i] -= currentBookWidth;
            if (dp[i] == 0) break;
        }
        return output;
    }



    public int maxSumDivThree(int[] nums) {
        int[] dpModulo = new int[3];
        for (int i : nums) {
            int a = dpModulo[0] + i;
            int b = dpModulo[1] + i;
            int c = dpModulo[2] + i;
            dpModulo[a%3] = Math.max(dpModulo[a%3], a);
            dpModulo[b%3] = Math.max(dpModulo[b%3], b);
            dpModulo[c%3] = Math.max(dpModulo[c%3], c);
        }
        return dpModulo[0];
    }



    public long maxAlternatingSum(int[] nums) {
        int n = nums.length;
        if (n == 1) return  nums[0];
        long[] dpArray = new long[n+1];
        for (int i = n - 1; i >= 0; i--) {
            long num = nums[i];
            long currentMax = num;
            for (int j = i+1; j < n; j++) {
                long l = nums[j];
                long result = num - l + dpArray[j+1];
                if (result < 0 ) break;
                currentMax = Math.max(currentMax, result);
            }
            dpArray[i] = Math.max(dpArray[i+1], currentMax);
        }
        return dpArray[0];
    }






    public static double levenshteinDistance(String aString, String bString) {
        char[] a = aString.toLowerCase().toCharArray();
        char[] b = bString.toLowerCase().toCharArray();
        double[][] dpArray = new double[a.length + 1][b.length + 1];
        for (int i = 0; i <= a.length; i++) {
            for (int j = 0; j <= b.length; j++) {
                if (i == 0) {
                    dpArray[i][j] = j;
                } else if (j == 0) {
                    dpArray[i][j] = i;
                } else {
                    double substitutionCost = (a[i - 1] == b[j - 1] ? 0 : 1);
                    double substitution = dpArray[i - 1][j - 1] + substitutionCost;
                    double insertion = dpArray[i - 1][j] + 1;
                    double deletion = dpArray[i][j - 1] + 1;
                    dpArray[i][j] = Math.min(substitution, Math.min(insertion, deletion));
                }
            }
        }
        return dpArray[a.length][b.length];
    }



    public static double levenshteinDistance(String aString,
                                             String bString,
                                             double insertDeleteCost,
                                             double subCost) {
        char[] a = aString.toLowerCase().toCharArray();
        char[] b = bString.toLowerCase().toCharArray();
        double[][] dpArray = new double[a.length + 1][b.length + 1];
        for (int i = 0; i <= a.length; i++) {
            for (int j = 0; j <= b.length; j++) {
                if (i == 0) {
                    dpArray[i][j] = j;
                } else if (j == 0) {
                    dpArray[i][j] = i;
                } else {
                    double substitutionCost = (a[i - 1] == b[j - 1] ? 0 : subCost);
                    double substitution = dpArray[i - 1][j - 1] + substitutionCost;
                    double insertion = dpArray[i - 1][j] + insertDeleteCost;
                    double deletion = dpArray[i][j - 1] + insertDeleteCost;
                    dpArray[i][j] = Math.min(substitution, Math.min(insertion, deletion));
                }
            }
        }
        return dpArray[a.length][b.length];
    }








    public int makeArrayIncreasing(int[] arr1, int[] arr2) {
        Arrays.sort(arr2);
        int n = arr1.length;
        int[][] dp = new int[n][n + 1];
        for (int[] row : dp) Arrays.fill(row, Integer.MAX_VALUE);
        dp[0][0] = arr1[0];
        dp[0][1] = arr2[0];

        for (int i = 1; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (dp[i - 1][j] < arr1[i]) {
                    dp[i][j] = Math.min(dp[i][j], arr1[i]);
                }
                if (j > 0 && dp[i - 1][j - 1] < arr2[i]) {
                    dp[i][j] = Math.min(dp[i][j], arr2[j - 1]);
                }
            }
        }

        for (int i = 0; i <= n; i++) {
            if (dp[n - 1][i] < Integer.MAX_VALUE) return i;
        }
        return -1;
    }

    private int maxSum = 0;

    public int maxSumBST(TreeNode root) {
        postOrder(root);
        return maxSum;
    }

    private int[] postOrder(TreeNode node) {
        if (node == null) {
            return new int[] {1, Integer.MAX_VALUE, Integer.MIN_VALUE, 0}; // isBST, min, max, sum
        }
        int[] left = postOrder(node.left);
        int[] right = postOrder(node.right);
        if (left[0] == 1 && right[0] == 1 && node.val > left[2] && node.val < right[1]) {
            int sum = node.val + left[3] + right[3];
            maxSum = Math.max(maxSum, sum);
            int minVal = (node.left == null) ? node.val : left[1];
            int maxVal = (node.right == null) ? node.val : right[2];
            return new int[] {1, minVal, maxVal, sum};
        } else {
            return new int[] {0, 0, 0, 0};
        }
    }




    public int mincostTickets(int[] days, int[] costs) {
        int n = days.length;
        int firstDay = days[0];
        int oneDay = costs[0];
        int sevenDay = costs[1];
        int thirtyDay = costs[2];
        int[] dpArray = new int[365 + 1 + 30];
        int pointer = 365;
        for (int i = n - 1; i >= 0; i--) {
            int currentDay = days[i];
            while (currentDay < pointer) {
                pointer--;
                dpArray[pointer] = dpArray[pointer+1];
            }
            int costOne = oneDay + dpArray[currentDay + 1];
            int costSeven = sevenDay + dpArray[currentDay + 7];
            int costThirty = thirtyDay + dpArray[currentDay + 30];
            int minCost = Math.min(costOne, Math.min(costSeven, costThirty));
            dpArray[currentDay] = minCost;
        }
        return dpArray[firstDay];
    }




    public int maxResult(int[] nums, int k) {
        int n = nums.length;
        int[] dp = new int[n];
        Deque<Integer> deque = new LinkedList<>();
        dp[0] = nums[0];
        deque.addFirst(0);
        for (int i = 1; i < n; i++) {
            if (deque.peekFirst() < i - k) {
                deque.pollFirst();
            }
            dp[i] = nums[i] + dp[deque.peekFirst()];
            while (!deque.isEmpty() && dp[i] >= dp[deque.peekLast()]) {
                deque.pollLast();
            }
            deque.addLast(i);
        }
        return dp[n - 1];
    }




    public int kthFactor(int n, int k) {
        for (int i = 1; i <= n; i++) {
            int remainder = n % i;
            if (remainder == 0) {
                if (k == 1) return i;
                k--;
            }
        }
        return -1;
    }



    public int partitionString(String s) {
        if (s.length() == 1) return 1;
        return partitionString(s.toCharArray(), 0);
    }
    public int partitionString(char[] sCharArray, int index) {
        int n = sCharArray.length;
        int[] frequencyArray = new int[26];
        for (int i = index; i < n; i++) {
            int charIndex = sCharArray[i] - 'a';
            if (frequencyArray[charIndex] > 0) {
                return 1 + partitionString(sCharArray, i);
            }
            frequencyArray[charIndex]++;
        }
        return 1;
    }



    public static boolean wordBreak(String s, List<String> wordDict) {
        int n = s.length();
        Set<String> wordSet = Set.copyOf(wordDict);
        boolean[] dpArray = new boolean[n+1];
        dpArray[n] = true;
        for (int i = n - 1; i >= 0; i--) {
            for (int j = n; j > i; j--) {
                String subString = s.substring(i,j);
                if (wordSet.contains(subString)) {
                    if (dpArray[i+1]) {
                        dpArray[i] = true;
                        break;
                    }
                }
            }
        }
        System.out.println(Arrays.toString(dpArray));
        return dpArray[0];
    }



    public int minDistance(String word1, String word2) {
        char[] a = word1.toLowerCase().toCharArray();
        char[] b = word2.toLowerCase().toCharArray();
        int[][] dpArray = new int[a.length + 1][b.length + 1];
        for (int i = 0; i <= a.length; i++) {
            for (int j = 0; j <= b.length; j++) {
                if (i == 0) {
                    dpArray[i][j] = j;
                } else if (j == 0) {
                    dpArray[i][j] = i;
                } else {
                    int substitutionCost = (a[i - 1] == b[j - 1] ? 0 : 1);
                    int substitution = dpArray[i - 1][j - 1] + substitutionCost;
                    int insertion = dpArray[i - 1][j] + 1;
                    int deletion = dpArray[i][j - 1] + 1;
                    dpArray[i][j] = Math.min(substitution, Math.min(insertion, deletion));
                }
            }
        }
        return dpArray[a.length][b.length];
    }










    public static int maxProfit(int k, int[] prices) {
        int n = prices.length;
        if (n == 1) return 0;
        if (n == 2) {
            return Math.max(prices[1]-prices[0], 0);
        }
        int[][] dpReferenceMatrix = new int[n+1][n+1];
        for (int i = 0; i < n-1; i++) {
            int minPrice = prices[i];
            for (int j = i+1; j < n; j++) {
                minPrice = Math.min(minPrice, prices[j]);
                dpReferenceMatrix[i][j] = Math.max(dpReferenceMatrix[i][j-1], Math.max(dpReferenceMatrix[i][j], prices[j] - minPrice));
            }
        }
        int[][] dpMatrix = new int[k][n+1];
        for (int i = 0; i < n; i++) {
            dpMatrix[0][i] = dpReferenceMatrix[i][n-1];
        }
        //i is the level
        for (int i = 1; i < k; i++) {
            int currentProfit = 0;
            //j is the start
            for (int j = n - 2; j >= 0; j--) {
                int currentLevelProfit = 0;
                //l is the end
                for (int l = n - 1; l > j; l--) {
                    int currentSingleProfit = dpReferenceMatrix[j][l];
                    int priorLevelProfit = dpMatrix[i-1][l+1];
                    int currentLevelBoundedProfit = currentSingleProfit + priorLevelProfit;
                    currentLevelProfit = Math.max(currentLevelProfit, currentLevelBoundedProfit);
                }
                dpMatrix[i][j] = Math.max(dpMatrix[i][j+1], currentLevelProfit);
            }
        }
        return dpMatrix[k-1][0];
    }


    public int shipWithinDays(int[] weights, int days) {
        int maxWeight = 0;
        int sumWeight = 0;
        for (int weight : weights) {
            maxWeight = Math.max(maxWeight, weight);
            sumWeight += weight;
        }

        int left = maxWeight;
        int right = sumWeight;

        while (left < right) {
            int middle = (left + right) / 2;
            int currentWeight = 0;
            int daysRequired = 1;

            for (int weight : weights) {
                if (currentWeight + weight > middle) {
                    daysRequired++;
                    currentWeight = 0;
                }
                currentWeight += weight;
            }

            if (daysRequired > days) {
                left = middle + 1;
            } else {
                right = middle;
            }
        }

        return left;
    }


    public int splitArray(int[] nums, int k) {
        int maxWeight = 0;
        int sumWeight = 0;
        for (int weight : nums) {
            maxWeight = Math.max(maxWeight, weight);
            sumWeight += weight;
        }

        int left = maxWeight;
        int right = sumWeight;

        while (left < right) {
            int middle = (left + right) / 2;
            int currentWeight = 0;
            int daysRequired = 1;

            for (int weight : nums) {
                if (currentWeight + weight > middle) {
                    daysRequired++;
                    currentWeight = 0;
                }
                currentWeight += weight;
            }

            if (daysRequired > k) {
                left = middle + 1;
            } else {
                right = middle;
            }
        }

        return left;
    }




    public boolean canCross(int[] stones) {
        int n = stones.length;
        if (n == 2) return (stones[1] == 1);
        Set<Integer> stoneSet = new HashSet<>();
        for (int stone : stones) stoneSet.add(stone);
        HashMap<Integer, Set<Integer>> hm = new HashMap<>();
        int index = n-2;
        while (index > 0) {
            int lastJump = stones[n-1] - stones[index];
            boolean output = canCross(stoneSet, hm, lastJump, stones[n-1]);
            if (output) return true;
        }
        return false;
    }
    public boolean canCross(Set<Integer> stoneSet, HashMap<Integer, Set<Integer>> hm, int lastJump, int currentPos) {
        if (currentPos == 0) {
            if (lastJump == 1) return true;
            return false;
        }
        if (lastJump <= 0) return false;
        if (currentPos < 0) return false;
        if (!stoneSet.contains(currentPos)) return false;
        if (!hm.containsKey(lastJump)) hm.put(lastJump, new HashSet<>());
        if (hm.get(lastJump).contains(currentPos)) return false;
        hm.get(lastJump).add(currentPos);
        if (canCross(stoneSet, hm, lastJump-1, currentPos - (lastJump-1))) return true;
        if (canCross(stoneSet, hm, lastJump, currentPos - (lastJump))) return true;
        if (canCross(stoneSet, hm, lastJump+1, currentPos - (lastJump+1))) return true;
        return false;
    }

    public int findNumberOfLIS(int[] nums) {
        int n = nums.length;
        if (n <= 1) return n;
        int[] lengths = new int[n];
        int[] counts = new int[n];
        Arrays.fill(counts, 1);
        int maxLISLength = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    if (lengths[j] + 1 > lengths[i]) {
                        lengths[i] = lengths[j] + 1;
                        counts[i] = counts[j]; // inherit count from counts[j]
                    } else if (lengths[j] + 1 == lengths[i]) {
                        counts[i] += counts[j];
                    }
                }
            }
            maxLISLength = Math.max(maxLISLength, lengths[i]);
        }
        int result = 0;
        for (int i = 0; i < n; i++) {
            if (lengths[i] == maxLISLength) {
                result += counts[i];
            }
        }
        return result;
    }





    public int minRefuelStops(int target, int startFuel, int[][] stations) {
        int n = stations.length;
        long[] dp = new long[n + 1];
        dp[0] = startFuel;
        for (int i = 0; i < n; i++) {
            for (int t = i; t >= 0; t--) {
                if (dp[t] >= stations[i][0]) {
                    dp[t + 1] = Math.max(dp[t + 1], dp[t] + stations[i][1]);
                }
            }
        }
        for (int i = 0; i <= n; i++) {
            if (dp[i] >= target) {
                return i;
            }
        }
        return -1;
    }

    int targetSumCounter;
    public int findTargetSumWays(int[] nums, int target) {
        targetSumCounter = 0;
        findTargetSumWays(nums, target, 0, 0);
        return targetSumCounter;
    }
    public void findTargetSumWays(int[] nums, int target, int index, int currentSum) {
        int n = nums.length;
        if (index == n) {
            if (currentSum == target) targetSumCounter++;
            return;
        }
        int currentNumber = nums[index];
        findTargetSumWays(nums, target, index+1, currentSum + currentNumber);
        findTargetSumWays(nums, target, index+1, currentSum - currentNumber);
    }


    public int maxProfit(int[] prices, int fee) {
        int n = prices.length;
        if (n == 1) {
            return 0;
        }
        int cash = 0;
        int hold = -prices[0];
        for (int i = 1; i < n; i++) {
            cash = Math.max(cash, hold + prices[i] - fee);
            hold = Math.max(hold, cash - prices[i]);
        }

        return cash;
    }

    public int orderOfLargestPlusSign(int n, int[][] mines) {
        int[][] grid = new int[n][n];
        for (int[] row : grid) Arrays.fill(row, n);
        for (int[] mine : mines) {
            grid[mine[0]][mine[1]] = 0;
        }
        int[][] left = new int[n][n];
        int[][] right = new int[n][n];
        int[][] up = new int[n][n];
        int[][] down = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) continue;
                left[i][j] = (j > 0 ? left[i][j - 1] : 0) + 1;
                up[i][j] = (i > 0 ? up[i - 1][j] : 0) + 1;
            }
        }
        for (int i = n - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                if (grid[i][j] == 0) continue;
                right[i][j] = (j < n - 1 ? right[i][j + 1] : 0) + 1;
                down[i][j] = (i < n - 1 ? down[i + 1][j] : 0) + 1;
            }
        }
        int maxOrder = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) continue;
                int order = Math.min(Math.min(left[i][j], right[i][j]), Math.min(up[i][j], down[i][j]));
                maxOrder = Math.max(maxOrder, order);
            }
        }
        return maxOrder;
    }


    public boolean isMatch(String s, String p) {
        return isMatch(s.toCharArray(), p.toCharArray(), 0, 0);
    }
    private boolean isMatch(char[] s, char[] p, int sIndex, int pIndex) {
        if (pIndex == p.length) {
            return sIndex == s.length;
        }
        boolean firstMatch = (sIndex < s.length && (p[pIndex] == s[sIndex] || p[pIndex] == '.'));
        if (pIndex + 1 < p.length && p[pIndex + 1] == '*') {
            return (isMatch(s, p, sIndex, pIndex + 2) ||
                    (firstMatch && isMatch(s, p, sIndex + 1, pIndex)));
        } else {
            return firstMatch && isMatch(s, p, sIndex + 1, pIndex + 1);
        }
    }

    public int minInsertions(String s) {
        int n = s.length();
        if (n == 1) return 0;
        if (n == 2) {
            if (s.charAt(0) == s.charAt(1)) return 0;
            return 1;
        }
        int[][] dpMatrix = new int[n+1][n+1];
        char[] sChar = s.toCharArray();

        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                boolean flag = sChar[i] == sChar[j];
                if (flag) {
                    if (j == 0) dpMatrix[i][j] = 0;
                    else {
                        dpMatrix[i][j] = dpMatrix[i+1][j-1];
                    }
                } else {
                    if (j == 0) dpMatrix[i][j] = dpMatrix[i+1][j] + 1;
                    else {
                        dpMatrix[i][j] = Math.min(dpMatrix[i + 1][j], dpMatrix[i][j - 1]) + 1;
                    }
                }
            }
        }
        return dpMatrix[0][n-1];
    }




    public int maxHeight(int[][] cuboids) {
        int n = cuboids.length;
        for (int[] cube : cuboids) {
            Arrays.sort(cube);
        }
        Arrays.sort(cuboids, (a, b) -> {
            if (a[0] != b[0]) return a[0] - b[0];
            if (a[1] != b[1]) return a[1] - b[1];
            return a[2] - b[2];
        });

        int[] dpArray = new int[n+1];
        for (int i = 0; i < n; i++) {
            dpArray[i] = cuboids[i][2];
        }
        int output = 0;
        for (int i = n - 1; i >= 0; i--) {
            int[] currentCube = cuboids[i];
            int a = currentCube[0];
            int b = currentCube[1];
            int c = currentCube[2];
            for (int j = i + 1; j < n; j++) {
                int[] nextCube = cuboids[j];
                int aa = nextCube[0];
                int bb = nextCube[1];
                int cc = nextCube[2];
                if (a <= aa && b <= bb && c <= cc) {
                    dpArray[i] = Math.max(dpArray[i], c + dpArray[j]);
                }
            }
            output = Math.max(output, dpArray[i]);
        }
        return output;
    }



    public int lenLongestFibSubseq(int[] arr) {
        int n = arr.length;
        int maxLen = 0;
        Map<Integer, Integer> indexMap = new HashMap<>();
        for (int i = 0; i < n; i++) {
            indexMap.put(arr[i], i);
        }
        Map<Integer, Integer> dp = new HashMap<>();
        for (int j = 1; j < n; j++) {
            for (int i = 0; i < j; i++) {
                int k = indexMap.getOrDefault(arr[j] - arr[i], -1);
                if (k >= 0 && k < i) {
                    int key = i * n + j;
                    int length = dp.getOrDefault(k * n + i, 2) + 1;
                    dp.put(key, length);
                    maxLen = Math.max(maxLen, length);
                }
            }
        }
        return maxLen >= 3 ? maxLen : 0;
    }



    public int deleteAndEarn(int[] nums) {
        HashMap<Integer, Integer> hm = new HashMap<>();
        for (int i : nums) hm.merge(i, 1, Integer::sum);
        hm.put(Integer.MAX_VALUE, 0);
        List<Integer> keyList = new ArrayList<>(hm.keySet());
        Collections.sort(keyList);
        for (int i = keyList.size() - 2; i >= 0; i--) {
            int key = keyList.get(i);
            System.out.println(key);
            int frequency = hm.get(key);
            int sum = key * frequency;
            int upperKey = keyList.get(i+1);
            if (upperKey == key + 1) upperKey = keyList.get((i+2));
            sum += hm.get(upperKey);
            hm.put(key, Math.max(sum, hm.get(key+1)));
        }
        return (Collections.max(hm.values()));
    }




    public static String stoneGameIII(int[] stoneValue) {
        int n = stoneValue.length;
        int[] nextTurnIndex = new int[n+10];
        nextTurnIndex[n] = n;
        int[] dpMaxScore = new int[n+10];
        for (int i = n - 1; i >= 0; i--) {
            int stonesTaken = 0;
            int maxScore = Integer.MIN_VALUE;
            int stoneScore = 0;
            for (int j = 0; j < 3; j++) {
                if (i + j == n) break;
                stoneScore += stoneValue[i+j];
                int currentScore = stoneScore + dpMaxScore[nextTurnIndex[i+j+1]];
                if (currentScore > maxScore) {
                    maxScore = currentScore;
                    stonesTaken = j+1;
                }
            }
            nextTurnIndex[i] = i + stonesTaken;
            dpMaxScore[i] = maxScore;
        }
        int aScore = dpMaxScore[0];
        int bScore = dpMaxScore[nextTurnIndex[0]];
        System.out.println(Arrays.toString(nextTurnIndex));
        System.out.println(Arrays.toString(dpMaxScore));
        if (aScore > bScore) return "Alice";
        else if (aScore < bScore) return "Bob";
        else return "Tie";
    }



    public int numDecodings(String s) {
        int n = s.length();
        long mod = 1000000007;
        if (n == 0) return 0;
        long[] dpArray = new long[n + 1];
        dpArray[n] = 1;
        for (int i = n - 1; i >= 0; i--) {
            if (s.charAt(i) == '0') {
                dpArray[i] = 0;
            } else if (s.charAt(i) == '*') {
                dpArray[i] = 9 * dpArray[i + 1] % mod;
                if (i + 1 < n) {
                    if (s.charAt(i + 1) == '*') {
                        dpArray[i] = (dpArray[i] + 15 * dpArray[i + 2]) % mod;
                    } else if (s.charAt(i + 1) <= '6') {
                        dpArray[i] = (dpArray[i] + 2 * dpArray[i + 2]) % mod;
                    } else {
                        dpArray[i] = (dpArray[i] + dpArray[i + 2]) % mod;
                    }
                }
            } else {
                dpArray[i] = dpArray[i + 1];
                if (i + 1 < n) {
                    if (s.charAt(i + 1) == '*') {
                        if (s.charAt(i) == '1') {
                            dpArray[i] = (dpArray[i] + 9 * dpArray[i + 2]) % mod;
                        } else if (s.charAt(i) == '2') {
                            dpArray[i] = (dpArray[i] + 6 * dpArray[i + 2]) % mod;
                        }
                    } else {
                        int num = (s.charAt(i) - '0') * 10 + (s.charAt(i + 1) - '0');
                        if (num <= 26) {
                            dpArray[i] = (dpArray[i] + dpArray[i + 2]) % mod;
                        }
                    }
                }
            }
        }
        return (int) dpArray[0];
    }




    public int longestSubsequence(int[] arr, int difference) {
        HashMap<Integer, Integer> hm = new HashMap<>();
        for (int i : arr) {
            int prior = i + difference;
            int currentLongest = 1;
            if (hm.containsKey(prior)) {
                currentLongest = hm.get(prior) + 1;
            }
            hm.merge(i, currentLongest, Integer::max);
        }
        System.out.println(hm.values());
        return (Collections.max(hm.values()));
    }


    public int subarraysDivByK(int[] nums, int k) {
        HashMap<Integer, Integer> moduloFrequencyMap = new HashMap<>();
        moduloFrequencyMap.put(0, 1);
        int prefixSum = 0;
        int count = 0;
        for (int i : nums) {
            prefixSum += i;
            int mod = ((prefixSum % k) + k) % k;
            count += moduloFrequencyMap.getOrDefault(mod, 0);
            moduloFrequencyMap.put(mod, moduloFrequencyMap.getOrDefault(mod, 0) + 1);
        }
        return count;
    }


    public static int formingMagicSquare(List<List<Integer>> s) {
        // Define all possible 3x3 magic squares
        int[][][] magicSquares = {
                {{2, 7, 6}, {9, 5, 1}, {4, 3, 8}},
                {{4, 9, 2}, {3, 5, 7}, {8, 1, 6}},
                {{8, 3, 4}, {1, 5, 9}, {6, 7, 2}},
                {{6, 1, 8}, {7, 5, 3}, {2, 9, 4}},
                {{6, 7, 2}, {1, 5, 9}, {8, 3, 4}},
                {{2, 9, 4}, {7, 5, 3}, {6, 1, 8}},
                {{4, 3, 8}, {9, 5, 1}, {2, 7, 6}},
                {{8, 1, 6}, {3, 5, 7}, {4, 9, 2}}
        };
        int output = Integer.MAX_VALUE;
        for (int i = 0; i < magicSquares.length; i++) {
            int currentScore = 0;
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    currentScore += Math.abs(magicSquares[i][j][k] - s.get(j).get(k));
                }
            }
            output = Math.min(output,currentScore);
        }
        return output;
    }

    public static List<Integer> climbingLeaderboard(List<Integer> ranked, List<Integer> player) {
        List<Integer> uniqueRanked = new ArrayList<>(new HashSet<>(ranked));
        Collections.sort(uniqueRanked, Collections.reverseOrder());
        List<Integer> output = new ArrayList<>();
        int index = uniqueRanked.size() - 1;
        for (int score : player) {
            while (index >= 0 && score >= uniqueRanked.get(index)) {
                index--;
            }
            output.add(index + 2);
        }
        return output;
    }

    public static long getWays(int n, List<Long> c) {
        long[] dpArray = new long[n + 1];
        dpArray[0] = 1;
        for (long coin : c) {
            for (int j = (int) coin; j <= n; j++) {
                dpArray[j] += dpArray[j - (int) coin];
            }
        }
        return dpArray[n];
    }

    public static int equal(List<Integer> arr) {
        int n = arr.size();
        int min = 0;
        int sum = arr.get(0);
        if (n != 1) {
            sum = 0;
            Collections.sort(arr);
            min = arr.get(0);
            for (int i = 1; i < n; i++) {
                 int current = arr.get(i);
                 sum += current - min;
            }
        }
        int output = 0;
        output += sum / 5;
        sum = sum % 5;
        output += sum / 2;
        sum = sum % 2;
        output += sum;
        return output;
    }




    public static void main(String[] args) {

        boolean b = wordBreak("leetcode", Arrays.asList(new String[]{"leet", "code"}));

        int[] prices = {4,5,0,-2,-3,1};

        System.out.println('*'-'a');
        System.out.println(-15%5);



//        long mod = 1000000007;
//
//        int[] a = {1, 1, 1, 1};
//        int[][] points = {{2, 2}, {0, 0}, {3, 10}, {5, 2}, {7, 0}};
////        minCostConnectPoints(points);
//
//        int[][] edges = {{3, 0}, {3, 1}, {3, 2}, {3, 4}, {5, 4}};
//        findMinHeightTrees(6, edges);
//
//        String[] words = {"dog", "cat", "dad", "good"};
//        char[] letters = {'a', 'a', 'c', 'd', 'd', 'd', 'g', 'o', 'o'};
//        int[] score = {1, 0, 9, 5, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
//        int res = maxScoreWords(words, letters, score);
//
//
//        int[] b = {2, 3, 6, 8, 4};
//        int cr = countRoutes(b, 1, 3, 5);
//
//        int[][] books = {{1,1},{2,3},{2,3},{1,1},{1,1},{1,1},{1,2}};
//        int fewji = minHeightShelves(books,4);

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





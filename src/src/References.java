
import java.util.*;
class Seed {
    int x;
    int y;
    static int totalSeedCount = 0;
    List<Seed> next = new ArrayList<>();
    public Seed(int x, int y) {
        this.x = x;
        this.y = y;
        totalSeedCount++;
    }
    @Override
    public String toString() {
        return "Seed{" +
                "x=" + x +
                ", y=" + y;
    }
}

public class References {



    //SHORTCUTS

    //itar
    //ritar
    //itco
    //Max and min
    //mx
    //mn

    //Last element
    //lst

    public static void main(String[] args) {

        int[][] prices = {{1,4,2},{2,2,7},{2,1,3},{3,2,10},{1,4,2},{4,1,3}};
        Arrays.sort(prices, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                double first = (double) a[2] / (a[0] * a[1]);
                double second = (double) b[2] / (b[0] * b[1]);
                if (second > first) return 1;
                else if (first > second) return -1;
                else return 0;
            }
        });


        Arrays.sort(prices, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                int a1 = a[0];
                int a2 = a[1];
                int b1 = b[0];
                int b2 = b[1];
                if (a1 != b1) return (a1-b1);
                return (a2-b2);
            }
        });


        for (int[] intArr : prices) {
            System.out.println(Arrays.toString(intArr));
        }


        //Int[] to deque
//        Deque<Integer> queue = new ArrayDeque<>(Arrays.stream(nums).boxed().toList());


        //Int array sum
//        int sum = Arrays.stream(nums).boxed().reduce(0, (a, b) -> a + b);



        //TreeMap
        //Ceiling key : Returns the least key greater than or equal to the given key, or null if there is no such key.
        //Floor key : Returns the greatest key less than or equal to the given key, or null if there is no such key.




        /*
            Static: Used when you dont care about the instance
            e.g. Seed class above
            Then totalSeedCount will return total of all seeds created
            Also e.g.
            Seed s = new Seed(a, b);
            Seed.toString() will not work because it is not calling toString on an instance,
            it is calling on a class.
            To make it work, add a static toString method in the class


         */


        //Map merge
//          Increment hashmap value by 1 if it exists, if not, set it to 1
        HashMap<Integer, Integer> map = new HashMap<>();
        HashMap<Integer, List<Integer>> indexMap = new HashMap<>();
        Integer key = 3;
        Integer value = 3;
        map.put(1, value);


        map.merge(key, 1, Integer::sum);

        indexMap.merge(1, new ArrayList<>(List.of(1)), (a, b) -> {a.addAll(b); return a;});




//          Sort based on hashmap values
        //List of the keys
        List<Integer> keyList = new ArrayList<>(map.keySet());
        //Sort the list based on hashmap value
        Collections.sort(keyList, new Comparator<Integer>() {
            public int compare(Integer a, Integer b) {
                return (map.get(b) - map.get(a));
            }
        });



        //Get hashmap key with largest value
//        Key key = Collections.max(map.entrySet(), Map.Entry.comparingByValue()).getKey();

        //Max value in hashmap
        int maxValueInMap = (Collections.max(map.values()));


//          Sort based on string length. Lambda shortened expression of comparator
//        Collections.sort(words, (a, b) -> {
//            if (a.length() != b.length()) {
//                return a.length() - b.length(); // Sort by length
//            }
//            return a.compareTo(b); // If lengths are the same, sort lexicographically
//        });


//        Char to int
        String n = "123";
        int i = 0;
        int x = n.charAt(i) - '0';


//        Map char to an int
//        a = 0, b = 1,...,z = 25
//        A = -32, B = -31,...,Z = -7
        n = "abc";
        x = n.charAt(0) - 'a';
        System.out.println(x);


        //Int to binary
//        Integer.toBinaryString(int i)

//        PriorityQueue
//        Insertion order is not retained, the order of the queue is based on the specified comparator
//        From smallest to biggest:
        PriorityQueue<Integer> PQ = new PriorityQueue<>();
//        From biggest to smallest:
        PriorityQueue<Integer> PQ2 = new PriorityQueue<>(Collections.reverseOrder());

        PriorityQueue<Integer> queue = new PriorityQueue<>((a,b) -> (a-b));

        PriorityQueue<int[]> pq = new PriorityQueue<>(
                Comparator.comparingInt((int[] a) -> a[1])
                        .thenComparingInt(a -> a[0])
        );



//        Condensed for loop for strings
        String s = "abcdefg";
        for (char c : s.toCharArray()) {
        }


//        Double int arrays
        int[] a = {1, 2, 3};
        int[] b = {4, 5, 6};
        int[][] doubleArray = {a, b};
//        doubleArray[0][0] will be 1
//        doubleArray[1][0] will be 4
//        doubleArray[y][x]
//        doubleArray.length will be 2

        //Sub array
        int[] subArray = Arrays.copyOfRange(a, 0, 0);


//        Instantiating List using Arrays.asList()
        List<Integer> list = Arrays.asList(1, 2, 3);



        //Int[] to list
        // Create a mutable list instead of an immutable one.
        List<Integer> listt = new ArrayList<>(Arrays.stream(a).boxed().toList());

        //List to int array
//        int[] output = list.stream().mapToInt(i->i).toArray();


        //Sort by reverse order
        Collections.sort(listt, Collections.reverseOrder());

        //Sort a set
//        Set<Integer> keySet = treeMapKey.keySet();
//        List<Integer> listSet = new ArrayList<>(keySet);
//        Collections.sort(listSet, Collections.reverseOrder());


//        Finding the middle of a linkedlist, use tortoise and hare method.







    }

}


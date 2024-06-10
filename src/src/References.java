
import com.google.gson.*;
import org.w3c.dom.*;

import javax.xml.parsers.*;
import javax.xml.transform.*;
import javax.xml.transform.dom.*;
import javax.xml.transform.stream.*;
import java.io.*;
import java.util.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

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

enum Key {
    RESET, UP, DOWN, RIGHT, LEFT, S, L;
}


public class References {

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
        // Section 1: Sorting Examples
        sortExamples();

        // Section 2: Integer and Array Conversions
        integerAndArrayConversions();

        // Section 3: HashMap Operations
        hashMapOperations();

        // Section 4: Queue Examples
        queueExamples();

        // Section 5: Miscellaneous Examples
        miscellaneousExamples();

        // Section 6: Reading and Writing
        readWrite();
    }

    // Section 1: Sorting Examples
    public static void sortExamples() {
        System.out.println("=== Sorting Examples ===");

        int[][] prices = {{1, 4, 2}, {2, 2, 7}, {2, 1, 3}, {3, 2, 10}, {1, 4, 2}, {4, 1, 3}};

        // Sort by a custom comparator
        Arrays.sort(prices, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                double first = (double) a[2] / (a[0] * a[1]);
                double second = (double) b[2] / (b[0] * b[1]);
                return Double.compare(second, first);
            }
        });

        // Sort by first then by second element
        Arrays.sort(prices, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                if (a[0] != b[0]) return Integer.compare(a[0], b[0]);
                return Integer.compare(a[1], b[1]);
            }
        });

        // Print sorted arrays
        for (int[] intArr : prices) {
            System.out.println(Arrays.toString(intArr));
        }
    }

    // Section 2: Integer and Array Conversions
    public static void integerAndArrayConversions() {
        System.out.println("=== Integer and Array Conversions ===");

        // Char to int
        String n = "123";
        int i = 0;
        int x = n.charAt(i) - '0';
        System.out.println("Char to int: " + x);

        // Map char to an int (a = 0, b = 1, ..., z = 25)
        n = "abc";
        x = n.charAt(0) - 'a';
        System.out.println("Char to int (a=0): " + x);

        // Int[] to List
        int[] a = {1, 2, 3};
        List<Integer> list = new ArrayList<>(Arrays.stream(a).boxed().toList());

        // List to int[]
        int[] output = list.stream().mapToInt(Integer::intValue).toArray();

        //int to binary string
        String binaryString = Integer.toBinaryString(i);

    }

    // Section 3: HashMap Operations
    public static void hashMapOperations() {
        System.out.println("=== HashMap Operations ===");

        // Increment HashMap value by 1 if it exists, if not, set it to 1
        HashMap<Integer, Integer> map = new HashMap<>();
        map.merge(3, 1, Integer::sum);

        // HashMap with List values
        HashMap<Integer, List<Integer>> indexMap = new HashMap<>();
        indexMap.merge(1, new ArrayList<>(List.of(1)), (a, b) -> {
            a.addAll(b);
            return a;
        });

        // Sort HashMap keys based on values
        List<Integer> keyList = new ArrayList<>(map.keySet());
        Collections.sort(keyList, (a, b) -> Integer.compare(map.get(b), map.get(a)));

        // Get HashMap key with largest value
        Integer maxValueInMap = Collections.max(map.values());
        Integer keyWithMaxValue = Collections.max(map.entrySet(), Map.Entry.comparingByValue()).getKey();


    }

    // Section 4: Queue Examples
    public static void queueExamples() {
        System.out.println("=== Queue Examples ===");
        int[] a = new int[100];

        //int[] to deque
        Deque<Integer> queue = new ArrayDeque<>(Arrays.stream(a).boxed().toList());


        // PriorityQueue from smallest to biggest
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        pq.add(3);
        pq.add(1);
        pq.add(2);
        System.out.println("PriorityQueue (smallest to biggest): " + pq);

        // PriorityQueue from biggest to smallest
        PriorityQueue<Integer> pq2 = new PriorityQueue<>(Collections.reverseOrder());
        pq2.add(3);
        pq2.add(1);
        pq2.add(2);
        System.out.println("PriorityQueue (biggest to smallest): " + pq2);

        // PriorityQueue with custom comparator
        PriorityQueue<int[]> pq3 = new PriorityQueue<>(Comparator.comparingInt((int[] z) -> z[1]).thenComparingInt(z -> z[0]));
        pq3.add(new int[]{1, 3});
        pq3.add(new int[]{2, 2});
        pq3.add(new int[]{3, 1});
        while (!pq3.isEmpty()) {
            System.out.println("PriorityQueue with custom comparator: " + Arrays.toString(pq3.poll()));
        }
    }

    // Section 5: Miscellaneous Examples
    public static void miscellaneousExamples() {
        System.out.println("=== Miscellaneous Examples ===");

        // Sub array
        int[] a = {1, 2, 3};
        int[] subArray = Arrays.copyOfRange(a, 0, 2);
        System.out.println("Sub array: " + Arrays.toString(subArray));

        // Instantiating List using Arrays.asList()
        List<Integer> list = Arrays.asList(1, 2, 3);

        // Find middle of a LinkedList (Tortoise and Hare method)
        LinkedList<Integer> linkedList = new LinkedList<>(List.of(1, 2, 3, 4, 5));
        findMiddleOfLinkedList(linkedList);
    }

    public static void findMiddleOfLinkedList(LinkedList<Integer> list) {
        System.out.println("=== Finding Middle of LinkedList ===");

        ListIterator<Integer> slow = list.listIterator();
        ListIterator<Integer> fast = list.listIterator();

        while (fast.hasNext()) {
            fast.next();
            if (fast.hasNext()) {
                fast.next();
                slow.next();
            }
        }

    }




    public static void readWrite() {
        System.out.println("=== ReadWrite ===");
        System.out.println(System.getProperty("user.dir"));
        //Reading XML Triples example
        List<List<String>> readTriple = readTripleXML("/Users/leong/Desktop/Personal Projects/self-learning-programming/src/src/res/triples.xml");
        //Reading XML readwrite example
        List<List<String>> readXML = readXML("/Users/leong/Desktop/Personal Projects/self-learning-programming/src/src/res/readwrite.xml");
        //Writing XML readwrite example
        List<String> keyNames = Arrays.asList("DOWN", "DOWN", "S", "RESET", "RIGHT", "DOWN", "RESET", "UP");
        List<String> states = Arrays.asList("LieProneState", "LieProneState", "ShootState", "StandState", "RunState",  "RunState",  "StandState",  "StandState");
        writeXML("xmlWriteFile.xml", keyNames, states);

        //Read md
        String mdString = readMD("src/src/res/data.md");
        //Write md
        writeMD("writeMD.md", mdString);


        //JSON
        Company company = new Company("TechCorp");
        company.addEmployee(new Person("Alice", 30, "Engineer"));
        company.addEmployee(new Person("Bob", 40, "Manager"));

        File writeFile = new File("/Users/leong/Desktop/Personal Projects/self-learning-programming/src/src/res/techcorp.JSON");
        writeJson(writeFile, company);
        File readFile = new File("/Users/leong/Desktop/Personal Projects/self-learning-programming/src/src/res/techcorp.JSON");
        Company techcorp = readJsonCompany(readFile);
        System.out.println(techcorp.getCompanyName());
    }


    public static List<List<String>>  readTripleXML(String filePath) {
        try {
            File file = new File(filePath);
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            DocumentBuilder builder = factory.newDocumentBuilder();
            Document document = builder.parse(file);
            document.getDocumentElement().normalize();

            System.out.println("Root element: " + document.getDocumentElement().getNodeName());
            NodeList nodeList = document.getElementsByTagName("triple");
            System.out.println("nodelist size");
            System.out.println(nodeList.getLength());

            List<List<String>> output = new ArrayList<>();
            List<String> subjectString = new ArrayList<>();
            List<String> predicateString = new ArrayList<>();
            List<String> objectString = new ArrayList<>();

            int n = nodeList.getLength();
            for (int i = 0; i < n; i++) {
                Node node = nodeList.item(i);
                if (node.getNodeType() == Node.ELEMENT_NODE) {
                    Element element = (Element) node;
                    String s = element.getElementsByTagName("subject").item(0).getTextContent();
                    String p = element.getElementsByTagName("predicate").item(0).getTextContent();
                    String o = element.getElementsByTagName("object").item(0).getTextContent();
                    subjectString.add(s);
                    predicateString.add(p);
                    objectString.add(o);
                }
            }
            output.add(subjectString);
            output.add(predicateString);
            output.add(objectString);
            return output;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    static String STATE_ROOT_ELEMENT = "States";

    public static List<List<String>> readXML(String fileName) {
        File f = new File(fileName);
        if (!f.exists()) {
            return Collections.emptyList();
        }
        DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
        DocumentBuilder db;
        try {
            db = dbf.newDocumentBuilder();
            Document doc = db.parse(f);
            doc.getDocumentElement().normalize();
            // ########## YOUR CODE STARTS HERE ##########
            // HINT: You can use getChildNodes() function in the XML library to obtain a
            // list of child nodes of the parent tag STATE_ROOT_ELEMENT.
            List<String> keys = new ArrayList<>();
            List<String> states = new ArrayList<>();

            NodeList nodeList = doc.getElementsByTagName(STATE_ROOT_ELEMENT);
            NodeList childList = nodeList.item(0).getChildNodes();
            for (int i = 0; i < childList.getLength(); i++) {
                if (childList.item(i) instanceof Element) {
                    Element e = (Element) childList.item(i);
                    String tagName = e.getTagName();
                    String text = e.getTextContent();
                    keys.add(tagName);
                    states.add(text);
                }
            }
            List<List<String>> result = new ArrayList<>();
            result.add(keys);
            result.add(states);
            return result;
            // ########## YOUR CODE ENDS HERE ##########
        } catch (Exception e) {
            e.printStackTrace();
        }

        return Collections.emptyList();
    }

    public static void writeXML(String fileName, List<String> keys, List<String> states) {
        File f = new File(fileName);
        if (f.exists()) {
            f.delete();
        }
        DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
        try {
            DocumentBuilder db = dbf.newDocumentBuilder();
            Document doc = db.newDocument();
            // ########## YOUR CODE STARTS HERE ##########
            Element rootElement = doc.createElement(STATE_ROOT_ELEMENT);
            doc.appendChild(rootElement);

            for (int i = 0; i < keys.size(); i++) {
                Element stateElement = doc.createElement(keys.get(i));
                stateElement.appendChild(doc.createTextNode(states.get(i)));
                rootElement.appendChild(stateElement);
            }
            // ########## YOUR CODE ENDS HERE ##########

            Transformer transformer = TransformerFactory.newInstance().newTransformer();
            transformer.setOutputProperty(OutputKeys.ENCODING, "utf-8");
            transformer.setOutputProperty(OutputKeys.INDENT, "yes");
            DOMSource source = new DOMSource(doc);
            StreamResult result = new StreamResult(f);
            transformer.transform(source, result);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }




    public static String readMD(String filePath)  {
        try {
            String content = readMDFile(filePath);
            return content;
        } catch (IOException e) {
            return null;
        }
    }
    public static String readMDFile(String filePath) throws IOException {
        StringBuilder content = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                content.append(line).append("\n");
            }
        }
        return content.toString();
    }

    public static void writeMD(String fileName, String content) {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))) {
            bw.write(content);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static void writeJson(File file, Object object) {
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        try (FileWriter writer = new FileWriter(file)) {
            gson.toJson(object, writer);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static Company readJsonCompany(File file) {
        Gson gson = new Gson();
        try (FileReader reader = new FileReader(file)) {
            return gson.fromJson(reader, Company.class);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }







}




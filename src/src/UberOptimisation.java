import java.util.*;
import java.util.stream.*;

public class UberOptimisation {

    HashMap<String, Double> output = new HashMap<>();

    HashMap<Integer, Integer> weekendQuest = new HashMap<>();

    {
        weekendQuest.put(15, 30);
        weekendQuest.put(25, 30);
        weekendQuest.put(35, 35);
        weekendQuest.put(45, 40);
        weekendQuest.put(60, 50);
    }

    List<Integer> friQuest = new ArrayList<>(Arrays.asList(9,11,12,13));
    List<Integer> satQuest = new ArrayList<>(Arrays.asList(9,11,12,13));
    List<Integer> sunQuest = new ArrayList<>(Arrays.asList(9,11,12,13));



    public static void main(String[] args) {
        UberOptimisation uo = new UberOptimisation();

        uo.friday(0,0);



        HashMap<String, Double> sortedOutput = uo.output.entrySet()
                .stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        Map.Entry::getValue,
                        (e1, e2) -> e1, // Handle duplicate keys (if any)
                        LinkedHashMap::new
                ));

        System.out.println("Sorted HashMap (Descending):");
        sortedOutput.forEach((key, value) -> System.out.println(key + ": " + value));
    }







    public void friday(int tripCount, int bonusCount) {
        saturday("Fri: 0, ", tripCount, bonusCount);
        int friCount = 0;
        for (int i : friQuest) {
            bonusCount += i;
            tripCount += 3;
            friCount += 3;
            String nextS = "Fri: " + friCount + ", ";
            saturday(nextS, tripCount, bonusCount);
        }
    }

    public void saturday(String s, int tripCount, int bonusCount) {
        sunday(s + "Sat: 0, ", tripCount, bonusCount);
        int satCount = 0;
        for (int i : satQuest) {
            bonusCount += i;
            tripCount += 3;
            satCount += 3;
            String nextS = s + "Sat: " + satCount + ", ";
            sunday(nextS, tripCount, bonusCount);
        }
    }

    public void sunday(String s, int tripCount, int bonusCount) {
        weekend(s + "Sun: 0, ", tripCount, bonusCount);
        int sunCount = 0;
        for (int i : sunQuest) {
            bonusCount += i;
            tripCount += 3;
            sunCount += 3;
            String nextS = s + "Sun: " + sunCount + ", ";
            weekend(nextS, tripCount, bonusCount);
        }
    }

    public void weekend(String s, int tripCount, int bonusCount) {
        for (var entry : weekendQuest.entrySet()) {
            if (tripCount >= entry.getKey()) bonusCount += entry.getValue();
        }
        double totalBonus = bonusCount;
        double totalTrips = tripCount;
        s += "Total: " + tripCount;
        output.put(s, (totalBonus / totalTrips));
    }












}

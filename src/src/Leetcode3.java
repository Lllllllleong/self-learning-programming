import java.util.*;

public class Leetcode3 {

    public List<Integer> diffWaysToCompute(String expression) {
        int n = expression.length();
        List<Integer> l = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            char c = expression.charAt(i);
            if (c == '*' || c == '+' || c == '-') {
                List<Integer> leftCombinations = diffWaysToCompute(expression.substring(0,i));
                List<Integer> rightCombinations = diffWaysToCompute(expression.substring(i+1));
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



}

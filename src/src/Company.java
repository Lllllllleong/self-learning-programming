import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Company {

    final String companyName;
    int workForce;
    List<Person> employees;

    // Constructor
    public Company(String companyName) {
        this.companyName = companyName;
        this.workForce = 0;
        this.employees = new ArrayList<>();
    }

    // Getters and Setters
    public String getCompanyName() {
        return companyName;
    }

    public int getWorkForce() {
        return workForce;
    }

    public List<Person> getEmployees() {
        return employees;
    }

    public void setWorkForce(int workForce) {
        this.workForce = workForce;
    }

    // Methods
    public void addEmployee(Person person) {
        employees.add(person);
        workForce++;
    }

    public void removeEmployee(Person person) {
        if (employees.remove(person)) {
            workForce--;
        }
    }

    public void listEmployees() {
        for (Person person : employees) {
            System.out.println("Name: " + person.getName() + ", Age: " + person.getAge() + ", Position: " + person.getPosition());
        }
    }




}
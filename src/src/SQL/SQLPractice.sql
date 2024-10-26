--
-- SQL Practice
--


SELECT
    customer_id
FROM
    Customers
WHERE
    year = 2021 AND
    revenue > 0;

WITH
    address2 AS (
        SELECT
            personId,
            city,
            state
        FROM
            Address
    )
SELECT
    firstName,
    lastName,
    city,
    state
FROM
    Person p LEFT JOIN address2 a ON p.personId = a.personId;

-- Q1
SELECT
    COUNT(*) AS r_restricted_usa_movies
FROM
    restriction
WHERE
    LOWER(description) = 'r'
    AND LOWER(country) = 'usa';


-- Q2
WITH
    filtered_people AS (SELECT
                            id
                        FROM
                            person
                        WHERE
                            year_born > 1959)
SELECT
    COUNT(DISTINCT filtered_people.id) AS num_writers_born_after_1960_inclusive
FROM
    filtered_people INNER JOIN writer ON filtered_people.id = writer.id;


-- Q3
SELECT
    country,
    COUNT(*) AS num_restriction_categories
FROM
    restriction_category
GROUP BY
    country
ORDER BY
    num_restriction_categories ASC;


-- Q4
WITH
    action_movies AS (SELECT
                          title,
                          production_year
                      FROM
                          movie
                      WHERE
                          LOWER(major_genre) = 'action')
SELECT
    COUNT(id) AS num_directors_without_action_major_genre
FROM (
    SELECT
        id
    FROM
        director
    EXCEPT
        SELECT
            id
        FROM
            director NATURAL JOIN action_movies) AS sub_query;


-- Q5
WITH
    aus_action_movies AS (SELECT
                              COUNT(*) AS num_aus_action_movies
                          FROM
                              movie
                          WHERE
                              LOWER(country) = 'australia'
                              AND LOWER(major_genre) = 'action'),
    aus_movies AS (SELECT
                       COUNT(*) AS num_aus_movies
                   FROM
                       movie
                   WHERE
                       LOWER(country) = 'australia')
SELECT
    CASE
        WHEN num_aus_movies = 0 THEN 0
        ELSE ROUND((num_aus_action_movies * 1.0) / num_aus_movies, 2)
    END AS proportion_of_action_movies_from_australia
FROM
    aus_action_movies,
    aus_movies;


-- Q6
WITH
    filtered_movies AS (SELECT
                            title,
                            production_year,
                            COUNT(*) AS num_crew_awards
                        FROM
                            crew_award
                        WHERE
                            LOWER(result) = 'won'
                        GROUP BY
                            title,
                            production_year)
SELECT
    title,
    production_year
FROM
    filtered_movies
WHERE
    num_crew_awards = (SELECT
                           MAX(num_crew_awards)
                       FROM
                           filtered_movies);


-- Q7
SELECT
    COUNT(*) AS num_movies_with_at_least_one_award
FROM (
    SELECT
        title,
        production_year
    FROM
        movie INTERSECT (SELECT
                             title,
                             production_year
                         FROM
                             movie_award
                         WHERE
                             LOWER(result) = 'won'
                         UNION
                         SELECT
                              title,
                              production_year
                         FROM
                              crew_award
                         WHERE
                              LOWER(result) = 'won'
                         UNION
                         SELECT
                             title,
                             production_year
                         FROM
                             director_award
                         WHERE
                             LOWER(result) = 'won'
                         UNION
                         SELECT
                             title,
                             production_year
                         FROM
                             writer_award
                         WHERE
                             LOWER(result) = 'won'
                         UNION
                         SELECT
                             title,
                             production_year
                         FROM
                             actor_award
                         WHERE
                             LOWER(result) = 'won')) AS sub_query;


-- Q8
WITH
    solo_written_movies AS (SELECT
                                title,
                                production_year
                            FROM
                                writer
                            GROUP BY
                                title,
                                production_year
                            HAVING COUNT(id) = 1),
    solo_writers AS (SELECT
                         id
                     FROM
                         writer NATURAL JOIN solo_written_movies),
    co_writers AS (SELECT
                       id
                   FROM
                       writer
                   EXCEPT
                       SELECT
                           id
                       FROM
                           solo_writers)
SELECT
    id,
    first_name,
    last_name
FROM
    co_writers NATURAL JOIN person
ORDER BY
    last_name DESC;


-- Q9
WITH
    filtered_movies AS (SELECT
                             title,
                             production_year,
                             year_of_award
                         FROM
                             movie_award
                         WHERE
                             LOWER(result) = 'won'
                         UNION
                         SELECT
                              title,
                              production_year,
                              year_of_award
                         FROM
                              crew_award
                         WHERE
                              LOWER(result) = 'won'
                         UNION
                         SELECT
                             title,
                             production_year,
                             year_of_award
                         FROM
                             director_award
                         WHERE
                             LOWER(result) = 'won'
                         UNION
                         SELECT
                             title,
                             production_year,
                             year_of_award
                         FROM
                             writer_award
                         WHERE
                             LOWER(result) = 'won'
                         UNION
                         SELECT
                             title,
                             production_year,
                             year_of_award
                         FROM
                             actor_award
                         WHERE
                             LOWER(result) = 'won')
SELECT DISTINCT
    CONCAT(fm1.title, ', ', fm1.production_year) AS a1,
    CONCAT(fm2.title, ', ', fm2.production_year) AS a2
FROM
    filtered_movies fm1 INNER JOIN filtered_movies fm2 ON fm1.year_of_award = fm2.year_of_award
WHERE
    (fm1.title, fm1.production_year) < (fm2.title, fm2.production_year);


WITH
    filteredCustomers AS (
        SELECT * FROM Customers
        where id NOT IN (SELECT customerID FROM Orders)
    )
SELECT
    name AS Customers
FROM
    filteredCustomers;




SELECT
    project_id,
    ROUND(AVG(experience_years), 2) as average_years
FROM
    Project p LEFT JOIN Employee e ON p.employee_id = e.employee_id
GROUP BY
    project_id;


SELECT
    p.product_id,
    IFNULL(ROUND(SUM(units*price)/SUM(units),2),0) AS average_price
FROM
    Prices p LEFT JOIN UnitsSold u ON p.product_id = u.product_id
AND
    u.purchase_date BETWEEN start_date AND end_date
GROUP BY
    product_id;

SELECT
    *
FROM
    cinema
WHERE
    mod(id, 2) = 1 AND description != 'boring'
ORDER BY
    rating DESC;

SELECT
    a.student_id,
    a.student_name,
    b.subject_name,
    IFNULL(c.attended_exams, 0) as attended_exams
FROM
    Students a CROSS JOIN Subjects b
LEFT JOIN (
    SELECT
        student_id,
        subject_name,
        COUNT(*) AS attended_exams
    FROM
        Examinations
    GROUP BY
        student_id,
        subject_name)
AS c ON a.student_id = c.student_id AND b.subject_name = c.subject_name
ORDER BY
    a.student_id,
    b.subject_name;



SELECT
    a.employee_id
FROM
    Employees a LEFT JOIN Employees b ON a.manager_id = b.employee_id
WHERE
    a.salary < 30000 AND a.manager_id IS NOT NULL AND b.employee_id IS NULL
ORDER BY
   a.employee_id;



SELECT
    x,
    y,
    z,
    CASE
        WHEN x + y > z AND x + z > y AND y + z > x THEN 'Yes'
        ELSE 'No'
    END AS 'triangle'
FROM
    triangle;


(SELECT
    employee_id,
    department_id
FROM
    Employee
WHERE
    primary_flag = 'Y')
UNION
(SELECT
    employee_id,
    department_id
FROM
    Employee
GROUP BY
    employee_id
HAVING
    COUNT(employee_id) = 1);

SELECT
    b.employee_id,
    b.name,
    COUNT(a.employee_id) AS reports_count,
    ROUND(AVG(a.age)) AS average_age
FROM
    Employees a JOIN Employees b ON a.reports_to = b.employee_id
GROUP BY
    employee_id
ORDER BY
    employee_id;


SELECT
    MAX(num) as num
FROM
(SELECT
    num
FROM
    MyNumbers
GROUP BY
    num
HAVING
    COUNT(num) = 1) as t;

SELECT
    user_id,
    COUNT(follower_id) as followers_count
FROM
    Followers
GROUP BY
    user_id
ORDER BY
    user_id ASC;


SELECT
    class
FROM
    Courses
GROUP BY
    class
HAVING COUNT(student) >= 5;



SELECT
    activity_date AS day,
    COUNT(DISTINCT(user_id)) AS active_users
FROM
    Activity
WHERE
    DATEDIFF('2019-07-27', activity_date) < 30 AND DATEDIFF('2019-07-27', activity_date) >= 0
GROUP BY
    activity_date;

select name, population, area
from World
where area >= 3000000 or population >= 25000000


select product_id from Products
where low_fats = 'Y' && recyclable = 'Y';


select name from customer
where referee_id is null or referee_id != 2;


select
    employee_id,
    IF (employee_id % 2 == 1 && name not regexp '^M', salary, 0) as bonus
from
    employees
order by
    employee_id;


select distinct author_id as id
from Views
where author_id = viewer_id
order by id;


select tweet_id
from Tweets
where char_length(content) > 15;


select unique_id, name
from Employees left join EmployeeUNI on Employees.id = EmployeeUNI.id;

select product_name, year, price
from Sales left join Product on Sales.product_id = Product.product_id;


select customer_id, count(*) as count_no_trans
from Visits left join Transactions on Visits.visit_id = Transactions.visit_id
where Transactions.visit_id is null
GROUP BY customer_id

SELECT
    w1.id
FROM
    Weather w1
JOIN
    Weather w2
ON
    DATEDIFF(w1.recordDate, w2.recordDate) = 1 && w1.temperature > w2.temperature;


SELECT
    a.machine_id,
    ROUND(AVG(b.timestamp - a.timestamp), 3) as processing_time
FROM
    Activity a,
    Activity b
WHERE
    a.machine_id = b.machine_id
    AND a.process_id = b.process_id
    AND a.activity_type = 'start'
    AND b.activity_type = 'end'
GROUP BY machine_id;


SELECT
    name,
    bonus
FROM
    Employee LEFT JOIN Bonus on Employee.empId = Bonus.empId
WHERE
    bonus IS NULL OR bonus < 1000;

SELECT
    teacher_id,
    COUNT(DISTINCT subject_id) as cnt
FROM
    Teacher
GROUP BY
    teacher_id



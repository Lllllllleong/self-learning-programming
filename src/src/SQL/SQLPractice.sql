--
-- SQL Practice
--
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



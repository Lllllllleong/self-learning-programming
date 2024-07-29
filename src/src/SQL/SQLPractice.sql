--
-- SQL Practice
--

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
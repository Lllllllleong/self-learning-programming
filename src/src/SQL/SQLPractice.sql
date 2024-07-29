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





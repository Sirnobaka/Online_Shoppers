# Online shoppers, prediction of revenue

Пользователи онлайн магазинов посещают их с целью изучения ассортимента товаров, анализа рынка, а также для совершения покупок. Прибыль онлайн магазина напрямую зависит от того, какая доля покупателей, посетивших страницу с товаром, совершила сделку, купив этот товар. Чем выше конверсия посетителей в совершенные ими сделки, тем выше доход магазина. В связи с этим интересно изучить связь поведения пользователя и параметров его системы с вероятностью покупки в интернет магазине. В данной работе проведено изучение данных и построение модели предсказания совершения сделки пользователем на основе информации о его действиях на сайте интернет магазина.

## Набор данных

Данные содержат 12330 строк, в каждой из которых содержится информация об одной сессии пользователя. Данные были собраны в течение одного года, пользователи не повторяются. В данных содрежится 17 векторов предикторов и 1 вектор целевой переменной.

Целевой переменной является является величина `Revenue`, принимающая значения True или False в зависимости от того, была совершена сделка или нет. Данные не сбалансированы относительно этой переменной, пользователей совершивших покупку в них значительно меньше, чем не совершивших.

Предикторы `Administrative`, `Administrative Duration`, `Informational`, `Informational Duration`, `Product Related` и `Product Related Duration` обозначают количество страниц различных, посещённых пользователем в течение сессии, а также время, проведённое им на этих страницах.

`Bounce Rate` - доля пользователей, зашедших на сайт с данной страницы и покинувших его без совершенния дальнейших действий.
`Exit Rate` - доля просмотров данной страницы, в которых она была последней в сессии по отношению к полному числу просмотров страницы
`Page Value` feature represents the average value for a web page that a user visited before completing an e-commerce transaction.

`Special Day` показывате близость дня, в который проходила сессия к какому-либо празнику.

Набор данных также содержит информацию об операционной сиситеме, браузере, регионе пользователя, типе его трафика, а также классификации самого пользователся (постоянный, новый). Также в данных есть сведения о месяце транзакции и то, был это будний день или выходной.





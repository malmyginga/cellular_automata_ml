# cellular_automata_ml

Конструирование клеточных автоматов с использованием методов машинного обучения

КА - клеточный автомат
<table>
    <tr>
        <th>Класс</th>
        <th>Описание функций класса</th>
    </tr>
    <tr>
        <td>CellularAutomataGenerator2D</td>
        <td>Генератор двумерных клеточных автоматов, принимает на вход размерность автомата, количество автоматов,
            правило клеточного автомата (параметр-функция), радиус правила, параметр использования случайного начального
            состояния или использования переданного начального состояния.
        </td>
    </tr>
    <tr>
        <td>CellularAutomataTrainedGenerator</td>
        <td>Класс, который представляет интерфейс по работе с КА на основе обученной модели машинного обучения,
            реализует методы, которые позволяют на основе переданного состояния КА возвращать список последующих
            состояний КА, а также методы предсказания на передаваемых правилах КА.
        </td>
    </tr>
    <tr>
        <td>CellularModelTrainer</td>
        <td>Класс, в который передается модель и тренировочные данные для обучения, обученная этим классом модель
            передается в класс CellularAutomataTrainedGenerator для генерации следующих состояний КА
        </td>
    </tr>
    <tr>
        <td>CellularAutomataPlotDrawer</td>
        <td>Класс, который принимает считанные с помощью классов CellularAutomataGenerator2D и
            CellularAutomataTrainedGenerator метрики и на их основе строит описанные в секции с метриками графики
        </td>
    </tr>
    <tr>
        <td>CellularAutomataExperimentator</td>
        <td>Класс, который использует все описанные выше классы для проведения эксперимента на клеточном автомате,
            заданной размерности, с заданным количеством состояний, с заданными правилами и моделью, которая
            используется для обучения.
        </td>
    </tr>
</table>

## Пример проведения эксперимента:

```
# Epidemic cellular automata

# Импортируем необходимые пакеты
from sklearn.neural_network import MLPClassifier
from random import choices

# Задаем функцию перехода
# n - окрестность
# c - кортеж с координатами клеток
# t - текущая итерация клеточного автомата
def probability_rule(n, c, t):
    p = 0.1
    out = 1
    if n[1][1] == 0:
        out = choices([0,1], [p**(np.sum(n)), 1 - p**(np.sum(n))])[0]
    return out

# Задаем модель для обучения
clf = MLPClassifier(solver='adam',
                    alpha=1e-5, 
                    hidden_layer_sizes=(10, 10), 
                    max_iter = 10000)
                    
# Создаем класс для проведения эксперимента
experimentator_epidemic = CellularAutomataExperimentator(clf,
                                                50,
                                                50,
                                                2,
                                                probability_rule,
                                                1,
                                                random_initial_state=False,
                                                distinct_training=False)

# Запуск эксперимента для вероятностного клеточного автомата
# В accuracy_epidemic получим точность на последующих итерациях
accuracy_epidemic = experimentator_epidemic.experimentate_proba([1, 2], 10, 'Epidemic cellular automata', simple_initial_state=True)
```

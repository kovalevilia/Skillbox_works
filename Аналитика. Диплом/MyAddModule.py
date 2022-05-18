# данные
import pandas as pd
import numpy as np
# статистика
from statsmodels.stats.stattools import medcouple
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import mannwhitneyu
from scipy.stats import shapiro
from scipy.stats import probplot
from scipy.stats import norm
# визуализация
import matplotlib.pyplot as plt
import seaborn as sns
# прочие удобства
import re
from collections import namedtuple
import inspect
# colors
#731982 - purple
#00b956 - green



# функция исправления ответа на первый вопрос
def fix_q1(answer: str) -> str:
    """
    Исправляем ответ на 1й вопрос.
    Проверяем ответ на наличие и пытаемся достать числа из ответа. 
    Если в ответе только одно число и оно выходит за пределы отрезка [1, 10], принимаем за ответ либо 1 либо 10.
    Если в ответе несколько чисел и все не превышают 10, берем среднее, округленное по правилам арифметики
    Во всех остальных случаях оставляем ответ без изменений
    Использует бибилиотеки Pandas, NumPy, re.
    
    Parameters
    ----------
    answer : str
        Строка ответа из датафрема.
    
    Returns
    -------
    : str
        Строка с корректными ответом. 
    """
    if not pd.isna(answer):
        tmp = re.compile(r'\d+').findall(answer)
        if len(tmp) != 0:
            tmp = list(map(int, tmp))
            if len(tmp) == 1:

                if tmp[0] > 10:
                    return '10'
                elif tmp[0] < 1:
                    return '1'
                else:
                    return str(tmp[0])

            elif len(tmp) > 1 and all([i <= 10 for i in tmp]):
                return np.mean(tmp).round().astype(int).astype(str)
            else:
                return answer
        else:
            return answer


# функция исправления ответа на второй вопрос
def fix_q2(answer: str) -> (list, str):
    """
    Исправляем ответ на 2й вопрос.
    Проверяем ответ на наличие, разбиваем на цифры и оставляем только допустимые [1, 7].
    Выбрасываем возможные повторы и возвращаем список чисел в виде строк.
    Во всех остальных случаях помечаем ответ как отсутствующий.
    Использует бибилиотеки Pandas, NumPy, re.

    Parameters
    ----------
    answer : str
        Строка ответа из датафрема.

    Returns
    -------
    : list of strings, str
        Список строк с корректными ответами или строка 'NaN'.
    """
    correct_q2 = np.arange(1, 8).astype(str)
    if not pd.isna(answer):
        tmp = np.array([elem for elem in re.compile(r'\d').findall(answer) if elem in correct_q2])
        return np.unique(tmp).tolist()

    return 'NaN'


# функция, дополняющая стандартный метод pandas.DataFrame.describe
# добавляет дисперсию, межквартильный размах (по умолчанию 25%-75%), эксцесс и ассиметрию
def my_describe(dataframe):
    """
    Дополненный стандартный метод pandas.DataFrame.describe.
    Добавляет дисперсию, межквартильный размах (по умолчанию 25%-75%), эксцесс и ассиметрию.
    Возвращает DataFrame.
    Использует бибилиотеку Pandas.

    Parameters
    ----------
    dataframe : DataFrame
        Датафрем.

    Returns
    -------
    result : Series or DataFrame
        Суммарные статистики.
    """
    result = dataframe.describe()  # базовый отчет
    result.loc['variance'] = dataframe.var().apply(lambda x: format(x, 'f')).values  # дисперсия
    result.loc['IQR'] = (result.loc['75%'] - result.loc['25%']).values  # межквартильное расстояние
    result.loc['kurtosis'] = dataframe.kurtosis().values  # эксцесс
    result.loc['skew'] = dataframe.skew().values  # ассиметрия
    return result


# функция для проведения тестов на нормальность и визуализации
def normality_visual_test(dataframe, column, bins='auto'):
    """
    Визуальный и инструментальный тест на нормальность распределения. Визуализация распределения и выбросов. 
    Критерий Шапиро—Уилка + QQ-plot (scipy.stats.shapiro, scipy.stats.probplot).
    Использует библиотеки scipy, seaborn.
    
    Parameters
    ----------
    dataframe : DataFrame
        Датафрейм данных.
    column : label
        Название исследуемого столбца данных.
    bins : int or 'auto', default 'auto'
        Количество столбцов для гистограммы распределения, по умолчанию 'auto' (как в sns.histplot).
    
    Returns
    -------
    fig : Figure
        Фигура с графиками
    """
    # стиль
    sns.set_style('darkgrid', {'axes.facecolor': 'whitesmoke'})
    # фигура с тремя графиками
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 8))
    # выделяем данные для обработки
    tst = dataframe[column]
    # заголовок фигуры
    fig.suptitle('Анализ параметра ' + column, y=1.25, fontsize=34)
    # текст как часть фигуры, результат теста Шапиро
    stat, pval = shapiro(tst)
    plt.figtext(0.12, 1,
                'Результат теста Шапиро—Уилка:\nstatistic={}\npvalue={}\nNormal distribution: {}'\
                .format(stat, pval, 'no' if pval < 0.05 else 'yes'),
                fontsize=26)

    # QQ-plot тест на первый график
    probplot(tst, plot=axes[0])
    axes[0].set_title('qq.plot-тест на нормальное распределение', fontsize=20)
    axes[0].get_lines()[0].set_color('darkgreen')
    axes[0].get_lines()[0].set_markerfacecolor('#00b956')
    axes[0].get_lines()[0].set_markersize(10)
    axes[0].xaxis.label.set_size(14)
    axes[0].yaxis.label.set_size(14)

    # гистограмма на второй график
    axes[1].set_title('Распределение и линии процентилей', fontsize=20)
    sns.histplot(tst, bins=bins, ax=axes[1], color='#00b956', kde=True, alpha=1)

    # 25й, 50й, 75й, 99й процентили
    axes[1].axvline(x=np.percentile(tst, 25), color='gold', linestyle='--', label='25%')
    axes[1].axvline(x=np.percentile(tst, 50), color='blue', linestyle='-.', label='50%')
    axes[1].axvline(x=np.percentile(tst, 75), color='gold', linestyle='--', label='75%')
    axes[1].axvline(x=np.percentile(tst, 99), color='black', linestyle='-', label='99%')
    axes[1].axvline(x=tst.mean(), color='red', linestyle='-', label='Среднее')
    axes[1].legend(fontsize=16)
    axes[1].xaxis.label.set_size(14)
    axes[1].yaxis.label.set_size(14)

    # boxplot на третий график
    axes[2].set_title('Boxplot', fontsize=20)
    sns.boxplot(x=tst, ax=axes[2], color='#00b956', saturation=1)
    axes[2].set_xlabel(column, fontsize=14)

    plt.show()
    return fig


# функция для маркировки выбросов
def adj_boxplot_outliers(dataframe, columns, min_perc=25, max_perc=75, way='both'):
    """
    Функция определения выбросов по квантилям. Маркирует строки датафрейма с выбросами (1 - выброс, 0 - норма).
    По умолчанию определяет выбросы за двумя квартилями (25й и 75й квантили).
    Использует бибилиотеки Pandas, NumPy и statsmodels.
    
    Используется метод 'An Adjusted Boxplot for Skewed Distributions' для смещенных распределений:
    𝑄1-exp(−4𝑀)1.5*IQR and 𝑄3+exp(3𝑀)1.5*IQR, if 𝑀≥0
    𝑄1-exp(−3𝑀)1.5*IQR and 𝑄3+exp(4𝑀)1.5*IQR, if 𝑀<0
    где М - коэффициент асимметрии (statsmodels.stats.stattools.medcouple)
    
    Данный метод подходит в том числе для нормальных симметричных распределений, 
    поскольку в этом случае M≈0 и границы межквантильного диапазона расчитываются по стандартной формуле.
    
    Parameters
    ----------
    dataframe : DataFrame
        Входной датафрейм для определения выбросов.
    columns : list of labels
        Список названий исследуемых столбцов данных.
    min_perc : number, int or float, default 25
        Нижний квантиль (процент, от 0 до 100).
    max_perc : number, int or float, default 75
        Верхний квантиль (процент, от 0 до 100).
    way : string, 'left', 'right' or 'both', default 'both'
        Направление поиска выбросов: 
        'left' - только значения ниже границы min_perc, 
        'right' - только значения выше границы max_perc, 
        'both' (по умолчанию) - все выбросы за обеими границами
    
    Returns
    -------
    res_frame : DataFrame
        Датафрейм со значениями верхней и нижней границы диапазона данных, 
        а также с количеством выбросов в данных (в абсолютном и процентном выражении).
    """
    # делаем копию входных данных
    res_frame = dataframe.copy()
    # создаем колонку с маркером выброса, по умолчанию все наблюдения маркируем как нормальные
    res_frame['outliers'] = 0
    # маска для записи маркера выбросов
    outliers = set()

    # обрабатываем каждую колонку данных
    for col in columns:
        # нижний квантиль
        q1 = res_frame[col].quantile(q=min_perc / 100)
        # верхний квантиль
        q3 = res_frame[col].quantile(q=max_perc / 100)
        # межквантильный диапазон
        iqr = q3 - q1
        # коэффициент асимметрии
        m = medcouple(res_frame[col])

        # верхняя и нижняя границы диапазона по методу Adjusted Boxplot
        if m >= 0:
            lower_lim = q1 - np.exp(-4 * m) * 1.5 * iqr
            upper_lim = q3 + np.exp(3 * m) * 1.5 * iqr
        else:
            lower_lim = q1 - np.exp(-3 * m) * 1.5 * iqr
            upper_lim = q3 + np.exp(4 * m) * 1.5 * iqr

        # записываем маску-маркиратор в зависимости от направления поиска
        if way == 'both':
            outliers = outliers | set(res_frame[(res_frame[col] < lower_lim) |
                                                (res_frame[col] > upper_lim)].index)
        if way == 'left':
            outliers = outliers | set(res_frame[(res_frame[col] < lower_lim)].index)
        if way == 'right':
            outliers = outliers | set(res_frame[(res_frame[col] > upper_lim)].index)

    # значения с выбросами перемаркируем как выбросы
    res_frame.loc[res_frame.index.isin(outliers), 'outliers'] = 1
    print('Количество найденных выбросов: {}'.format(res_frame.index.isin(outliers).sum()))
    return res_frame


# функции для вычисления среднего и медианы по группам и визуализации на графике
def mean_median_comparison(dataframe, columns, grouper, visual=False):
    """
    Функция для подсчета среднего и медианы по столбцам датафрейма.
    Группирует датафрейм по параметру grouper, считает статистики и возвращает датафрейм с результатом.
    Визуализирует результат на графике.
    Использует бибилиотеки Pandas, seaborn.

    Parameters
    ----------
    dataframe : DataFrame
        Входной датафрейм.
    columns : list of labels
        Список названий исследуемых столбцов данных.
    grouper : string
        Столбец-признак для разделения наблюдейний на группы.
    visual : bool, default False
        Сигнал для отрисовки графика. По умолчанию график не рисуется.

    Returns
    -------
    res_frame : DataFrame
        Датафрейм со значениями среденго и медианы по каждому столбцу.
    fig : Figure
        Фигура с графиками. Если был указан соответствующий параметр.
    """
    # группируем и считаем результат
    test_calc = dataframe[columns + [grouper]].groupby(grouper).agg(['mean', 'median']).stack()
    test_calc.index.set_names('statistic', level=1, inplace=True)
    test_calc.reset_index(inplace=True)
    # если стоит флаг, строим график
    if visual:
        fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(8, 20))
        fig.suptitle('Сравнение средних и медиан для параметров', fontsize=18, y=1)
        # для каждого пронумерованного столбца из подсчитанного результата строим график
        for i, param in enumerate(columns):
            # строим соответствующий столбчатый график
            sns.barplot(data=test_calc, x=param, y='statistic', hue=grouper,
                        palette=['#00b956', '#731982'], ax=axes[i], alpha=1, saturation=1)
            # оформление
            axes[i].set(xlabel=None, ylabel=None)
            handles, _ = axes[i].get_legend_handles_labels()
            axes[i].set_title(label=param, fontsize=16)
            axes[i].legend_.remove()
            axes[i].tick_params(labelsize=12)
        # общее оформление графика и легенда
        fig.tight_layout()
        fig.legend(handles, ['Группа 0', 'Группа 1'], fontsize=12,
                   title='Группа абонентов', title_fontsize=16,
                   loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.show()

    return test_calc, fig


def mannwhitney_test(dataframe, columns, splitter, p_value=0.05):
    """
    Функция вычисления результатов теста Манна-Уитни: определяем наличие статистически значимых различий 
    в двух независимых выборках. На вход подаем датафрейм, список тестируемых параметров и пороговое значение 
    уровня значимости для принятия результатов теста. Делит данные по разделителю splitter и тестирует.
    Использует бибилиотеки Pandas, scipy.
    
    Parameters
    ----------
    dataframe : DataFrame
        Входной датафрейм.
    columns : list of labels
        Список названий колонок датафрейма (тестируемые признаки).
    splitter : string
        Столбец-признак для разделения наблюдейний на две группы.
    p_value : number, float, default 0.05
        Выбранные уровень значимости для интерпретации результатов теста. По умолчанию 5%.
    
    Returns
    -------
    result : DataFrame
        Датафрейм с результатами теста для каждого параметра.
    """
    # проверка на количество групп
    if dataframe[splitter].nunique() != 2:
        raise ValueError('Должно сравниваться две группы!')

    # шаблон результируещего датафрейма
    result = pd.DataFrame(columns=['U-statistic', 'p_value', 'significant_difference'],
                          index=columns)

    # для каждого параметра в данных
    for col in columns:
        # делим наблюдения на группы
        x = dataframe.loc[dataframe[splitter] == dataframe[splitter].unique()[0]][col]
        y = dataframe.loc[dataframe[splitter] == dataframe[splitter].unique()[1]][col]
        # тестируем
        test_res = mannwhitneyu(x, y)
        # пишем результат
        result.loc[col, 'U-statistic'] = test_res[0]
        result.loc[col, 'p_value'] = test_res[1]
        result.loc[col, 'significant_difference'] = 'yes' if (result.loc[col, 'p_value'] < 0.05) else 'no'

    # выводим размеры двух групп
    print("Количество наблюдений в группе '{}': {}".format(dataframe[splitter].unique()[0], x.size))
    print("Количество наблюдений в группе '{}': {}".format(dataframe[splitter].unique()[1], y.size))
    return result


# для печати имен переменных
def var_name_print(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


# бутстрэп
def bootstrap(group_1, group_2, output, statistic=np.mean, n_resamples=10000, confidence_level=0.95):
    """
    Функция бутстрэп для одного параметра.
    Использует бибилиотеки Pandas, NumPy, scipy.

    Parameters
    ----------
    group_1 : Series
        Входные данные первой группы.
    group_2 : Series
        Входные данные второй группы.
    output : list of labels, array-like
        Список названий колонок датафрейма с результатом.
    statistic : callable, function, default numpy.mean
        Статистика, для которой считается доверительный интервал.
    n_resamples : number, int, default 10000
        Количество повторений для бутстрэпа.
    confidence_level : number, float, default 0.05
        Уровень доверия для доверительного интервала. По умолчанию 95%.

    Returns
    -------
    boot_result : Series
        Результат бутстрэпа в виде серии, в качестве индекса используются заданные названия полей,
        полученные из output.
    """
    # шаблон результата
    boot_result = pd.Series(index=output, dtype=np.float64)
    # выбираем размер подвыборки
    sample_size = max(len(group_1), len(group_2))
    # контейнер для результатов вычислений
    boot_data = list()

    # запускаем цикл
    for _ in range(n_resamples):
        # формируем подвыборки с возвращением, делаем размер подвыборок одинаковым для сохранения дисперсии
        sample_1 = group_1.sample(sample_size, replace=True).values
        sample_2 = group_2.sample(sample_size, replace=True).values
        # считаем разницу статистик по подвыборкам и записываем в контейнер
        boot_data.append(statistic(sample_1) - statistic(sample_2))

    # считаем квантили интервала
    # уровень значимости 5% по умолчанию пополам
    l_quantile = (1 - confidence_level) / 2
    r_quantile = 1 - (1 - confidence_level) / 2
    quantiles = pd.Series(boot_data).quantile([l_quantile, r_quantile])

    # p-value
    p_1 = norm.cdf(x=0, loc=np.mean(boot_data), scale=np.std(boot_data))
    p_2 = norm.cdf(x=0, loc=-np.mean(boot_data), scale=np.std(boot_data))
    p_value = min(p_1, p_2) * 2

    # пишем результаты
    boot_result['statistic'] = statistic.__name__  # название рассчитаной статистики
    boot_result['confidence_level'] = confidence_level  # заданный уровень доверия
    boot_result['quantiles'] = [round(l_quantile, 4), round(r_quantile, 4)]  # границы доверительного интервала
    boot_result['group1_statistic'] = statistic(group_1)  # рассчитанная статистика для первой группы
    boot_result['group2_statistic'] = statistic(group_2)  # рассчитанная статистика для второй группы
    boot_result['difference'] = (statistic(group_1) - statistic(group_2))  # разница статитстик
    boot_result['confidence_interval'] = list(round(quantiles, 5))  # доверительный интервал для статистики

    # рассчитаное p-value и флаг результата
    boot_result['pval_H0'] = p_value
    boot_result['significant_diff'] = 'yes' if (boot_result['pval_H0'] < (1 - confidence_level)) else 'no'

    # данные, сформированные в бутстрэпе, на которых считалась статистика
    boot_result['boot_sample'] = boot_data

    return boot_result


# бутстрэп-пакет
def bootstrap_package(dataset_1, dataset_2, columns, statistic=np.mean, n_resamples=10000, confidence_level=0.95):
    """
    Функция бутстрэп.
    Использует бибилиотеки Pandas, NumPy, collections.

    Parameters
    ----------
    dataset_1 : DataFrame
        Входной датафрейм. Данные первой группы
    dataset_2 : DataFrame
        Входной датафрейм. Данные второй группы
    columns : list of labels, array-like
        Список названий колонок датафрейма (тестируемые признаки).
    statistic : callable, function, default numpy.mean
        Статистика, для которой считается доверительный интервал.
    n_resamples : number, int, default 10000
        Количество повторений для бутстрэпа.
    confidence_level : number, float, default 0.05
        Уровень доверия для доверительного интервала. По умолчанию 95%.

    Returns
    -------
    boot_result : BootstrapResult
        Результат бутстрэпа в виде именованного словаря.
        An object with attributes:
        summary : DataFrame
            Датафрейм с рассчитанными результатами.
        bootstrap_data : Series
            Данные, сформированные в бутстрэпе, на которых произвордились рассчеты.
    """
    # контейнер для результата функции
    BootstrapResult = namedtuple('BootstrapResult', ['summary', 'bootstrap_data'])
    # шаблон
    result = pd.DataFrame(columns=['significant_diff',
                                   'statistic',
                                   'confidence_level',
                                   'quantiles',
                                   'group1_statistic',
                                   'group2_statistic',
                                   'difference',
                                   'confidence_interval',
                                   'pval_H0',
                                   'boot_sample'],
                          index=columns)

    # для каждого параметра в данных
    for column in columns:
        # делим наблюдения на группы
        group_1 = dataset_1[column]
        group_2 = dataset_2[column]
        # считаем
        temp = bootstrap(group_1=group_1,
                         group_2=group_2,
                         output=result.columns,
                         statistic=statistic,
                         n_resamples=n_resamples,
                         confidence_level=confidence_level)
        # пишем рассчеты в шаблон строками
        result.loc[column, :] = temp

    print('Bootstrap results for null hypothesis: statistics in the group_1 and group_2 are equal')
    return BootstrapResult(summary=result.loc[:, ~result.columns.isin(['boot_sample'])],
                           bootstrap_data=result['boot_sample'])


# рисует столбчатые диаграммы для долей ответов на Q2 в двух группах
def q2_reasons_visual_prop(group1, group2):
    """
    Функция для визуализации долей ответов на Q2 в двух переданных группах.
    В первой группе посчитаны абоненты, указавшие обе причины,
    во второй группе - абоненты, указавшие только первую причину.
    Использует бибилиотеки Pandas, Seaborn.

    Parameters
    ----------
    group1 : Series
        Входные данные первой группы с количеством абонентов.
    group2 : Series
        Входные данные второй группы с количеством абонентов.

    Returns
    -------
    fig : Figure
        Фигура с графиками.
     """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 4), sharex=True)
    # доли ответов в первой группе
    sns.barplot(x=group1.values / (group1.values + group2.values) * 100,
                y=group1.index,
                ax=axes[0],
                palette=['#00b956', '#731982'], alpha=0.9, saturation=1)
    # доли ответов во второй группе
    sns.barplot(x=group2.values / (group1.values + group2.values) * 100,
                y=group2.index,
                ax=axes[1],
                palette=['#00b956', '#731982'], alpha=0.9, saturation=1)
    # подписи
    fig.supxlabel('% ответов')
    fig.supylabel('Группы причин', x=0)
    plt.show()

    return fig


# z-тест на равенство долей одного ответа в двух группах (для Q2).
def z_test(group1, group2, p_value=0.05):
    """
    Тест равенства долей одного параметра в двух выборках.
    Использует бибилиотеки Pandas, NumPy, statsmodels.

    Parameters
    ----------
    group1 : Series
        Входные данные первой группы.
    group2 : Series
        Входные данные первой группы.
    p_value : number, float, default 0.05
        Выбранные уровень значимости для интерпретации результатов теста. По умолчанию 5%.

    Returns
    -------
    result : DataFrame
        Результаты теста.
    """
    # шаблон результируещего датафрейма
    result = pd.DataFrame(columns=['Z-statistic', 'p_value', 'p_value < {}'.format(p_value)],
                          index=['value'])
    # считаем успехи
    count = np.array((group1[1], group2[1]))
    # все наблюдения
    nobs = np.array((group1.sum(), group2.sum()))
    # тест
    stat, pval = proportions_ztest(count, nobs)
    # пишем результаты
    result['Z-statistic'] = stat
    result['p_value'] = pval
    result.iloc[:, 2] = result['p_value'] < p_value

    return result

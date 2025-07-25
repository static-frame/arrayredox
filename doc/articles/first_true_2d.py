


import os
import sys
import timeit
import typing as tp
from itertools import repeat

from arraykit import first_true_2d as ak_first_true_2d
from arrayredox import first_true_2d as ar_first_true_2d
import arraykit as ak

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())



class ArrayProcessor:
    NAME = ''
    SORT = -1

    def __init__(self, array: np.ndarray):
        self.array = array

#-------------------------------------------------------------------------------
class AKFirstTrueAxis0Forward(ArrayProcessor):
    NAME = 'ak.first_true_2d(forward=True, axis=0)'
    SORT = 0

    def __call__(self):
        _ = ak_first_true_2d(self.array, forward=True, axis=0)

class AKFirstTrueAxis1Forward(ArrayProcessor):
    NAME = 'ak.first_true_2d(forward=True, axis=1)'
    SORT = 1

    def __call__(self):
        _ = ak_first_true_2d(self.array, forward=True, axis=1)

class AKFirstTrueAxis0Reverse(ArrayProcessor):
    NAME = 'ak.first_true_2d(forward=False, axis=0)'
    SORT = 2

    def __call__(self):
        _ = ak_first_true_2d(self.array, forward=False, axis=0)

class AKFirstTrueAxis1Reverse(ArrayProcessor):
    NAME = 'ak.first_true_2d(forward=False, axis=1)'
    SORT = 3

    def __call__(self):
        _ = ak_first_true_2d(self.array, forward=False, axis=1)



#-------------------------------------------------------------------------------
class ARFirstTrueAxis0Forward(ArrayProcessor):
    NAME = 'ar.first_true_2d(forward=True, axis=0)'
    SORT = 10

    def __call__(self):
        _ = ar_first_true_2d(self.array, forward=True, axis=0)

class ARFirstTrueAxis1Forward(ArrayProcessor):
    NAME = 'ar.first_true_2d(forward=True, axis=1)'
    SORT = 11

    def __call__(self):
        _ = ar_first_true_2d(self.array, forward=True, axis=1)

class ARFirstTrueAxis0Reverse(ArrayProcessor):
    NAME = 'ar.first_true_2d(forward=False, axis=0)'
    SORT = 12

    def __call__(self):
        _ = ar_first_true_2d(self.array, forward=False, axis=0)

class ARFirstTrueAxis1Reverse(ArrayProcessor):
    NAME = 'ar.first_true_2d(forward=False, axis=1)'
    SORT = 13

    def __call__(self):
        _ = ar_first_true_2d(self.array, forward=False, axis=1)


#-------------------------------------------------------------------------------


class NPNonZero(ArrayProcessor):
    NAME = 'np.nonzero()'
    SORT = 3

    def __call__(self):
        x, y = np.nonzero(self.array)
        # list(zip(x, y)) # simulate iteration


class NPArgMaxAxis0(ArrayProcessor):
    NAME = 'np.any(axis=0), np.argmax(axis=0)'
    SORT = 4

    def __call__(self):
        _ = ~np.any(self.array, axis=0)
        _ = np.argmax(self.array, axis=0)

class NPArgMaxAxis1(ArrayProcessor):
    NAME = 'np.any(axis=1), np.argmax(axis=1)'
    SORT = 4

    def __call__(self):
        _ = ~np.any(self.array, axis=1)
        _ = np.argmax(self.array, axis=1)



#-------------------------------------------------------------------------------
NUMBER = 100

def seconds_to_display(seconds: float) -> str:
    seconds /= NUMBER
    if seconds < 1e-4:
        return f'{seconds * 1e6: .1f} (µs)'
    if seconds < 1e-1:
        return f'{seconds * 1e3: .1f} (ms)'
    return f'{seconds: .1f} (s)'


def plot_performance(frame):
    fixture_total = len(frame['fixture'].unique())
    cat_total = len(frame['size'].unique())
    processor_total = len(frame['cls_processor'].unique())
    fig, axes = plt.subplots(cat_total, fixture_total)

    # cmap = plt.get_cmap('terrain')
    cmap = plt.get_cmap('plasma')

    color = cmap(np.arange(processor_total) / processor_total)

    # category is the size of the array
    for cat_count, (cat_label, cat) in enumerate(frame.groupby('size')):
        for fixture_count, (fixture_label, fixture) in enumerate(
                cat.groupby('fixture')):
            ax = axes[cat_count][fixture_count]

            # set order
            fixture['sort'] = [f.SORT for f in fixture['cls_processor']]
            fixture = fixture.sort_values('sort')

            results = fixture['time'].values.tolist()
            names = [cls.NAME for cls in fixture['cls_processor']]
            # x = np.arange(len(results))
            names_display = names
            post = ax.bar(names_display, results, color=color)

            density, position = fixture_label.split('-')
            # cat_label is the size of the array
            title = f'{cat_label:.0e}\n{FixtureFactory.DENSITY_TO_DISPLAY[density]}\n{FixtureFactory.POSITION_TO_DISPLAY[position]}'

            ax.set_title(title, fontsize=6)
            ax.set_box_aspect(0.75) # makes taller tan wide
            time_max = fixture['time'].max()
            ax.set_yticks([0, time_max * 0.5, time_max])
            ax.set_yticklabels(['',
                    seconds_to_display(time_max * .5),
                    seconds_to_display(time_max),
                    ], fontsize=6)
            # ax.set_xticks(x, names_display, rotation='vertical')
            ax.tick_params(
                    axis='x',
                    which='both',
                    bottom=False,
                    top=False,
                    labelbottom=False,
                    )

    fig.set_size_inches(9, 3.5) # width, height
    fig.legend(post, names_display, loc='center right', fontsize=6)
    # horizontal, vertical
    fig.text(.05, .96, f'ak_first_true_2d() Performance: {NUMBER} Iterations', fontsize=10)
    fig.text(.05, .90, get_versions(), fontsize=6)

    fp = '/tmp/first_true.png'
    plt.subplots_adjust(
            left=0.075,
            bottom=0.05,
            right=0.75,
            top=0.85,
            wspace=1, # width
            hspace=0.1,
            )
    # plt.rcParams.update({'font.size': 22})
    plt.savefig(fp, dpi=300)

    if sys.platform.startswith('linux'):
        os.system(f'eog {fp}&')
    else:
        os.system(f'open {fp}')


#-------------------------------------------------------------------------------

class FixtureFactory:
    NAME = ''

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        return np.full(size, False, dtype=bool)

    def _get_array_filled(
            size: int,
            start_third: int, # 1 or 2
            density: float, # less than 1
            ) -> np.ndarray:
        a = FixtureFactory.get_array(size)
        count = size * density
        start = int(len(a) * (start_third/3))
        length = len(a) - start
        step = int(length / count)
        fill = np.arange(start, len(a), step)
        a[fill] = True
        return a

    @classmethod
    def get_label_array(cls, size: int) -> tp.Tuple[str, np.ndarray]:
        array = cls.get_array(size)
        return cls.NAME, array

    DENSITY_TO_DISPLAY = {
        'single': '1 True',
        'tenth': '10% True',
        'third': '33% True',
    }

    POSITION_TO_DISPLAY = {
        'first_third': 'Fill 1/3 to End',
        'second_third': 'Fill 2/3 to End',
    }


class FFSingleFirstThird(FixtureFactory):
    NAME = 'single-first_third'

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        a = FixtureFactory.get_array(size)
        a[int(len(a) * (1/3))] = True
        return a

class FFSingleSecondThird(FixtureFactory):
    NAME = 'single-second_third'

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        a = FixtureFactory.get_array(size)
        a[int(len(a) * (2/3))] = True
        return a


class FFTenthPostFirstThird(FixtureFactory):
    NAME = 'tenth-first_third'

    @classmethod
    def get_array(cls, size: int) -> np.ndarray:
        return cls._get_array_filled(size, start_third=1, density=.1)


class FFTenthPostSecondThird(FixtureFactory):
    NAME = 'tenth-second_third'

    @classmethod
    def get_array(cls, size: int) -> np.ndarray:
        return cls._get_array_filled(size, start_third=2, density=.1)


class FFThirdPostFirstThird(FixtureFactory):
    NAME = 'third-first_third'

    @classmethod
    def get_array(cls, size: int) -> np.ndarray:
        return cls._get_array_filled(size, start_third=1, density=1/3)


class FFThirdPostSecondThird(FixtureFactory):
    NAME = 'third-second_third'

    @classmethod
    def get_array(cls, size: int) -> np.ndarray:
        return cls._get_array_filled(size, start_third=2, density=1/3)


def get_versions() -> str:
    import platform
    return f'OS: {platform.system()} / ArrayKit: {ak.__version__} / NumPy: {np.__version__}\n'


CLS_PROCESSOR = (
    AKFirstTrueAxis0Forward,
    AKFirstTrueAxis1Forward,
    AKFirstTrueAxis0Reverse,
    AKFirstTrueAxis1Reverse,

    ARFirstTrueAxis0Forward,
    ARFirstTrueAxis1Forward,
    ARFirstTrueAxis0Reverse,
    ARFirstTrueAxis1Reverse,

    # NPNonZero,
    # NPArgMaxAxis0,
    # NPArgMaxAxis1
    )

CLS_FF = (
    FFSingleFirstThird,
    FFSingleSecondThird,
    FFTenthPostFirstThird,
    FFTenthPostSecondThird,
    FFThirdPostFirstThird,
    FFThirdPostSecondThird,
)


def run_test():
    records = []
    for size in (100_000, 1_000_000, 10_000_000):
        for ff in CLS_FF:
            fixture_label, fixture = ff.get_label_array(size)
            # TEMP
            fixture = fixture.reshape(size // 10, 10)

            for cls in CLS_PROCESSOR:
                runner = cls(fixture)

                record = [cls, NUMBER, fixture_label, size]
                print(record)
                try:
                    result = timeit.timeit(
                            f'runner()',
                            globals=locals(),
                            number=NUMBER)
                except OSError:
                    result = np.nan
                finally:
                    pass
                record.append(result)
                records.append(record)

    f = pd.DataFrame.from_records(records,
            columns=('cls_processor', 'number', 'fixture', 'size', 'time')
            )
    print(f)
    plot_performance(f)

if __name__ == '__main__':

    run_test()



